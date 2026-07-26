"""
Scene Analyzer for GMNER Instance-Aware Framework

功能: 将记录分类为 single-entity 或 multi-entity 场景
输入: 文本特征 (不使用 gold entity count - 那是作弊)
输出: scene_type ∈ {single, multi}

作者: Claude (Kiro AI Assistant)
日期: 2026-07-27
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class SceneAnalyzer(nn.Module):
    """
    场景分析器 - 判断记录是否包含多个实体

    设计原则:
    1. 不使用 gold entity count (那是作弊)
    2. 只使用文本特征和预测的 span 信息
    3. 轻量级模型，不增加过多计算开销
    """

    def __init__(
        self,
        text_dim: int = 768,
        hidden_dim: int = 128,
        num_classes: int = 2,  # single vs multi
        dropout: float = 0.2,
        use_span_features: bool = True,
    ):
        """
        Args:
            text_dim: 文本编码维度 (RoBERTa: 768)
            hidden_dim: 隐藏层维度
            num_classes: 分类数 (2: single/multi)
            dropout: Dropout 比例
            use_span_features: 是否使用预测的 span 特征
        """
        super().__init__()

        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.use_span_features = use_span_features

        # 文本全局表示编码器
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 统计特征编码器
        # 特征: [text_length, num_predicted_spans, span_density,
        #        avg_span_length, type_diversity, ...]
        self.stat_feature_dim = 8
        self.stat_encoder = nn.Sequential(
            nn.Linear(self.stat_feature_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Span 特征聚合器 (可选)
        if use_span_features:
            self.span_aggregator = nn.Sequential(
                nn.Linear(text_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
            )
            fusion_dim = hidden_dim + hidden_dim // 2 + hidden_dim // 2
        else:
            self.span_aggregator = None
            fusion_dim = hidden_dim + hidden_dim // 2

        # 融合与分类
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def extract_statistical_features(
        self,
        text_lengths: torch.Tensor,
        num_predicted_spans: torch.Tensor,
        span_lengths: torch.Tensor,
        type_logits: torch.Tensor,
        span_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        提取统计特征

        Args:
            text_lengths: [batch_size] 文本长度
            num_predicted_spans: [batch_size] 预测的 span 数量
            span_lengths: [batch_size, max_spans] 每个 span 的长度
            type_logits: [batch_size, max_spans, num_types] 类型预测 logits
            span_mask: [batch_size, max_spans] span 有效性 mask

        Returns:
            features: [batch_size, stat_feature_dim]
        """
        batch_size = text_lengths.size(0)
        device = text_lengths.device

        # 1. 文本长度 (归一化到 [0, 1])
        norm_text_length = text_lengths.float() / 100.0  # 假设最大 100 tokens

        # 2. Span 数量 (归一化)
        norm_span_count = num_predicted_spans.float() / 10.0  # 假设最大 10 spans

        # 3. Span 密度
        span_density = num_predicted_spans.float() / (text_lengths.float() + 1e-8)

        # 4. 平均 span 长度
        valid_span_lengths = span_lengths.float() * span_mask.float()
        avg_span_length = valid_span_lengths.sum(dim=1) / (num_predicted_spans.float() + 1e-8)
        avg_span_length = avg_span_length / 10.0  # 归一化

        # 5. 类型多样性 (使用 entropy)
        # 计算每条记录的类型分布
        type_probs = F.softmax(type_logits, dim=-1)  # [batch, spans, types]
        # 聚合到记录级别
        record_type_dist = (type_probs * span_mask.unsqueeze(-1)).sum(dim=1)  # [batch, types]
        record_type_dist = record_type_dist / (record_type_dist.sum(dim=-1, keepdim=True) + 1e-8)
        # 计算 entropy
        type_entropy = -(record_type_dist * torch.log(record_type_dist + 1e-8)).sum(dim=-1)
        type_entropy = type_entropy / 2.0  # 归一化 (假设 4 类，max entropy ~1.4)

        # 6. Span 间距特征 (如果 spans 很分散，可能是多实体)
        # 简化版本：计算 span 位置的标准差
        span_positions = torch.arange(span_lengths.size(1), device=device).float()
        span_positions = span_positions.unsqueeze(0).expand(batch_size, -1)
        weighted_positions = (span_positions * span_mask.float()).sum(dim=1) / (num_predicted_spans.float() + 1e-8)
        position_std = torch.sqrt(
            ((span_positions - weighted_positions.unsqueeze(1)).pow(2) * span_mask.float()).sum(dim=1)
            / (num_predicted_spans.float() + 1e-8)
        )
        position_std = position_std / 10.0  # 归一化

        # 7. 最大类型置信度
        max_type_conf, _ = type_probs.max(dim=-1)  # [batch, spans]
        avg_max_conf = (max_type_conf * span_mask.float()).sum(dim=1) / (num_predicted_spans.float() + 1e-8)

        # 8. Span 数量的二次项 (捕捉非线性)
        span_count_squared = (norm_span_count ** 2).clamp(0, 1)

        # 拼接所有特征
        features = torch.stack([
            norm_text_length,
            norm_span_count,
            span_density,
            avg_span_length,
            type_entropy,
            position_std,
            avg_max_conf,
            span_count_squared,
        ], dim=1)  # [batch, 8]

        return features

    def forward(
        self,
        text_repr: torch.Tensor,
        text_lengths: torch.Tensor,
        span_reprs: torch.Tensor = None,
        span_mask: torch.Tensor = None,
        span_lengths: torch.Tensor = None,
        type_logits: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            text_repr: [batch, text_dim] 文本全局表示 (如 [CLS] 或 mean pooling)
            text_lengths: [batch] 文本长度
            span_reprs: [batch, max_spans, text_dim] span 表示 (可选)
            span_mask: [batch, max_spans] span 有效性 mask
            span_lengths: [batch, max_spans] 每个 span 的长度
            type_logits: [batch, max_spans, num_types] 类型预测 logits

        Returns:
            outputs: {
                'logits': [batch, num_classes] 分类 logits,
                'probs': [batch, num_classes] 分类概率,
                'scene_type': [batch] 预测的场景类型 (0=single, 1=multi)
            }
        """
        batch_size = text_repr.size(0)

        # 1. 编码文本全局表示
        text_feat = self.text_encoder(text_repr)  # [batch, hidden]

        # 2. 提取统计特征
        num_predicted_spans = span_mask.sum(dim=1) if span_mask is not None else torch.zeros(batch_size, device=text_repr.device)

        # 如果没有提供详细特征，使用简化版本
        if span_lengths is None:
            span_lengths = torch.ones_like(span_mask).float() * 3.0  # 假设平均长度为 3
        if type_logits is None:
            # 创建 dummy type logits
            num_types = 4
            type_logits = torch.zeros(batch_size, span_mask.size(1), num_types, device=text_repr.device)

        stat_features = self.extract_statistical_features(
            text_lengths=text_lengths,
            num_predicted_spans=num_predicted_spans,
            span_lengths=span_lengths,
            type_logits=type_logits,
            span_mask=span_mask,
        )
        stat_feat = self.stat_encoder(stat_features)  # [batch, hidden//2]

        # 3. 聚合 span 特征 (可选)
        if self.use_span_features and span_reprs is not None:
            # Mean pooling over valid spans
            masked_span_reprs = span_reprs * span_mask.unsqueeze(-1)
            span_feat = masked_span_reprs.sum(dim=1) / (num_predicted_spans.unsqueeze(-1).float() + 1e-8)
            span_feat = self.span_aggregator(span_feat)  # [batch, hidden//2]

            # 融合所有特征
            fused_feat = torch.cat([text_feat, stat_feat, span_feat], dim=1)
        else:
            fused_feat = torch.cat([text_feat, stat_feat], dim=1)

        # 4. 分类
        logits = self.classifier(fused_feat)  # [batch, num_classes]
        probs = F.softmax(logits, dim=-1)
        scene_type = torch.argmax(logits, dim=-1)  # 0=single, 1=multi

        return {
            'logits': logits,
            'probs': probs,
            'scene_type': scene_type,
            'confidence': probs.max(dim=-1)[0],
        }


def create_scene_analyzer(config: Dict) -> SceneAnalyzer:
    """创建 Scene Analyzer 实例"""
    return SceneAnalyzer(
        text_dim=config.get('text_dim', 768),
        hidden_dim=config.get('hidden_dim', 128),
        num_classes=config.get('num_classes', 2),
        dropout=config.get('dropout', 0.2),
        use_span_features=config.get('use_span_features', True),
    )
