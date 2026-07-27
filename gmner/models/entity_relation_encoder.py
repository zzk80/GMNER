"""
Entity Relation Encoder for Multi-entity Scene

功能: 编码实体间的关系（spatial, semantic, context）
用于 Instance-Aware Region Matching

作者: Claude (Kiro AI Assistant)
日期: 2026-07-27
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class EntityRelationEncoder(nn.Module):
    """
    编码多实体场景中实体间的关系

    三种关系类型:
    1. Spatial: 实体在文本中的位置关系
    2. Semantic: 实体的语义关系（类型、共指等）
    3. Context: 实体共享的上下文信息
    """

    def __init__(
        self,
        entity_dim: int = 768,
        relation_dim: int = 128,
        num_relation_types: int = 8,
        use_position_encoding: bool = True,
        dropout: float = 0.1,
    ):
        """
        Args:
            entity_dim: 实体表示维度
            relation_dim: 关系表示维度
            num_relation_types: 关系类型数量
            use_position_encoding: 是否使用位置编码
            dropout: Dropout 比例
        """
        super().__init__()

        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        self.use_position_encoding = use_position_encoding

        # Spatial relation encoder
        self.spatial_encoder = SpatialRelationEncoder(
            relation_dim=relation_dim,
            use_position_encoding=use_position_encoding,
        )

        # Semantic relation encoder
        self.semantic_encoder = SemanticRelationEncoder(
            entity_dim=entity_dim,
            relation_dim=relation_dim,
            num_types=4,  # PER, LOC, ORG, OTHER
        )

        # Context relation encoder
        self.context_encoder = ContextRelationEncoder(
            entity_dim=entity_dim,
            relation_dim=relation_dim,
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(relation_dim * 3, relation_dim),
            nn.LayerNorm(relation_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Relation type classifier (可选)
        self.relation_classifier = nn.Linear(relation_dim, num_relation_types)

    def forward(
        self,
        entity_reprs: torch.Tensor,
        entity_positions: torch.Tensor,
        entity_types: torch.Tensor,
        entity_mask: torch.Tensor,
        text_reprs: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        编码实体间关系

        Args:
            entity_reprs: [batch, max_entities, entity_dim] 实体表示
            entity_positions: [batch, max_entities, 2] 实体位置 (start, end)
            entity_types: [batch, max_entities, num_types] 实体类型 logits
            entity_mask: [batch, max_entities] 实体有效性 mask
            text_reprs: [batch, seq_len, entity_dim] 文本表示（可选）

        Returns:
            outputs: {
                'relation_matrix': [batch, max_entities, max_entities, relation_dim],
                'relation_types': [batch, max_entities, max_entities, num_relation_types],
                'adjacency': [batch, max_entities, max_entities] (0/1),
            }
        """
        batch_size, max_entities = entity_reprs.size(0), entity_reprs.size(1)

        # 1. Encode spatial relations
        spatial_relations = self.spatial_encoder(
            entity_positions=entity_positions,
            entity_mask=entity_mask,
        )  # [batch, max_entities, max_entities, relation_dim]

        # 2. Encode semantic relations
        semantic_relations = self.semantic_encoder(
            entity_reprs=entity_reprs,
            entity_types=entity_types,
            entity_mask=entity_mask,
        )  # [batch, max_entities, max_entities, relation_dim]

        # 3. Encode context relations
        context_relations = self.context_encoder(
            entity_reprs=entity_reprs,
            entity_positions=entity_positions,
            entity_mask=entity_mask,
            text_reprs=text_reprs,
        )  # [batch, max_entities, max_entities, relation_dim]

        # 4. Fuse all relations
        all_relations = torch.cat([
            spatial_relations,
            semantic_relations,
            context_relations,
        ], dim=-1)  # [batch, max_entities, max_entities, relation_dim * 3]

        relation_matrix = self.fusion(all_relations)  # [batch, M, M, relation_dim]

        # 5. Classify relation types (optional)
        relation_types = self.relation_classifier(relation_matrix)

        # 6. Compute adjacency matrix
        # 如果两个实体有关系（距离近、类型相关等），则为 1
        adjacency = self._compute_adjacency(
            spatial_relations=spatial_relations,
            semantic_relations=semantic_relations,
            entity_mask=entity_mask,
        )

        return {
            'relation_matrix': relation_matrix,
            'relation_types': relation_types,
            'adjacency': adjacency,
            'spatial_relations': spatial_relations,
            'semantic_relations': semantic_relations,
            'context_relations': context_relations,
        }

    def _compute_adjacency(
        self,
        spatial_relations: torch.Tensor,
        semantic_relations: torch.Tensor,
        entity_mask: torch.Tensor,
    ) -> torch.Tensor:
        """计算邻接矩阵"""
        # 简化版本：基于 L2 距离判断
        spatial_dist = torch.norm(spatial_relations, dim=-1)  # [batch, M, M]
        semantic_dist = torch.norm(semantic_relations, dim=-1)

        # 归一化
        spatial_dist = spatial_dist / (spatial_dist.max(dim=-1, keepdim=True)[0] + 1e-8)
        semantic_dist = semantic_dist / (semantic_dist.max(dim=-1, keepdim=True)[0] + 1e-8)

        # 组合距离
        combined_dist = 0.5 * spatial_dist + 0.5 * semantic_dist

        # 阈值化
        adjacency = (combined_dist < 0.5).float()

        # 应用 mask
        mask_2d = entity_mask.unsqueeze(1) * entity_mask.unsqueeze(2)
        adjacency = adjacency * mask_2d

        return adjacency


class SpatialRelationEncoder(nn.Module):
    """编码空间关系（实体在文本中的位置）"""

    def __init__(self, relation_dim: int = 128, use_position_encoding: bool = True):
        super().__init__()
        self.relation_dim = relation_dim
        self.use_position_encoding = use_position_encoding

        # Position embedding
        if use_position_encoding:
            self.position_encoder = nn.Sequential(
                nn.Linear(4, relation_dim // 2),  # [start_i, end_i, start_j, end_j]
                nn.LayerNorm(relation_dim // 2),
                nn.GELU(),
            )

        # Relative position features
        self.relative_encoder = nn.Sequential(
            nn.Linear(3, relation_dim // 2),  # [distance, overlap, order]
            nn.LayerNorm(relation_dim // 2),
            nn.GELU(),
        )

        self.fusion = nn.Linear(relation_dim, relation_dim)

    def forward(
        self,
        entity_positions: torch.Tensor,
        entity_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            entity_positions: [batch, max_entities, 2] (start, end)
            entity_mask: [batch, max_entities]

        Returns:
            spatial_relations: [batch, max_entities, max_entities, relation_dim]
        """
        batch_size, max_entities = entity_positions.size(0), entity_positions.size(1)

        # Expand to pairwise
        pos_i = entity_positions.unsqueeze(2).expand(-1, -1, max_entities, -1)
        pos_j = entity_positions.unsqueeze(1).expand(-1, max_entities, -1, -1)

        # Absolute positions
        if self.use_position_encoding:
            abs_pos = torch.cat([pos_i, pos_j], dim=-1)  # [B, M, M, 4]
            abs_feat = self.position_encoder(abs_pos.float())
        else:
            abs_feat = torch.zeros(batch_size, max_entities, max_entities,
                                  self.relation_dim // 2, device=entity_positions.device)

        # Relative positions
        start_i, end_i = pos_i[..., 0], pos_i[..., 1]
        start_j, end_j = pos_j[..., 0], pos_j[..., 1]

        # Distance (中心点距离)
        center_i = (start_i + end_i) / 2.0
        center_j = (start_j + end_j) / 2.0
        distance = torch.abs(center_i - center_j).unsqueeze(-1)  # [B, M, M, 1]

        # Overlap (重叠长度)
        overlap_start = torch.max(start_i, start_j)
        overlap_end = torch.min(end_i, end_j)
        overlap = torch.clamp(overlap_end - overlap_start, min=0).unsqueeze(-1)

        # Order (相对顺序: -1/0/1)
        order = torch.sign(center_i - center_j).unsqueeze(-1)

        rel_feat_input = torch.cat([distance, overlap, order], dim=-1)
        rel_feat = self.relative_encoder(rel_feat_input.float())

        # Fuse
        spatial_relations = self.fusion(torch.cat([abs_feat, rel_feat], dim=-1))

        return spatial_relations


class SemanticRelationEncoder(nn.Module):
    """编码语义关系（类型、共指等）"""

    def __init__(self, entity_dim: int = 768, relation_dim: int = 128, num_types: int = 4):
        super().__init__()

        # Type similarity
        self.type_encoder = nn.Sequential(
            nn.Linear(num_types * 2, relation_dim // 2),
            nn.LayerNorm(relation_dim // 2),
            nn.GELU(),
        )

        # Entity representation similarity
        self.repr_encoder = nn.Sequential(
            nn.Linear(entity_dim * 2, relation_dim // 2),
            nn.LayerNorm(relation_dim // 2),
            nn.GELU(),
        )

        self.fusion = nn.Linear(relation_dim, relation_dim)

    def forward(
        self,
        entity_reprs: torch.Tensor,
        entity_types: torch.Tensor,
        entity_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            entity_reprs: [batch, max_entities, entity_dim]
            entity_types: [batch, max_entities, num_types] (logits)
            entity_mask: [batch, max_entities]

        Returns:
            semantic_relations: [batch, max_entities, max_entities, relation_dim]
        """
        batch_size, max_entities = entity_reprs.size(0), entity_reprs.size(1)

        # Normalize type logits
        type_probs = F.softmax(entity_types, dim=-1)  # [B, M, num_types]

        # Pairwise type features
        type_i = type_probs.unsqueeze(2).expand(-1, -1, max_entities, -1)
        type_j = type_probs.unsqueeze(1).expand(-1, max_entities, -1, -1)
        type_pair = torch.cat([type_i, type_j], dim=-1)  # [B, M, M, num_types*2]
        type_feat = self.type_encoder(type_pair)

        # Pairwise entity representations
        repr_i = entity_reprs.unsqueeze(2).expand(-1, -1, max_entities, -1)
        repr_j = entity_reprs.unsqueeze(1).expand(-1, max_entities, -1, -1)
        repr_pair = torch.cat([repr_i, repr_j], dim=-1)  # [B, M, M, entity_dim*2]
        repr_feat = self.repr_encoder(repr_pair)

        # Fuse
        semantic_relations = self.fusion(torch.cat([type_feat, repr_feat], dim=-1))

        return semantic_relations


class ContextRelationEncoder(nn.Module):
    """编码上下文关系（共享的上下文信息）"""

    def __init__(self, entity_dim: int = 768, relation_dim: int = 128):
        super().__init__()

        self.context_encoder = nn.Sequential(
            nn.Linear(entity_dim, relation_dim),
            nn.LayerNorm(relation_dim),
            nn.GELU(),
        )

    def forward(
        self,
        entity_reprs: torch.Tensor,
        entity_positions: torch.Tensor,
        entity_mask: torch.Tensor,
        text_reprs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            entity_reprs: [batch, max_entities, entity_dim]
            entity_positions: [batch, max_entities, 2]
            entity_mask: [batch, max_entities]
            text_reprs: [batch, seq_len, entity_dim] (可选)

        Returns:
            context_relations: [batch, max_entities, max_entities, relation_dim]
        """
        batch_size, max_entities = entity_reprs.size(0), entity_reprs.size(1)

        # 简化版本：使用实体表示的点积作为上下文相似度
        # [B, M, entity_dim] @ [B, entity_dim, M] -> [B, M, M]
        similarity = torch.bmm(entity_reprs, entity_reprs.transpose(1, 2))

        # Normalize
        similarity = similarity / (self.entity_dim ** 0.5)
        similarity = F.softmax(similarity, dim=-1)

        # Expand to relation_dim
        context_relations = self.context_encoder(
            similarity.unsqueeze(-1).expand(-1, -1, -1, entity_reprs.size(-1))
            * entity_reprs.unsqueeze(1)
        )

        return context_relations


def create_entity_relation_encoder(config: Dict) -> EntityRelationEncoder:
    """创建 EntityRelationEncoder 实例"""
    return EntityRelationEncoder(
        entity_dim=config.get('entity_dim', 768),
        relation_dim=config.get('relation_dim', 128),
        num_relation_types=config.get('num_relation_types', 8),
        use_position_encoding=config.get('use_position_encoding', True),
        dropout=config.get('dropout', 0.1),
    )
