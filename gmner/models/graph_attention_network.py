"""
Graph Attention Network for Entity Relation Aggregation

功能: 使用图注意力机制聚合实体间的关系信息
用于 Multi-entity Branch

作者: Claude (Kiro AI Assistant)
日期: 2026-07-27
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class GraphAttentionLayer(nn.Module):
    """
    单层图注意力
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        concat: bool = True,
    ):
        """
        Args:
            in_dim: 输入特征维度
            out_dim: 输出特征维度（每个 head）
            num_heads: 注意力头数
            dropout: Dropout 比例
            concat: 是否拼接多头输出（否则取平均）
        """
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.concat = concat

        # Multi-head attention
        self.W = nn.ModuleList([
            nn.Linear(in_dim, out_dim, bias=False)
            for _ in range(num_heads)
        ])

        self.a = nn.ModuleList([
            nn.Linear(2 * out_dim, 1, bias=False)
            for _ in range(num_heads)
        ])

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            node_features: [batch, num_nodes, in_dim]
            adjacency: [batch, num_nodes, num_nodes] (0/1 or weighted)
            edge_features: [batch, num_nodes, num_nodes, edge_dim] (可选)

        Returns:
            output: [batch, num_nodes, out_dim * num_heads] (concat=True)
                    or [batch, num_nodes, out_dim] (concat=False)
        """
        batch_size, num_nodes = node_features.size(0), node_features.size(1)

        outputs = []
        for head in range(self.num_heads):
            # Linear transformation
            h = self.W[head](node_features)  # [batch, num_nodes, out_dim]

            # Compute attention coefficients
            # [batch, num_nodes, 1, out_dim] || [batch, 1, num_nodes, out_dim]
            h_i = h.unsqueeze(2).expand(-1, -1, num_nodes, -1)
            h_j = h.unsqueeze(1).expand(-1, num_nodes, -1, -1)

            # Concatenate and compute attention
            cat_features = torch.cat([h_i, h_j], dim=-1)  # [B, N, N, 2*out_dim]
            e = self.leakyrelu(self.a[head](cat_features).squeeze(-1))  # [B, N, N]

            # Mask attention with adjacency
            # 将不相邻的节点的注意力设为 -inf
            mask = (adjacency == 0)
            e = e.masked_fill(mask, float('-inf'))

            # Softmax
            alpha = F.softmax(e, dim=-1)  # [batch, num_nodes, num_nodes]
            alpha = self.dropout(alpha)

            # Aggregate
            h_prime = torch.bmm(alpha, h)  # [batch, num_nodes, out_dim]

            outputs.append(h_prime)

        # Combine heads
        if self.concat:
            output = torch.cat(outputs, dim=-1)  # [B, N, out_dim * num_heads]
        else:
            output = torch.mean(torch.stack(outputs), dim=0)  # [B, N, out_dim]

        return output


class GraphAttentionNetwork(nn.Module):
    """
    多层图注意力网络
    """

    def __init__(
        self,
        node_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_residual: bool = True,
    ):
        """
        Args:
            node_dim: 节点特征维度
            hidden_dim: 隐藏层维度
            num_layers: GAT 层数
            num_heads: 每层的注意力头数
            dropout: Dropout 比例
            use_residual: 是否使用残差连接
        """
        super().__init__()

        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_residual = use_residual

        # Input projection
        self.input_proj = nn.Linear(node_dim, hidden_dim)

        # GAT layers
        self.gat_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for i in range(num_layers):
            # 前几层拼接多头，最后一层取平均
            concat = (i < num_layers - 1)
            in_dim = hidden_dim if i == 0 else hidden_dim * num_heads
            out_dim = hidden_dim

            self.gat_layers.append(
                GraphAttentionLayer(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    concat=concat,
                )
            )

            # Layer norm
            norm_dim = hidden_dim * num_heads if concat else hidden_dim
            self.layer_norms.append(nn.LayerNorm(norm_dim))

        # Output projection
        final_dim = hidden_dim * num_heads if num_layers > 1 else hidden_dim
        self.output_proj = nn.Linear(final_dim, node_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            node_features: [batch, num_nodes, node_dim]
            adjacency: [batch, num_nodes, num_nodes]
            node_mask: [batch, num_nodes] (可选)

        Returns:
            outputs: {
                'node_features': [batch, num_nodes, node_dim],
                'attention_weights': List of [batch, num_nodes, num_nodes],
            }
        """
        # Input projection
        h = self.input_proj(node_features)
        h = F.elu(h)

        # Store attention weights
        attention_weights = []

        # Apply GAT layers
        for i, (gat_layer, layer_norm) in enumerate(zip(self.gat_layers, self.layer_norms)):
            h_prev = h

            # GAT layer
            h = gat_layer(h, adjacency)

            # Layer norm
            h = layer_norm(h)

            # Activation (除了最后一层)
            if i < self.num_layers - 1:
                h = F.elu(h)
                h = self.dropout(h)

            # Residual connection
            if self.use_residual and h.size(-1) == h_prev.size(-1):
                h = h + h_prev

        # Output projection
        output_features = self.output_proj(h)

        # Apply mask if provided
        if node_mask is not None:
            output_features = output_features * node_mask.unsqueeze(-1)

        return {
            'node_features': output_features,
            'attention_weights': attention_weights,
        }


class EntityGraphNetwork(nn.Module):
    """
    完整的实体图网络
    结合 EntityRelationEncoder 和 GraphAttentionNetwork
    """

    def __init__(
        self,
        entity_dim: int = 768,
        relation_dim: int = 128,
        hidden_dim: int = 256,
        num_gat_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.entity_dim = entity_dim
        self.relation_dim = relation_dim

        # Relation encoder (imported from entity_relation_encoder.py)
        from gmner.models.entity_relation_encoder import EntityRelationEncoder
        self.relation_encoder = EntityRelationEncoder(
            entity_dim=entity_dim,
            relation_dim=relation_dim,
            dropout=dropout,
        )

        # Graph attention network
        self.gat = GraphAttentionNetwork(
            node_dim=entity_dim,
            hidden_dim=hidden_dim,
            num_layers=num_gat_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Edge feature incorporation
        self.edge_proj = nn.Linear(relation_dim, hidden_dim)

    def forward(
        self,
        entity_reprs: torch.Tensor,
        entity_positions: torch.Tensor,
        entity_types: torch.Tensor,
        entity_mask: torch.Tensor,
        text_reprs: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        完整的实体图处理流程

        Args:
            entity_reprs: [batch, max_entities, entity_dim]
            entity_positions: [batch, max_entities, 2]
            entity_types: [batch, max_entities, num_types]
            entity_mask: [batch, max_entities]
            text_reprs: [batch, seq_len, entity_dim] (可选)

        Returns:
            outputs: {
                'enhanced_entity_reprs': [batch, max_entities, entity_dim],
                'relation_matrix': [batch, max_entities, max_entities, relation_dim],
                'adjacency': [batch, max_entities, max_entities],
            }
        """
        # 1. Encode relations
        relation_outputs = self.relation_encoder(
            entity_reprs=entity_reprs,
            entity_positions=entity_positions,
            entity_types=entity_types,
            entity_mask=entity_mask,
            text_reprs=text_reprs,
        )

        # 2. Apply graph attention
        gat_outputs = self.gat(
            node_features=entity_reprs,
            adjacency=relation_outputs['adjacency'],
            node_mask=entity_mask,
        )

        return {
            'enhanced_entity_reprs': gat_outputs['node_features'],
            'relation_matrix': relation_outputs['relation_matrix'],
            'adjacency': relation_outputs['adjacency'],
            'spatial_relations': relation_outputs['spatial_relations'],
            'semantic_relations': relation_outputs['semantic_relations'],
            'context_relations': relation_outputs['context_relations'],
        }


def create_entity_graph_network(config: Dict) -> EntityGraphNetwork:
    """创建 EntityGraphNetwork 实例"""
    return EntityGraphNetwork(
        entity_dim=config.get('entity_dim', 768),
        relation_dim=config.get('relation_dim', 128),
        hidden_dim=config.get('hidden_dim', 256),
        num_gat_layers=config.get('num_gat_layers', 2),
        num_heads=config.get('num_heads', 4),
        dropout=config.get('dropout', 0.1),
    )
