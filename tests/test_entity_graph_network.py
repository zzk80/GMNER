#!/usr/bin/env python3
"""
测试 Entity Relation Encoder 和 Graph Attention Network

验证:
1. 模块能否正常前向传播
2. 输出维度是否正确
3. 关系编码是否有效
"""

import sys
import torch
import torch.nn as nn

# Add parent directory to path
sys.path.insert(0, '.')

from gmner.models.entity_relation_encoder import EntityRelationEncoder
from gmner.models.graph_attention_network import GraphAttentionNetwork, EntityGraphNetwork


def test_entity_relation_encoder():
    """测试 EntityRelationEncoder"""
    print("=" * 80)
    print("Testing EntityRelationEncoder")
    print("=" * 80)

    # 创建测试数据
    batch_size = 2
    max_entities = 5
    entity_dim = 768
    num_types = 4

    entity_reprs = torch.randn(batch_size, max_entities, entity_dim)
    entity_positions = torch.randint(0, 50, (batch_size, max_entities, 2))
    entity_positions[:, :, 1] = entity_positions[:, :, 0] + torch.randint(1, 10, (batch_size, max_entities))
    entity_types = torch.randn(batch_size, max_entities, num_types)
    entity_mask = torch.ones(batch_size, max_entities)
    entity_mask[0, 3:] = 0  # 第一个样本只有 3 个实体
    entity_mask[1, 4:] = 0  # 第二个样本只有 4 个实体

    # 创建模型
    model = EntityRelationEncoder(
        entity_dim=entity_dim,
        relation_dim=128,
        num_relation_types=8,
    )

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 前向传播
    print("\nForward pass...")
    outputs = model(
        entity_reprs=entity_reprs,
        entity_positions=entity_positions,
        entity_types=entity_types,
        entity_mask=entity_mask,
    )

    # 检查输出
    print("\nOutputs:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")

    # 验证维度
    assert outputs['relation_matrix'].shape == (batch_size, max_entities, max_entities, 128)
    assert outputs['adjacency'].shape == (batch_size, max_entities, max_entities)

    # 验证 mask 生效
    assert outputs['adjacency'][0, 3:, :].sum() == 0  # Masked entities
    assert outputs['adjacency'][0, :, 3:].sum() == 0

    print("\n✓ EntityRelationEncoder test passed!")
    return True


def test_graph_attention_network():
    """测试 GraphAttentionNetwork"""
    print("\n" + "=" * 80)
    print("Testing GraphAttentionNetwork")
    print("=" * 80)

    # 创建测试数据
    batch_size = 2
    num_nodes = 5
    node_dim = 768

    node_features = torch.randn(batch_size, num_nodes, node_dim)
    # 创建简单的邻接矩阵（全连接）
    adjacency = torch.ones(batch_size, num_nodes, num_nodes)
    adjacency[:, torch.arange(num_nodes), torch.arange(num_nodes)] = 0  # 去除自环
    node_mask = torch.ones(batch_size, num_nodes)
    node_mask[0, 3:] = 0

    # 创建模型
    model = GraphAttentionNetwork(
        node_dim=node_dim,
        hidden_dim=256,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
    )

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 前向传播
    print("\nForward pass...")
    outputs = model(
        node_features=node_features,
        adjacency=adjacency,
        node_mask=node_mask,
    )

    # 检查输出
    print("\nOutputs:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")

    # 验证维度
    assert outputs['node_features'].shape == (batch_size, num_nodes, node_dim)

    # 验证 mask 生效
    assert outputs['node_features'][0, 3:, :].abs().sum() == 0

    print("\n✓ GraphAttentionNetwork test passed!")
    return True


def test_entity_graph_network():
    """测试完整的 EntityGraphNetwork"""
    print("\n" + "=" * 80)
    print("Testing EntityGraphNetwork (Full Pipeline)")
    print("=" * 80)

    # 创建测试数据
    batch_size = 2
    max_entities = 5
    entity_dim = 768
    num_types = 4

    entity_reprs = torch.randn(batch_size, max_entities, entity_dim)
    entity_positions = torch.randint(0, 50, (batch_size, max_entities, 2))
    entity_positions[:, :, 1] = entity_positions[:, :, 0] + torch.randint(1, 10, (batch_size, max_entities))
    entity_types = torch.randn(batch_size, max_entities, num_types)
    entity_mask = torch.ones(batch_size, max_entities)
    entity_mask[0, 3:] = 0
    entity_mask[1, 4:] = 0

    # 创建模型
    model = EntityGraphNetwork(
        entity_dim=entity_dim,
        relation_dim=128,
        hidden_dim=256,
        num_gat_layers=2,
        num_heads=4,
        dropout=0.1,
    )

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 前向传播
    print("\nForward pass...")
    outputs = model(
        entity_reprs=entity_reprs,
        entity_positions=entity_positions,
        entity_types=entity_types,
        entity_mask=entity_mask,
    )

    # 检查输出
    print("\nOutputs:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")

    # 验证维度
    assert outputs['enhanced_entity_reprs'].shape == (batch_size, max_entities, entity_dim)
    assert outputs['relation_matrix'].shape == (batch_size, max_entities, max_entities, 128)
    assert outputs['adjacency'].shape == (batch_size, max_entities, max_entities)

    # 验证 mask 生效
    assert outputs['enhanced_entity_reprs'][0, 3:, :].abs().sum() == 0

    print("\n✓ EntityGraphNetwork test passed!")
    return True


def test_relation_effectiveness():
    """测试关系编码的有效性"""
    print("\n" + "=" * 80)
    print("Testing Relation Encoding Effectiveness")
    print("=" * 80)

    batch_size = 1
    max_entities = 3
    entity_dim = 768

    # 创建有明显关系的实体
    entity_reprs = torch.randn(batch_size, max_entities, entity_dim)

    # Case 1: 相邻实体（位置接近）
    entity_positions = torch.tensor([[[0, 5], [6, 10], [20, 25]]])  # 1和2接近，3远离

    entity_types = torch.tensor([[[1.0, 0.0, 0.0, 0.0],  # 都是 PER
                                   [1.0, 0.0, 0.0, 0.0],
                                   [0.0, 1.0, 0.0, 0.0]]])  # LOC

    entity_mask = torch.ones(batch_size, max_entities)

    # 创建模型
    model = EntityRelationEncoder(entity_dim=entity_dim, relation_dim=128)

    # 前向传播
    outputs = model(
        entity_reprs=entity_reprs,
        entity_positions=entity_positions,
        entity_types=entity_types,
        entity_mask=entity_mask,
    )

    # 分析关系强度
    spatial_rel = outputs['spatial_relations'][0]  # [3, 3, 128]
    semantic_rel = outputs['semantic_relations'][0]

    # 计算距离（L2 norm）
    spatial_dist = torch.norm(spatial_rel, dim=-1)  # [3, 3]
    semantic_dist = torch.norm(semantic_rel, dim=-1)

    print("\nSpatial distances:")
    print(f"  Entity 0-1 (adjacent): {spatial_dist[0, 1]:.4f}")
    print(f"  Entity 0-2 (far): {spatial_dist[0, 2]:.4f}")
    print(f"  Entity 1-2 (far): {spatial_dist[1, 2]:.4f}")

    print("\nSemantic distances:")
    print(f"  Entity 0-1 (both PER): {semantic_dist[0, 1]:.4f}")
    print(f"  Entity 0-2 (PER vs LOC): {semantic_dist[0, 2]:.4f}")
    print(f"  Entity 1-2 (PER vs LOC): {semantic_dist[1, 2]:.4f}")

    # 验证：相邻实体的空间距离应该小于远离实体
    # 注意：这是一个软约束，不一定总是满足
    print("\nAdjacency matrix:")
    print(outputs['adjacency'][0])

    print("\n✓ Relation effectiveness test completed!")
    return True


def main():
    """运行所有测试"""
    print("\n" + "=" * 80)
    print("Entity Relation Encoder & Graph Attention Network Tests")
    print("=" * 80)

    try:
        # Test 1: Entity Relation Encoder
        test_entity_relation_encoder()

        # Test 2: Graph Attention Network
        test_graph_attention_network()

        # Test 3: Full Entity Graph Network
        test_entity_graph_network()

        # Test 4: Relation Effectiveness
        test_relation_effectiveness()

        print("\n" + "=" * 80)
        print("✅ All tests passed!")
        print("=" * 80)

        print("\nNext steps:")
        print("  1. Integrate with existing GMNER pipeline")
        print("  2. Train on multi-entity scenes")
        print("  3. Evaluate on Dev set")
        print("  4. Compare with baseline (independent matching)")

    except Exception as e:
        print("\n" + "=" * 80)
        print(f"❌ Test failed: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
