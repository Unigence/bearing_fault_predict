"""
多级自适应特征融合模块
整合三个Level: 模态重要性学习 + 跨模态交互 + 多粒度整合
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .adaptive_fusion import ModalImportanceLearning
from .cross_modal_attention import CrossModalInteraction


class HierarchicalFusion(nn.Module):
    """
    多级自适应特征融合模块
    三级渐进式融合:
    - Level 1: 模态重要性学习
    - Level 2: 跨模态交互增强(单层)
    - Level 3: 多粒度特征整合
    """

    def __init__(self,
                 modal_dim=256,
                 hidden_dim=256,
                 output_dim=128,
                 num_heads=4,
                 dropout_l1=0.25,
                 dropout_l2=0.35,
                 dropout_l3=0.3,
                 residual_weight=0.5):
        """
        Args:
            modal_dim: 每个模态的特征维度
            hidden_dim: 隐藏层维度
            output_dim: 最终输出维度
            num_heads: 注意力头数
            dropout_l1: Level 1的Dropout率
            dropout_l2: Level 2的Dropout率(用于integration)
            dropout_l3: Level 3的Dropout率
            residual_weight: 残差连接权重
        """
        super(HierarchicalFusion, self).__init__()

        self.modal_dim = modal_dim
        self.output_dim = output_dim

        # ==================== Level 1: 模态重要性学习 ====================
        self.level1_modal_importance = ModalImportanceLearning(
            modal_dim=modal_dim,
            hidden_dim=hidden_dim,
            dropout=dropout_l1
        )

        # ==================== Level 2: 跨模态交互增强 ====================
        # 修复: 只使用单层CrossModalInteraction
        self.level2_cross_modal = CrossModalInteraction(
            modal_dim=modal_dim,
            num_heads=num_heads,
            dropout=0.1,  # attention内部dropout
            residual_weight=residual_weight
        )

        # ==================== Level 3: 多粒度特征整合 ====================
        # 拼接: [F_t, F_f, F_tf, F_weighted, F_interact] = 256*5 = 1280
        fusion_input_dim = modal_dim * 5

        self.level3_integration = nn.Sequential(
            nn.Linear(fusion_input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_l2),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_l2),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_l3),

            nn.Linear(256, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.Dropout(0.2)
        )

    def forward(self, feat_temporal, feat_frequency, feat_timefreq, return_intermediate=False):
        """
        前向传播

        Args:
            feat_temporal: 时域特征 (B, modal_dim)
            feat_frequency: 频域特征 (B, modal_dim)
            feat_timefreq: 时频特征 (B, modal_dim)
            return_intermediate: 是否返回中间特征

        Returns:
            fused_feat: 融合特征 (B, output_dim)
            intermediate_dict: 中间特征字典(可选)
        """
        # Level 1: 模态重要性学习
        weighted_feat, modal_weights = self.level1_modal_importance(
            feat_temporal, feat_frequency, feat_timefreq
        )

        # Level 2: 跨模态交互增强
        # 修复: 直接调用单层的cross_modal,不再使用循环
        interact_feat, attn_weights = self.level2_cross_modal(
            weighted_feat, feat_temporal, feat_frequency, feat_timefreq
        )

        # Level 3: 多粒度特征整合
        # 拼接: 原始特征 + 加权特征 + 交互特征
        multi_granularity_feat = torch.cat([
            feat_temporal,      # 原始时域 (B, 256)
            feat_frequency,     # 原始频域 (B, 256)
            feat_timefreq,      # 原始时频 (B, 256)
            weighted_feat,      # 加权融合 (B, 256)
            interact_feat       # 交互增强 (B, 256)
        ], dim=1)  # (B, 1280)

        # 特征压缩与精炼
        fused_feat = self.level3_integration(multi_granularity_feat)  # (B, output_dim)

        # 返回中间特征（用于分析和可视化）
        if return_intermediate:
            intermediate_dict = {
                'weighted_feat': weighted_feat,
                'interact_feat': interact_feat,
                'modal_weights': modal_weights,
                'attn_weights': attn_weights,
                'multi_granularity_feat': multi_granularity_feat
            }
            return fused_feat, intermediate_dict

        return fused_feat


class HierarchicalFusionV2(nn.Module):
    """
    多级融合模块 V2版本
    使用多层交互网络,适合需要更深层次交互的场景
    """

    def __init__(self,
                 modal_dim=256,
                 hidden_dim=256,
                 output_dim=128,
                 num_heads=8,
                 num_interaction_layers=2,
                 dropout=0.3):
        """
        Args:
            modal_dim: 每个模态的特征维度
            hidden_dim: 隐藏层维度
            output_dim: 最终输出维度
            num_heads: 注意力头数
            num_interaction_layers: 交互层数
            dropout: Dropout率
        """
        super(HierarchicalFusionV2, self).__init__()

        self.modal_dim = modal_dim
        self.output_dim = output_dim
        self.num_interaction_layers = num_interaction_layers

        # Level 1: 模态重要性学习
        self.modal_importance = ModalImportanceLearning(
            modal_dim=modal_dim,
            hidden_dim=hidden_dim,
            dropout=0.25
        )

        # Level 2: 多层交互
        self.cross_modal_layers = nn.ModuleList([
            CrossModalInteraction(
                modal_dim=modal_dim,
                num_heads=num_heads,
                dropout=0.1,
                residual_weight=0.5
            )
            for _ in range(num_interaction_layers)
        ])

        # Level 3: 深度特征整合
        # 拼接: 原始特征(3) + 加权特征(1) + 各层交互特征(num_interaction_layers)
        fusion_input_dim = modal_dim * (3 + 1 + num_interaction_layers)

        self.integration = nn.Sequential(
            nn.Linear(fusion_input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.7),

            nn.Linear(256, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.Dropout(0.2)
        )

    def forward(self, feat_temporal, feat_frequency, feat_timefreq, return_intermediate=False):
        """前向传播 - V2版本"""
        # Level 1
        weighted_feat, modal_weights = self.modal_importance(
            feat_temporal, feat_frequency, feat_timefreq
        )

        # Level 2: 多层交互
        interaction_feats = [weighted_feat]
        current_feat = weighted_feat
        attn_weights_list = []

        for cross_modal_layer in self.cross_modal_layers:
            current_feat, attn_weights = cross_modal_layer(
                current_feat, feat_temporal, feat_frequency, feat_timefreq
            )
            interaction_feats.append(current_feat)
            attn_weights_list.append(attn_weights)

        # Level 3: 拼接所有特征
        all_feats = [feat_temporal, feat_frequency, feat_timefreq] + interaction_feats
        multi_granularity_feat = torch.cat(all_feats, dim=1)

        # 最终融合
        fused_feat = self.integration(multi_granularity_feat)

        if return_intermediate:
            intermediate_dict = {
                'weighted_feat': weighted_feat,
                'interaction_feats': interaction_feats,
                'modal_weights': modal_weights,
                'attn_weights_list': attn_weights_list,
                'multi_granularity_feat': multi_granularity_feat
            }
            return fused_feat, intermediate_dict

        return fused_feat


# 单元测试
if __name__ == '__main__':
    print("="  * 70)
    print("测试修复后的HierarchicalFusion")
    print("="  * 70)

    # 创建模块
    fusion = HierarchicalFusion(
        modal_dim=256,
        output_dim=128,
        num_heads=4
    )

    # 创建测试数据
    batch_size = 8
    feat_t = torch.randn(batch_size, 256)
    feat_f = torch.randn(batch_size, 256)
    feat_tf = torch.randn(batch_size, 256)

    # 前向传播
    print("\n1. 测试基本前向传播...")
    fused_feat = fusion(feat_t, feat_f, feat_tf)
    print(f"✓ 融合特征形状: {fused_feat.shape}")  # (8, 128)

    # 测试返回中间特征
    print("\n2. 测试返回中间特征...")
    fused_feat, intermediate = fusion(feat_t, feat_f, feat_tf, return_intermediate=True)

    print(f"✓ 融合特征形状: {fused_feat.shape}")
    print(f"✓ 中间特征:")
    print(f"  - weighted_feat: {intermediate['weighted_feat'].shape}")
    print(f"  - interact_feat: {intermediate['interact_feat'].shape}")
    print(f"  - modal_weights: {intermediate['modal_weights'].shape}")
    print(f"  - attn_weights: {intermediate['attn_weights'].shape}")
    print(f"  - multi_granularity_feat: {intermediate['multi_granularity_feat'].shape}")

    print(f"\n✓ 示例模态权重: {intermediate['modal_weights'][0]}")

    # 测试V2版本
    print("\n3. 测试HierarchicalFusionV2...")
    fusion_v2 = HierarchicalFusionV2(
        modal_dim=256,
        output_dim=128,
        num_heads=8,
        num_interaction_layers=2
    )

    fused_feat_v2 = fusion_v2(feat_t, feat_f, feat_tf)
    print(f"✓ V2融合特征形状: {fused_feat_v2.shape}")

    # 参数统计
    print("\n4. 参数统计...")
    params = sum(p.numel() for p in fusion.parameters())
    params_v2 = sum(p.numel() for p in fusion_v2.parameters())

    print(f"✓ HierarchicalFusion参数量: {params:,}")
    print(f"✓ HierarchicalFusionV2参数量: {params_v2:,}")

    print("\n" + "=" * 70)
    print("✅ 所有测试通过!")
    print("=" * 70)
