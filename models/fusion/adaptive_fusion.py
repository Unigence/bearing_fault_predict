"""
自适应特征融合模块 - Level 1: 模态重要性学习
根据输入自适应学习每个模态的贡献权重
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalImportanceLearning(nn.Module):
    """
    模态重要性学习网络
    动态学习时域、频域、时频域三个模态的重要性权重
    """
    
    def __init__(self, modal_dim=256, hidden_dim=256, dropout=0.25):
        """
        Args:
            modal_dim: 每个模态的特征维度
            hidden_dim: 隐藏层维度
            dropout: Dropout率
        """
        super(ModalImportanceLearning, self).__init__()
        
        self.modal_dim = modal_dim
        self.num_modals = 3  # 时域、频域、时频域
        
        # 权重生成网络
        self.weight_net = nn.Sequential(
            nn.Linear(modal_dim * 3, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            
            nn.Linear(64, 3),  # 输出3个权重
            nn.Softmax(dim=1)  # 归一化到[0,1]且和为1
        )
    
    def forward(self, feat_temporal, feat_frequency, feat_timefreq):
        """
        前向传播
        
        Args:
            feat_temporal: 时域特征 (B, modal_dim)
            feat_frequency: 频域特征 (B, modal_dim)
            feat_timefreq: 时频特征 (B, modal_dim)
        
        Returns:
            weighted_feat: 加权融合特征 (B, modal_dim)
            weights: 模态权重 (B, 3) - [w_t, w_f, w_tf]
        """
        batch_size = feat_temporal.size(0)
        
        # 拼接所有模态特征
        concat_feat = torch.cat([feat_temporal, feat_frequency, feat_timefreq], dim=1)  # (B, 768)
        
        # 生成权重
        weights = self.weight_net(concat_feat)  # (B, 3)
        
        # 加权融合
        w_t = weights[:, 0].unsqueeze(1)  # (B, 1)
        w_f = weights[:, 1].unsqueeze(1)  # (B, 1)
        w_tf = weights[:, 2].unsqueeze(1)  # (B, 1)
        
        weighted_feat = (w_t * feat_temporal + 
                        w_f * feat_frequency + 
                        w_tf * feat_timefreq)  # (B, modal_dim)
        
        return weighted_feat, weights


class AdaptiveFusionV1(nn.Module):
    """
    自适应融合模块 V1版本
    只包含Level 1的模态重要性学习
    """
    
    def __init__(self, modal_dim=256, hidden_dim=256, dropout=0.25):
        """
        Args:
            modal_dim: 每个模态的特征维度
            hidden_dim: 隐藏层维度
            dropout: Dropout率
        """
        super(AdaptiveFusionV1, self).__init__()
        
        self.modal_importance = ModalImportanceLearning(
            modal_dim=modal_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
    
    def forward(self, feat_temporal, feat_frequency, feat_timefreq):
        """
        前向传播
        
        Args:
            feat_temporal: 时域特征 (B, modal_dim)
            feat_frequency: 频域特征 (B, modal_dim)
            feat_timefreq: 时频特征 (B, modal_dim)
        
        Returns:
            fused_feat: 融合特征 (B, modal_dim)
            weights: 模态权重 (B, 3)
        """
        fused_feat, weights = self.modal_importance(
            feat_temporal, feat_frequency, feat_timefreq
        )
        
        return fused_feat, weights


# 单元测试
if __name__ == '__main__':
    print("Testing Modal Importance Learning...")
    
    # 创建模块
    fusion = ModalImportanceLearning(modal_dim=256)
    
    # 创建测试数据
    batch_size = 8
    feat_t = torch.randn(batch_size, 256)
    feat_f = torch.randn(batch_size, 256)
    feat_tf = torch.randn(batch_size, 256)
    
    # 前向传播
    fused_feat, weights = fusion(feat_t, feat_f, feat_tf)
    
    print(f"✓ Fused feature shape: {fused_feat.shape}")  # 应该是 (8, 256)
    print(f"✓ Weights shape: {weights.shape}")  # 应该是 (8, 3)
    print(f"✓ Weights sum: {weights.sum(dim=1)}")  # 每个样本的权重和应该是1
    print(f"✓ Sample weights: {weights[0]}")  # 示例权重
    
    # 测试完整融合模块
    print("\nTesting AdaptiveFusionV1...")
    fusion_v1 = AdaptiveFusionV1(modal_dim=256)
    fused_feat, weights = fusion_v1(feat_t, feat_f, feat_tf)
    
    print(f"✓ AdaptiveFusionV1 output shape: {fused_feat.shape}")
    print(f"✓ AdaptiveFusionV1 weights: {weights[0]}")
    
    print("\n✅ All tests passed!")
