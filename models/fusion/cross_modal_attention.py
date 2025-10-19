"""
跨模态交互增强模块 - Level 2
让不同模态的特征互相"看到"对方的信息
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CrossModalAttention(nn.Module):
    """
    跨模态注意力机制
    使用简化版Multi-Head Cross Attention
    """
    
    def __init__(self, modal_dim=256, num_heads=4, dropout=0.1):
        """
        Args:
            modal_dim: 模态特征维度
            num_heads: 注意力头数
            dropout: Dropout率
        """
        super(CrossModalAttention, self).__init__()
        
        assert modal_dim % num_heads == 0, "modal_dim must be divisible by num_heads"
        
        self.modal_dim = modal_dim
        self.num_heads = num_heads
        self.head_dim = modal_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # Q, K, V投影层
        self.query_proj = nn.Linear(modal_dim, modal_dim)
        self.key_proj = nn.Linear(modal_dim, modal_dim)
        self.value_proj = nn.Linear(modal_dim, modal_dim)
        
        # 输出投影
        self.out_proj = nn.Linear(modal_dim, modal_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query_feat, key_value_feats):
        """
        前向传播
        
        Args:
            query_feat: Query特征 (B, modal_dim) - 通常是融合特征
            key_value_feats: Key和Value特征 (B, 3, modal_dim) - 三个模态特征堆叠
        
        Returns:
            attn_out: 注意力输出 (B, modal_dim)
            attn_weights: 注意力权重 (B, num_heads, 1, 3)
        """
        batch_size = query_feat.size(0)
        
        # 投影Q, K, V
        Q = self.query_proj(query_feat)  # (B, modal_dim)
        Q = Q.unsqueeze(1)  # (B, 1, modal_dim) - 1个query
        
        K = self.key_proj(key_value_feats)  # (B, 3, modal_dim)
        V = self.value_proj(key_value_feats)  # (B, 3, modal_dim)
        
        # 重塑为多头形式
        Q = Q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, 1, head_dim)
        K = K.view(batch_size, 3, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, 3, head_dim)
        V = V.view(batch_size, 3, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, 3, head_dim)
        
        # 计算注意力分数
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, num_heads, 1, 3)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, num_heads, 1, 3)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力
        attn_out = torch.matmul(attn_weights, V)  # (B, num_heads, 1, head_dim)
        
        # 合并多头
        attn_out = attn_out.transpose(1, 2).contiguous()  # (B, 1, num_heads, head_dim)
        attn_out = attn_out.view(batch_size, 1, self.modal_dim)  # (B, 1, modal_dim)
        attn_out = attn_out.squeeze(1)  # (B, modal_dim)
        
        # 输出投影
        attn_out = self.out_proj(attn_out)  # (B, modal_dim)
        
        return attn_out, attn_weights


class CrossModalInteraction(nn.Module):
    """
    跨模态交互模块
    包含注意力机制和残差连接
    """
    
    def __init__(self, modal_dim=256, num_heads=4, dropout=0.1, residual_weight=0.5):
        """
        Args:
            modal_dim: 模态特征维度
            num_heads: 注意力头数
            dropout: Dropout率
            residual_weight: 残差连接权重
        """
        super(CrossModalInteraction, self).__init__()
        
        self.cross_attn = CrossModalAttention(
            modal_dim=modal_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.residual_weight = residual_weight
        
        # Layer Norm
        self.norm = nn.LayerNorm(modal_dim)
        
    def forward(self, weighted_feat, feat_temporal, feat_frequency, feat_timefreq):
        """
        前向传播
        
        Args:
            weighted_feat: 加权融合特征 (B, modal_dim)
            feat_temporal: 时域特征 (B, modal_dim)
            feat_frequency: 频域特征 (B, modal_dim)
            feat_timefreq: 时频特征 (B, modal_dim)
        
        Returns:
            interact_feat: 交互增强特征 (B, modal_dim)
            attn_weights: 注意力权重 (B, num_heads, 1, 3)
        """
        # 堆叠三个模态特征作为Key和Value
        key_value_feats = torch.stack([feat_temporal, feat_frequency, feat_timefreq], dim=1)  # (B, 3, modal_dim)
        
        # 计算交互注意力
        attn_out, attn_weights = self.cross_attn(weighted_feat, key_value_feats)
        
        # 残差连接
        interact_feat = weighted_feat + self.residual_weight * attn_out
        
        # Layer Norm
        interact_feat = self.norm(interact_feat)
        
        return interact_feat, attn_weights


# 单元测试
if __name__ == '__main__':
    print("Testing Cross-Modal Attention...")
    
    # 创建模块
    cross_attn = CrossModalAttention(modal_dim=256, num_heads=4)
    
    # 创建测试数据
    batch_size = 8
    query_feat = torch.randn(batch_size, 256)  # 融合特征
    key_value_feats = torch.randn(batch_size, 3, 256)  # 三个模态
    
    # 前向传播
    attn_out, attn_weights = cross_attn(query_feat, key_value_feats)
    
    print(f"✓ Attention output shape: {attn_out.shape}")  # (8, 256)
    print(f"✓ Attention weights shape: {attn_weights.shape}")  # (8, 4, 1, 3)
    print(f"✓ Sample attention weights (head 0): {attn_weights[0, 0, 0, :]}")
    
    # 测试完整交互模块
    print("\nTesting Cross-Modal Interaction...")
    interaction = CrossModalInteraction(modal_dim=256)
    
    weighted_feat = torch.randn(batch_size, 256)
    feat_t = torch.randn(batch_size, 256)
    feat_f = torch.randn(batch_size, 256)
    feat_tf = torch.randn(batch_size, 256)
    
    interact_feat, attn_weights = interaction(weighted_feat, feat_t, feat_f, feat_tf)
    
    print(f"✓ Interaction output shape: {interact_feat.shape}")  # (8, 256)
    print(f"✓ Mean attention weights across heads:")
    print(f"  Time: {attn_weights[:, :, 0, 0].mean():.3f}")
    print(f"  Freq: {attn_weights[:, :, 0, 1].mean():.3f}")
    print(f"  Time-Freq: {attn_weights[:, :, 0, 2].mean():.3f}")
    
    print("\n✅ All tests passed!")
