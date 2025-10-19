"""
注意力机制模块
包含: CBAM, SE, EMA, Temporal Self-Attention等
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """通道注意力模块 (CBAM的通道注意力部分)"""
    
    def __init__(self, in_channels, reduction=8):
        """
        Args:
            in_channels: 输入通道数
            reduction: 降维比例
        """
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        # 共享MLP
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Args:
            x: (B, C, L)
        Returns:
            out: (B, C, L)
        """
        b, c, _ = x.size()
        
        # 平均池化和最大池化
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        
        # 融合
        out = self.sigmoid(avg_out + max_out).view(b, c, 1)
        return x * out.expand_as(x)


class SpatialAttention(nn.Module):
    """空间注意力模块 (CBAM的空间注意力部分)"""
    
    def __init__(self, kernel_size=7):
        """
        Args:
            kernel_size: 卷积核大小
        """
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size, 
                             padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Args:
            x: (B, C, L)
        Returns:
            out: (B, C, L)
        """
        # 沿通道维度计算平均和最大
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (B, 1, L)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, L)
        
        # 拼接
        concat = torch.cat([avg_out, max_out], dim=1)  # (B, 2, L)
        
        # 卷积
        out = self.sigmoid(self.conv(concat))  # (B, 1, L)
        return x * out.expand_as(x)


class CBAM(nn.Module):
    """
    卷积块注意力模块 (Convolutional Block Attention Module)
    结合通道注意力和空间注意力
    """
    
    def __init__(self, in_channels, reduction=8, kernel_size=7):
        """
        Args:
            in_channels: 输入通道数
            reduction: 通道注意力降维比例
            kernel_size: 空间注意力卷积核大小
        """
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        """
        Args:
            x: (B, C, L)
        Returns:
            out: (B, C, L)
        """
        out = self.channel_attention(x)
        out = self.spatial_attention(out)
        return out


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block
    通过全局池化和全连接层学习通道权重
    """
    
    def __init__(self, in_channels, reduction=8):
        """
        Args:
            in_channels: 输入通道数
            reduction: 降维比例
        """
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, C, L)
        Returns:
            out: (B, C, L)
        """
        b, c, _ = x.size()
        # Squeeze: 全局平均池化
        y = self.avg_pool(x).view(b, c)
        # Excitation: FC层
        y = self.fc(y).view(b, c, 1)
        # Scale
        return x * y.expand_as(x)


class TemporalSelfAttention(nn.Module):
    """
    时序自注意力模块
    用于捕捉长距离时间依赖
    """
    
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        """
        Args:
            embed_dim: 嵌入维度
            num_heads: 注意力头数
            dropout: Dropout率
        """
        super(TemporalSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim必须能被num_heads整除"
        
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x):
        """
        Args:
            x: (B, L, C) - Batch, Length, Channels
        Returns:
            out: (B, L, C)
        """
        B, L, C = x.size()
        
        # 线性变换
        Q = self.q_linear(x)  # (B, L, C)
        K = self.k_linear(x)
        V = self.v_linear(x)
        
        # 重塑为多头
        Q = Q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, L, head_dim)
        K = K.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, num_heads, L, L)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力
        attn_output = torch.matmul(attn_weights, V)  # (B, num_heads, L, head_dim)
        
        # 重塑回原始维度
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, C)
        
        # 输出线性变换
        output = self.out_linear(attn_output)
        
        return output


class EMA(nn.Module):
    """
    高效多尺度注意力 (Efficient Multi-scale Attention)
    用于2D特征图(时频分支)
    """
    
    def __init__(self, in_channels, reduction=8):
        """
        Args:
            in_channels: 输入通道数
            reduction: 降维比例
        """
        super(EMA, self).__init__()
        
        # 多尺度特征提取
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv3x3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, 
                                 padding=2, groups=in_channels)  # depthwise
        
        # 空间注意力
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
        
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            out: (B, C, H, W)
        """
        # 多尺度特征
        feat1 = self.conv1x1(x)
        feat3 = self.conv3x3(x)
        feat5 = self.conv5x5(x)
        
        # 融合多尺度特征
        multi_scale = feat1 + feat3 + feat5
        
        # 空间注意力
        avg_out = torch.mean(multi_scale, dim=1, keepdim=True)
        max_out, _ = torch.max(multi_scale, dim=1, keepdim=True)
        spatial_concat = torch.cat([avg_out, max_out], dim=1)
        spatial_attn = self.spatial_conv(spatial_concat)
        
        # 通道注意力
        channel_attn = self.channel_attention(multi_scale)
        
        # 应用注意力
        out = x * spatial_attn * channel_attn
        
        return out


class ChannelAttention2D(nn.Module):
    """2D通道注意力 (用于时频分支)"""
    
    def __init__(self, in_channels, reduction=8):
        super(ChannelAttention2D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return x * out


class SpatialAttention2D(nn.Module):
    """2D空间注意力 (用于时频分支)"""
    
    def __init__(self, kernel_size=7):
        super(SpatialAttention2D, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, 
                             padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(concat))
        return x * out
