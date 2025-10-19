"""
池化层模块
包含各种全局池化和自适应池化
"""
import torch
import torch.nn as nn


class GlobalPooling1D(nn.Module):
    """
    全局池化 (1D)
    同时进行平均池化和最大池化，然后拼接
    """
    
    def __init__(self, pooling_type='both'):
        """
        Args:
            pooling_type: 'avg', 'max', 或 'both'
        """
        super(GlobalPooling1D, self).__init__()
        self.pooling_type = pooling_type
        
        if pooling_type in ['avg', 'both']:
            self.avg_pool = nn.AdaptiveAvgPool1d(1)
        if pooling_type in ['max', 'both']:
            self.max_pool = nn.AdaptiveMaxPool1d(1)
    
    def forward(self, x):
        """
        Args:
            x: (B, C, L)
        Returns:
            out: (B, C) 或 (B, 2*C) if pooling_type='both'
        """
        if self.pooling_type == 'avg':
            out = self.avg_pool(x).squeeze(-1)
        elif self.pooling_type == 'max':
            out = self.max_pool(x).squeeze(-1)
        else:  # both
            avg_out = self.avg_pool(x).squeeze(-1)
            max_out = self.max_pool(x).squeeze(-1)
            out = torch.cat([avg_out, max_out], dim=1)
        
        return out


class GlobalPooling2D(nn.Module):
    """
    全局池化 (2D)
    同时进行平均池化和最大池化，然后拼接
    """
    
    def __init__(self, pooling_type='both'):
        """
        Args:
            pooling_type: 'avg', 'max', 或 'both'
        """
        super(GlobalPooling2D, self).__init__()
        self.pooling_type = pooling_type
        
        if pooling_type in ['avg', 'both']:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if pooling_type in ['max', 'both']:
            self.max_pool = nn.AdaptiveMaxPool2d(1)
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            out: (B, C) 或 (B, 2*C) if pooling_type='both'
        """
        if self.pooling_type == 'avg':
            out = self.avg_pool(x).squeeze(-1).squeeze(-1)
        elif self.pooling_type == 'max':
            out = self.max_pool(x).squeeze(-1).squeeze(-1)
        else:  # both
            avg_out = self.avg_pool(x).squeeze(-1).squeeze(-1)
            max_out = self.max_pool(x).squeeze(-1).squeeze(-1)
            out = torch.cat([avg_out, max_out], dim=1)
        
        return out


class AdaptivePooling2D(nn.Module):
    """
    自适应池化 (2D)
    可以指定输出尺寸
    """
    
    def __init__(self, output_size=(4, 4), pooling_type='both'):
        """
        Args:
            output_size: 输出尺寸 (H, W)
            pooling_type: 'avg', 'max', 或 'both'
        """
        super(AdaptivePooling2D, self).__init__()
        self.pooling_type = pooling_type
        
        if pooling_type in ['avg', 'both']:
            self.avg_pool = nn.AdaptiveAvgPool2d(output_size)
        if pooling_type in ['max', 'both']:
            self.max_pool = nn.AdaptiveMaxPool2d(output_size)
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H_in, W_in)
        Returns:
            out: (B, C, H_out, W_out) 或 (B, 2*C, H_out, W_out) if pooling_type='both'
        """
        if self.pooling_type == 'avg':
            out = self.avg_pool(x)
        elif self.pooling_type == 'max':
            out = self.max_pool(x)
        else:  # both
            avg_out = self.avg_pool(x)
            max_out = self.max_pool(x)
            out = torch.cat([avg_out, max_out], dim=1)
        
        return out


class GeM(nn.Module):
    """
    Generalized Mean Pooling
    可学习的p参数的广义均值池化
    """
    
    def __init__(self, p=3.0, eps=1e-6):
        """
        Args:
            p: 初始的p值，p越大越接近max pooling
            eps: 避免除零的小常数
        """
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
    
    def forward(self, x):
        """
        Args:
            x: (B, C, L) 或 (B, C, H, W)
        Returns:
            out: (B, C)
        """
        # GeM pooling: (1/n * Σ x_i^p)^(1/p)
        if x.dim() == 3:  # 1D
            return self.gem_1d(x)
        elif x.dim() == 4:  # 2D
            return self.gem_2d(x)
        else:
            raise ValueError(f"不支持的输入维度: {x.dim()}")
    
    def gem_1d(self, x):
        """1D GeM pooling"""
        return F.avg_pool1d(x.clamp(min=self.eps).pow(self.p), 
                           kernel_size=x.size(-1)).pow(1. / self.p).squeeze(-1)
    
    def gem_2d(self, x):
        """2D GeM pooling"""
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), 
                           kernel_size=(x.size(-2), x.size(-1))).pow(1. / self.p).squeeze(-1).squeeze(-1)


import torch.nn.functional as F


class MultiHeadPooling1D(nn.Module):
    """
    多头池化 (1D)
    结合全局平均、全局最大和最后时间步特征
    """
    
    def __init__(self, in_channels):
        """
        Args:
            in_channels: 输入通道数
        """
        super(MultiHeadPooling1D, self).__init__()
        self.in_channels = in_channels
    
    def forward(self, x):
        """
        Args:
            x: (B, C, L) 或 (B, L, C) 取决于from_rnn参数
        Returns:
            out: (B, 3*C) - [avg_pool, max_pool, last_hidden]拼接
        """
        # 如果是RNN输出格式 (B, L, C)，转换为 (B, C, L)
        if x.size(1) != self.in_channels:
            x = x.transpose(1, 2)  # (B, L, C) -> (B, C, L)
        
        # Head 1: Global Average Pooling
        avg_pool = F.adaptive_avg_pool1d(x, 1).squeeze(-1)  # (B, C)
        
        # Head 2: Global Max Pooling
        max_pool = F.adaptive_max_pool1d(x, 1).squeeze(-1)  # (B, C)
        
        # Head 3: Last Hidden State
        last_hidden = x[:, :, -1]  # (B, C)
        
        # 拼接
        out = torch.cat([avg_pool, max_pool, last_hidden], dim=1)  # (B, 3*C)
        
        return out
