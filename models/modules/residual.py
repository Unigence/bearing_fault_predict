"""
残差块模块
包含1D和2D残差块
"""
import torch
import torch.nn as nn


class ResidualBlock1D(nn.Module):
    """
    1D残差块 (用于时域和频域分支)
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.2):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核大小
            dropout: Dropout率
        """
        super(ResidualBlock1D, self).__init__()
        
        padding = (kernel_size - 1) // 2
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                     padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.dropout = nn.Dropout1d(dropout)
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                     padding=padding, bias=False),
            nn.BatchNorm1d(out_channels)
        )
        
        # 如果通道数不匹配，使用1x1卷积调整
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        """
        Args:
            x: (B, C_in, L)
        Returns:
            out: (B, C_out, L)
        """
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.dropout(out)
        out = self.conv2(out)
        
        out = out + identity
        out = self.relu(out)
        
        return out


class ResidualBlock2D(nn.Module):
    """
    2D残差块 (用于时频分支)
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.2):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核大小
            dropout: Dropout率
        """
        super(ResidualBlock2D, self).__init__()
        
        padding = (kernel_size - 1) // 2
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                     padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.dropout = nn.Dropout2d(dropout)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size,
                     padding=padding, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # 如果通道数不匹配，使用1x1卷积调整
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        """
        Args:
            x: (B, C_in, H, W)
        Returns:
            out: (B, C_out, H, W)
        """
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.dropout(out)
        out = self.conv2(out)
        
        out = out + identity
        out = self.relu(out)
        
        return out


class BottleneckResidualBlock1D(nn.Module):
    """
    瓶颈残差块 (1D版本)
    使用1x1卷积降维，减少计算量
    """
    
    def __init__(self, in_channels, bottleneck_channels, out_channels, dropout=0.2):
        """
        Args:
            in_channels: 输入通道数
            bottleneck_channels: 瓶颈层通道数
            out_channels: 输出通道数
            dropout: Dropout率
        """
        super(BottleneckResidualBlock1D, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(bottleneck_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(bottleneck_channels, bottleneck_channels, kernel_size=3,
                     padding=1, bias=False),
            nn.BatchNorm1d(bottleneck_channels),
            nn.ReLU(inplace=True)
        )
        
        self.dropout = nn.Dropout1d(dropout)
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(bottleneck_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels)
        )
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        """
        Args:
            x: (B, C_in, L)
        Returns:
            out: (B, C_out, L)
        """
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.dropout(out)
        out = self.conv3(out)
        
        out = out + identity
        out = self.relu(out)
        
        return out
