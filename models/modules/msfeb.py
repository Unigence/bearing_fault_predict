"""
多尺度特征提取块 (Multi-Scale Feature Extraction Block)
使用不同尺度的卷积核和膨胀率捕捉多尺度特征
"""
import torch
import torch.nn as nn


class MSFEB(nn.Module):
    """
    多尺度特征提取块
    三路并行卷积，不同卷积核大小和膨胀率
    """
    
    def __init__(self, in_channels, out_channels):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
        """
        super(MSFEB, self).__init__()
        
        # 三路并行卷积
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, 
                     padding=1, dilation=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=5, 
                     padding=4, dilation=2, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=7, 
                     padding=12, dilation=4, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 1x1卷积进行通道融合
        self.fusion = nn.Sequential(
            nn.Conv1d(out_channels * 3, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels)
        )
        
        # 残差连接的1x1卷积(如果通道数不匹配)
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
        # 三路并行
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        
        # 拼接
        concat = torch.cat([out1, out2, out3], dim=1)
        
        # 融合
        out = self.fusion(concat)
        
        # 残差连接
        shortcut = self.shortcut(x)
        out = self.relu(out + shortcut)
        
        return out


class InceptionBlock1D(nn.Module):
    """
    Inception-like模块 (1D版本，用于频域分支)
    """
    
    def __init__(self, in_channels, out_channels):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 每个分支的输出通道数
        """
        super(InceptionBlock1D, self).__init__()
        
        # Branch 1: 1x1卷积
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Branch 2: 1x1 -> 3x3
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Branch 3: 1x1 -> 5x5
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Branch 4: MaxPool -> 1x1
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 1x1卷积融合
        self.fusion = nn.Sequential(
            nn.Conv1d(out_channels * 4, out_channels * 4, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels * 4),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, C_in, L)
        Returns:
            out: (B, C_out*4, L)
        """
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        
        # 拼接所有分支
        concat = torch.cat([out1, out2, out3, out4], dim=1)
        
        # 融合
        out = self.fusion(concat)
        
        return out
