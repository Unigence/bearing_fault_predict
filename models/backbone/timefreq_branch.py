"""
时频分支 (Time-Frequency Branch) - 重构版本
只负责特征提取，不包含STFT/CWT预处理
用于捕捉时变频率特征、瞬态冲击和能量分布演变
"""
import torch
import torch.nn as nn
import numpy as np
from ..modules import (
    EMA,
    ResidualBlock2D,
    AdaptivePooling2D,
    ChannelAttention2D,
    SpatialAttention2D
)


class TimeFrequencyBranch(nn.Module):
    """
    时频分支 - 纯特征提取网络

    输入: (B, 1, 64, 128) - 时频图（已在Dataset中通过STFT/CWT提供）
    输出: (B, output_dim) - 时频特征向量

    架构:
    Stage 1: 轻量级CNN特征提取
    Stage 2: 高效多尺度注意力 (EMA)
    Stage 3: 自适应池化与特征压缩
    """

    def __init__(self,
                 input_channels=1,
                 input_size=(64, 128),
                 output_dim=256):
        """
        Args:
            input_channels: 输入通道数 (默认1)
            input_size: 输入时频图尺寸 (H, W)
            output_dim: 输出特征维度
        """
        super(TimeFrequencyBranch, self).__init__()

        self.input_size = input_size
        self.output_dim = output_dim

        # =====================================================================
        # Stage 1: 轻量级CNN Backbone
        # =====================================================================
        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)  # -> (32, 32, 64)
        )

        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.15),
            nn.MaxPool2d(kernel_size=2, stride=2)  # -> (64, 16, 32)
        )

        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2)  # -> (128, 8, 16)
        )

        # =====================================================================
        # Stage 2: 高效多尺度注意力 (EMA)
        # =====================================================================
        self.ema = EMA(in_channels=128, reduction=8)

        # =====================================================================
        # Stage 3: 自适应池化与特征压缩
        # =====================================================================
        # 自适应池化
        self.adaptive_pool = AdaptivePooling2D(
            output_size=(4, 4),
            pooling_type='both'
        )
        # 输出: (256, 4, 4) -> flatten to 4096

        # 特征压缩
        self.feature_compression = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(512, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.Dropout(0.2)
        )


    def forward(self, x):
        """
        Args:
            x: (B, 1, 64, 128) - 时频图输入（已经是STFT/CWT时频图，无需预处理）
        Returns:
            out: (B, output_dim) - 时频特征向量
        """
        # Stage 1: CNN特征提取
        x = self.block1(x)  # (B, 32, 32, 64)
        x = self.block2(x)  # (B, 64, 16, 32)
        x = self.block3(x)  # (B, 128, 8, 16)

        # Stage 2: EMA注意力
        x = self.ema(x)  # (B, 128, 8, 16)

        # Stage 3: 自适应池化
        x = self.adaptive_pool(x)  # (B, 256, 4, 4)

        # Flatten
        x = x.view(x.size(0), -1)  # (B, 4096)

        # 特征压缩
        out = self.feature_compression(x)  # (B, output_dim)

        return out


def create_timefreq_branch(config='medium', **kwargs):
    """
    创建时频分支的工厂函数

    Args:
        config: 'small', 'medium', 'large'
        **kwargs: 额外参数

    Returns:
        TimeFrequencyBranch实例
    """
    configs = {
        'small': {
            'output_dim': 128,
        },
        'medium': {
            'output_dim': 256,
        },
        'large': {
            'output_dim': 512,
        }
    }

    params = configs.get(config, configs['medium'])
    params.update(kwargs)

    return TimeFrequencyBranch(**params)


if __name__ == "__main__":
    """测试代码"""
    print("=" * 70)
    print("时频分支模型测试 - 重构版本（无预处理）")
    print("=" * 70)

    # 创建模型
    model = TimeFrequencyBranch(input_size=(64, 128), output_dim=256)

    print("\n时频分支结构:")
    print(model)

    # 参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n总参数量: {total_params:,}")

    # 测试 - 注意：输入已经是时频图，不是时域信号
    batch_size = 4
    x = torch.randn(batch_size, 1, 64, 128)  # 时频图输入

    print(f"\n输入形状: {x.shape} (已经是STFT/CWT时频图)")

    with torch.no_grad():
        out = model(x)

    print(f"输出形状: {out.shape}")
    print(f"输出范围: [{out.min().item():.4f}, {out.max().item():.4f}]")

    print("\n" + "=" * 70)
    print("✓ 时频分支测试完成!")
    print("=" * 70)
    print("\n注意: 此版本期望输入已经是时频图 (B, 1, 64, 128)")
    print("      STFT/CWT变换应在Dataset中通过STFTTransform/CWTTransform完成")
