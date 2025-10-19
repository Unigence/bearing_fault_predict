"""
频域分支 (Frequency Branch) - 重构版本
只负责特征提取，不包含FFT预处理
用于提取频谱峰值、谐波特征和频带能量分布
"""
import torch
import torch.nn as nn
import numpy as np
from ..modules import (
    InceptionBlock1D,
    SEBlock,
    ResidualBlock1D,
    GlobalPooling1D
)


class FrequencyBranch(nn.Module):
    """
    频域分支 - 纯特征提取网络

    输入: (B, 1, 257) - FFT频谱（已在Dataset中通过FFTTransform提供）
    输出: (B, output_dim) - 频域特征向量

    架构:
    Stage 1: 频域多尺度卷积 (Inception-like)
    Stage 2: 频谱增强注意力 (SE)
    Stage 3: 深度特征提取 (残差块 × 2)
    Stage 4: 全局特征聚合
    """

    def __init__(self,
                 input_channels=1,
                 freq_len=257,
                 output_dim=256):
        """
        Args:
            input_channels: 输入通道数 (默认1)
            freq_len: 频谱长度 (512点信号的单边频谱为257)
            output_dim: 输出特征维度
        """
        super(FrequencyBranch, self).__init__()

        self.freq_len = freq_len
        self.output_dim = output_dim

        # =====================================================================
        # Stage 1: 频域多尺度卷积
        # =====================================================================
        # 初始卷积
        self.initial_conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout1d(0.15)
        )

        # Inception-like多尺度模块
        self.inception = InceptionBlock1D(in_channels=64, out_channels=32)
        # 输出: 32*4 = 128通道

        # =====================================================================
        # Stage 2: 频谱增强注意力 (SE)
        # =====================================================================
        self.se_block = SEBlock(in_channels=128, reduction=8)

        # =====================================================================
        # Stage 3: 深度特征提取 (残差块 × 2)
        # =====================================================================
        self.residual1 = ResidualBlock1D(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            dropout=0.2
        )

        self.residual2 = ResidualBlock1D(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            dropout=0.2
        )

        # =====================================================================
        # Stage 4: 全局特征聚合
        # =====================================================================
        # 双路全局池化
        self.global_pool = GlobalPooling1D(pooling_type='both')
        # 输出: 128*2 = 256维

        # 特征压缩
        self.feature_compression = nn.Sequential(
            nn.Linear(256, 512),
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
            x: (B, 1, 257) - 频域输入（已经是FFT频谱，无需预处理）
        Returns:
            out: (B, output_dim) - 频域特征向量
        """
        # Stage 1: 多尺度卷积
        x = self.initial_conv(x)  # (B, 64, 257)
        x = self.inception(x)  # (B, 128, 257)

        # Stage 2: SE注意力
        x = self.se_block(x)  # (B, 128, 257)

        # Stage 3: 残差块
        x = self.residual1(x)  # (B, 128, 257)
        x = self.residual2(x)  # (B, 128, 257)

        # Stage 4: 全局池化
        x = self.global_pool(x)  # (B, 256)

        # 特征压缩
        out = self.feature_compression(x)  # (B, output_dim)

        return out


class LightweightFrequencyBranch(nn.Module):
    """
    轻量级频域分支 - 纯特征提取
    更少的参数，更快的推理速度

    输入: (B, 1, 257) - FFT频谱（已预处理）
    输出: (B, output_dim) - 频域特征向量
    """

    def __init__(self,
                 input_channels=1,
                 freq_len=257,
                 output_dim=256):
        """
        Args:
            input_channels: 输入通道数
            freq_len: 频谱长度
            output_dim: 输出特征维度
        """
        super(LightweightFrequencyBranch, self).__init__()

        self.freq_len = freq_len
        self.output_dim = output_dim

        # 简化的卷积网络
        self.conv_layers = nn.Sequential(
            # Layer 1
            nn.Conv1d(1, 32, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout1d(0.15),

            # Layer 2
            nn.Conv1d(32, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout1d(0.2),

            # Layer 3
            nn.Conv1d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout1d(0.2),
        )

        # SE注意力
        self.se = SEBlock(128, reduction=8)

        # 全局池化
        self.global_pool = GlobalPooling1D(pooling_type='both')

        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(256, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.Dropout(0.3)
        )


    def forward(self, x):
        """
        Args:
            x: (B, 1, 257) - 频域输入（已经是FFT频谱）
        Returns:
            out: (B, output_dim) - 频域特征向量
        """
        # 卷积特征提取
        x = self.conv_layers(x)  # (B, 128, 257)

        # SE注意力
        x = self.se(x)  # (B, 128, 257)

        # 全局池化
        x = self.global_pool(x)  # (B, 256)

        # 输出
        out = self.fc(x)  # (B, output_dim)

        return out


def create_frequency_branch(config='medium', **kwargs):
    """
    创建频域分支的工厂函数

    Args:
        config: 'small', 'medium', 'large'
        **kwargs: 额外参数

    Returns:
        FrequencyBranch实例
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

    return FrequencyBranch(**params)


if __name__ == "__main__":
    """测试代码"""
    print("=" * 70)
    print("频域分支模型测试 - 重构版本（无预处理）")
    print("=" * 70)

    # 测试标准频域分支
    model = FrequencyBranch(freq_len=257, output_dim=256)

    print("\n标准频域分支结构:")
    print(model)

    # 参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n总参数量: {total_params:,}")

    # 测试 - 注意：输入已经是频谱，不是时域信号
    batch_size = 4
    x = torch.randn(batch_size, 1, 257)  # 频谱输入

    print(f"\n输入形状: {x.shape} (已经是FFT频谱)")

    with torch.no_grad():
        out = model(x)

    print(f"输出形状: {out.shape}")
    print(f"输出范围: [{out.min().item():.4f}, {out.max().item():.4f}]")

    # 测试轻量级版本
    print("\n" + "=" * 70)
    print("轻量级频域分支")
    print("=" * 70)

    light_model = LightweightFrequencyBranch(freq_len=257, output_dim=256)
    light_params = sum(p.numel() for p in light_model.parameters())
    print(f"轻量级参数量: {light_params:,}")

    with torch.no_grad():
        light_out = light_model(x)
    print(f"输出形状: {light_out.shape}")

    print("\n" + "=" * 70)
    print("✓ 频域分支测试完成!")
    print("=" * 70)
    print("\n注意: 此版本期望输入已经是FFT频谱 (B, 1, 257)")
    print("      FFT变换应在Dataset中通过FFTTransform完成")
