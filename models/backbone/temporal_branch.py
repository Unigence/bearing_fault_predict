"""
时域分支 (Temporal Branch) - 重构版本
只负责特征提取，不包含信号预处理
用于捕捉时域冲击特征、周期性模式和局部异常
"""
import torch
import torch.nn as nn
from ..modules import (
    MSFEB,
    CBAM,
    TemporalSelfAttention,
    MultiHeadPooling1D
)


class TemporalBranch(nn.Module):
    """
    时域分支 - 纯特征提取网络

    输入: (B, 1, 512) - 原始时域信号（已在Dataset中提供）
    输出: (B, output_dim) - 时域特征向量

    架构:
    Stage 1: 宽卷积去噪层
    Stage 2: 多尺度特征提取块 (MSFEB × 2)
    Stage 3: 通道-空间协同注意力 (CBAM)
    Stage 4: 时序建模层 (Bi-GRU + 时序注意力)
    Stage 5: 特征聚合与压缩
    """

    def __init__(self,
                 input_channels=1,
                 seq_len=512,
                 output_dim=256,
                 gru_hidden=128,
                 gru_layers=2,
                 gru_dropout=0.2,
                 attention_heads=4):
        """
        Args:
            input_channels: 输入通道数 (默认1)
            seq_len: 输入序列长度
            output_dim: 输出特征维度
            gru_hidden: GRU隐藏层维度
            gru_layers: GRU层数
            gru_dropout: GRU dropout率
            attention_heads: 时序注意力头数
        """
        super(TemporalBranch, self).__init__()

        self.seq_len = seq_len
        self.output_dim = output_dim

        # Stage 1: 宽卷积去噪层
        self.stage1 = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=64, padding=31, bias=False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout1d(p=0.1)
        )

        # Stage 2: 多尺度特征提取块 (MSFEB × 2)
        self.msfeb1 = MSFEB(in_channels=32, out_channels=64)
        self.msfeb2 = MSFEB(in_channels=64, out_channels=128)
        self.dropout_stage2 = nn.Dropout1d(p=0.15)

        # Stage 3: 通道-空间协同注意力 (CBAM)
        self.cbam = CBAM(in_channels=128, reduction=8, kernel_size=7)

        # Stage 4: 时序建模层
        self.bi_gru = nn.GRU(
            input_size=128,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=gru_dropout if gru_layers > 1 else 0
        )

        self.temporal_attention = TemporalSelfAttention(
            embed_dim=gru_hidden * 2,
            num_heads=attention_heads,
            dropout=0.1
        )

        # Stage 5: 特征聚合与压缩
        self.multi_head_pooling = MultiHeadPooling1D(in_channels=gru_hidden * 2)

        pooled_dim = gru_hidden * 2 * 3
        self.feature_compression = nn.Sequential(
            nn.Linear(pooled_dim, 384),
            nn.BatchNorm1d(384),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(384, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        """
        Args:
            x: (B, 1, 512) - 时域信号（无需预处理）
        Returns:
            out: (B, output_dim) - 时域特征向量
        """
        # Stage 1: 宽卷积去噪
        x = self.stage1(x)  # (B, 32, 512)

        # Stage 2: 多尺度特征提取
        x = self.msfeb1(x)  # (B, 64, 512)
        x = self.dropout_stage2(x)
        x = self.msfeb2(x)  # (B, 128, 512)
        x = self.dropout_stage2(x)

        # Stage 3: CBAM注意力
        x = self.cbam(x)  # (B, 128, 512)

        # Stage 4: 时序建模
        x = x.transpose(1, 2)  # (B, 512, 128)
        gru_out, _ = self.bi_gru(x)  # (B, 512, 256)
        attn_out = self.temporal_attention(gru_out)  # (B, 512, 256)

        # Stage 5: 特征聚合
        pooled = self.multi_head_pooling(attn_out)  # (B, 768)
        out = self.feature_compression(pooled)  # (B, output_dim)

        return out


def create_temporal_branch(config='medium', **kwargs):
    """
    创建时域分支的工厂函数

    Args:
        config: 'small', 'medium', 'large'
        **kwargs: 额外参数

    Returns:
        TemporalBranch实例
    """
    configs = {
        'small': {
            'output_dim': 128,
            'gru_hidden': 64,
            'gru_layers': 1,
            'attention_heads': 2
        },
        'medium': {
            'output_dim': 256,
            'gru_hidden': 128,
            'gru_layers': 2,
            'attention_heads': 4
        },
        'large': {
            'output_dim': 512,
            'gru_hidden': 256,
            'gru_layers': 2,
            'attention_heads': 8
        }
    }

    params = configs.get(config, configs['medium'])
    params.update(kwargs)

    return TemporalBranch(**params)


if __name__ == "__main__":
    """测试代码"""
    print("=" * 70)
    print("时域分支模型测试 - 重构版本（无预处理）")
    print("=" * 70)

    # 创建模型
    model = TemporalBranch(seq_len=512, output_dim=256)

    print("\n时域分支结构:")
    print(model)

    # 参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")

    # 测试 - 注意：输入已经是时域信号（来自Dataset）
    batch_size = 4
    x = torch.randn(batch_size, 1, 512)  # 时域信号输入

    print(f"\n输入形状: {x.shape} (原始时域信号)")

    with torch.no_grad():
        out = model(x)

    print(f"输出形状: {out.shape}")
    print(f"输出范围: [{out.min().item():.4f}, {out.max().item():.4f}]")

    # 测试不同配置
    print("\n" + "=" * 70)
    print("测试不同配置")
    print("=" * 70)

    for config in ['small', 'medium', 'large']:
        model = create_temporal_branch(config=config)
        params = sum(p.numel() for p in model.parameters())
        print(f"\n{config.capitalize():8s} - 参数量: {params:,}")

        with torch.no_grad():
            out = model(x)
        print(f"           输出形状: {out.shape}")

    print("\n" + "=" * 70)
    print("✓ 时域分支测试完成!")
    print("=" * 70)
    print("\n注意: 此版本期望输入为原始时域信号 (B, 1, 512)")
    print("      信号来自Dataset的temporal字段，无需额外预处理")
