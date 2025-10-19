"""
测试重构后的三分支模型
验证所有分支都接受正确的输入格式（已预处理）
"""
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.backbone import (
    TemporalBranch,
    FrequencyBranch,
    TimeFrequencyBranch
)

def test_branches():
    """测试三个分支"""
    print("="  * 70)
    print("测试重构后的三分支模型")
    print("="  * 70)

    batch_size = 4

    # =================================================================
    # 1. 测试时域分支
    # =================================================================
    print("\n1. 时域分支 (Temporal Branch)")
    print("-" * 70)

    temporal_model = TemporalBranch(
        input_channels=1,
        seq_len=512,
        output_dim=256
    )

    # 输入：原始时域信号 (Dataset中通过IdentityTransform提供)
    temporal_input = torch.randn(batch_size, 1, 512)
    print(f"   输入: {temporal_input.shape} - 原始时域信号")

    temporal_output = temporal_model(temporal_input)
    print(f"   输出: {temporal_output.shape}")
    print(f"   参数量: {sum(p.numel() for p in temporal_model.parameters()):,}")
    print("   ✓ 时域分支测试通过")

    # =================================================================
    # 2. 测试频域分支
    # =================================================================
    print("\n2. 频域分支 (Frequency Branch)")
    print("-" * 70)

    frequency_model = FrequencyBranch(
        input_channels=1,
        freq_len=257,
        output_dim=256
    )

    # 输入：FFT频谱 (Dataset中通过FFTTransform提供)
    frequency_input = torch.randn(batch_size, 1, 257)
    print(f"   输入: {frequency_input.shape} - FFT频谱")

    frequency_output = frequency_model(frequency_input)
    print(f"   输出: {frequency_output.shape}")
    print(f"   参数量: {sum(p.numel() for p in frequency_model.parameters()):,}")
    print("   ✓ 频域分支测试通过")

    # =================================================================
    # 3. 测试时频分支
    # =================================================================
    print("\n3. 时频分支 (Time-Frequency Branch)")
    print("-" * 70)

    timefreq_model = TimeFrequencyBranch(
        input_channels=1,
        input_size=(64, 128),
        output_dim=256
    )

    # 输入：时频图 (Dataset中通过STFTTransform/CWTTransform提供)
    timefreq_input = torch.randn(batch_size, 1, 64, 128)
    print(f"   输入: {timefreq_input.shape} - STFT/CWT时频图")

    timefreq_output = timefreq_model(timefreq_input)
    print(f"   输出: {timefreq_output.shape}")
    print(f"   参数量: {sum(p.numel() for p in timefreq_model.parameters()):,}")
    print("   ✓ 时频分支测试通过")

    # =================================================================
    # 4. 模拟三分支联合
    # =================================================================
    print("\n4. 三分支联合输出")
    print("-" * 70)

    # 拼接三个分支的输出
    combined_features = torch.cat([
        temporal_output,
        frequency_output,
        timefreq_output
    ], dim=1)

    print(f"   融合前: {temporal_output.shape} + {frequency_output.shape} + {timefreq_output.shape}")
    print(f"   融合后: {combined_features.shape}")

    # =================================================================
    # 总结
    # =================================================================
    print("\n" + "=" * 70)
    print("✓ 所有分支测试通过！")
    print("=" * 70)

    print("\n关键信息:")
    print("  1. 时域分支: 接受原始信号 (B, 1, 512)")
    print("  2. 频域分支: 接受FFT频谱 (B, 1, 257)")
    print("  3. 时频分支: 接受时频图 (B, 1, 64, 128)")
    print("  4. 所有预处理在Dataset层完成")
    print("  5. 模型只做特征提取")

    total_params = (
        sum(p.numel() for p in temporal_model.parameters()) +
        sum(p.numel() for p in frequency_model.parameters()) +
        sum(p.numel() for p in timefreq_model.parameters())
    )
    print(f"\n总参数量: {total_params:,}")


if __name__ == "__main__":
    test_branches()
