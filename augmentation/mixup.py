"""
Mixup数据增强
支持时域和频域的Mixup增强
"""
import torch
import numpy as np
from typing import Tuple, Optional


class TimeDomainMixup:
    """
    时域Mixup增强
    在时域信号上进行混合
    """

    def __init__(self, alpha: float = 0.2, prob: float = 0.5):
        """
        Args:
            alpha: Beta分布参数，控制混合强度
            prob: 应用Mixup的概率
        """
        self.alpha = alpha
        self.prob = prob

    def __call__(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        y1: torch.Tensor,
        y2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        对时域信号进行mixup

        Args:
            x1, x2: 时域信号 (seq_len,) 或 (C, seq_len)
            y1, y2: 标签（标量或one-hot）

        Returns:
            x_mix: 混合后的时域信号
            y_mix: 混合后的标签
            lam: 混合系数
        """
        # 判断是否应用
        if np.random.rand() > self.prob:
            return x1, y1, 1.0

        # 从Beta分布采样混合系数
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        # 时域线性混合
        x_mix = lam * x1 + (1 - lam) * x2

        # 标签混合
        y_mix = lam * y1 + (1 - lam) * y2

        return x_mix, y_mix, lam


class FrequencyDomainMixup:
    """
    频域Mixup增强

    注意：频域mixup需要谨慎使用，因为：
    1. 频谱有特定的物理意义（能量、相位）
    2. 简单的线性混合可能破坏频谱结构
    3. 对于时频图（2D），直接混合可能导致不自然的模式

    当前实现：
    - 对频域幅度谱进行混合（保持相位信息）
    - 或对时频图的低频部分进行选择性混合
    """

    def __init__(
        self,
        alpha: float = 0.2,
        prob: float = 0.3,
        mix_mode: str = 'magnitude'
    ):
        """
        Args:
            alpha: Beta分布参数
            prob: 应用概率
            mix_mode: 混合模式
                - 'magnitude': 仅混合幅度谱（推荐）
                - 'full': 完全混合（可能破坏频谱特性）
                - 'low_freq': 仅混合低频部分
        """
        self.alpha = alpha
        self.prob = prob
        self.mix_mode = mix_mode

    def __call__(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        y1: torch.Tensor,
        y2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        对频域表示进行mixup

        Args:
            x1, x2: 频域表示 (freq_bins,) 或 (C, freq_bins) 或 (C, H, W) 时频图
            y1, y2: 标签

        Returns:
            x_mix: 混合后的频域表示
            y_mix: 混合后的标签
            lam: 混合系数
        """
        # 判断是否应用
        if np.random.rand() > self.prob:
            return x1, y1, 1.0

        # 从Beta分布采样
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        if self.mix_mode == 'magnitude':
            # 仅混合幅度（对2D时频图也适用）
            x_mix = lam * x1 + (1 - lam) * x2

        elif self.mix_mode == 'low_freq':
            # 仅混合低频部分（对1D频谱）
            if x1.dim() == 1:
                # 1D频谱
                cutoff = x1.size(0) // 4  # 低频部分（前25%）
                x_mix = x1.clone()
                x_mix[:cutoff] = lam * x1[:cutoff] + (1 - lam) * x2[:cutoff]
            elif x1.dim() == 2:
                # 2D时频图 (H, W)
                h_cutoff = x1.size(0) // 4
                w_cutoff = x1.size(1) // 4
                x_mix = x1.clone()
                x_mix[:h_cutoff, :w_cutoff] = (
                    lam * x1[:h_cutoff, :w_cutoff] +
                    (1 - lam) * x2[:h_cutoff, :w_cutoff]
                )
            else:
                # 3D (C, H, W)
                h_cutoff = x1.size(1) // 4
                w_cutoff = x1.size(2) // 4
                x_mix = x1.clone()
                x_mix[:, :h_cutoff, :w_cutoff] = (
                    lam * x1[:, :h_cutoff, :w_cutoff] +
                    (1 - lam) * x2[:, :h_cutoff, :w_cutoff]
                )
        else:
            # 完全混合（可能破坏频谱特性，慎用）
            x_mix = lam * x1 + (1 - lam) * x2

        # 标签混合
        y_mix = lam * y1 + (1 - lam) * y2

        return x_mix, y_mix, lam


class ManifoldMixup:
    """
    流形Mixup（特征层mixup）
    在中间层特征空间进行混合
    """

    def __init__(self, alpha: float = 0.2, prob: float = 0.5):
        self.alpha = alpha
        self.prob = prob

    def __call__(
        self,
        features1: torch.Tensor,
        features2: torch.Tensor,
        y1: torch.Tensor,
        y2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        在特征空间混合

        Args:
            features1, features2: 中间层特征
            y1, y2: 标签

        Returns:
            mixed_features: 混合特征
            mixed_y: 混合标签
            lam: 混合系数
        """
        if np.random.rand() > self.prob:
            return features1, y1, 1.0

        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        mixed_features = lam * features1 + (1 - lam) * features2
        mixed_y = lam * y1 + (1 - lam) * y2

        return mixed_features, mixed_y, lam


class MultiModalMixup:
    """
    多模态Mixup管理器
    根据配置选择性地对时域、频域进行mixup
    """

    def __init__(
        self,
        time_domain_config: Optional[dict] = None,
        frequency_domain_config: Optional[dict] = None,
        feature_level_config: Optional[dict] = None
    ):
        """
        Args:
            time_domain_config: 时域mixup配置
                {
                    'enable': bool,
                    'alpha': float,
                    'prob': float
                }
            frequency_domain_config: 频域mixup配置
                {
                    'enable': bool,
                    'alpha': float,
                    'prob': float,
                    'mix_mode': str  # 'magnitude', 'low_freq', 'full'
                }
            feature_level_config: 特征层mixup配置
                {
                    'enable': bool,
                    'alpha': float,
                    'prob': float
                }
        """
        # 时域mixup
        self.time_domain_enabled = False
        self.time_mixup = None
        if time_domain_config and time_domain_config.get('enable', False):
            self.time_domain_enabled = True
            self.time_mixup = TimeDomainMixup(
                alpha=time_domain_config.get('alpha', 0.2),
                prob=time_domain_config.get('prob', 0.5)
            )

        # 频域mixup
        self.frequency_domain_enabled = False
        self.frequency_mixup = None
        if frequency_domain_config and frequency_domain_config.get('enable', False):
            self.frequency_domain_enabled = True
            self.frequency_mixup = FrequencyDomainMixup(
                alpha=frequency_domain_config.get('alpha', 0.2),
                prob=frequency_domain_config.get('prob', 0.3),
                mix_mode=frequency_domain_config.get('mix_mode', 'magnitude')
            )

        # 特征层mixup
        self.feature_level_enabled = False
        self.feature_mixup = None
        if feature_level_config and feature_level_config.get('enable', False):
            self.feature_level_enabled = True
            self.feature_mixup = ManifoldMixup(
                alpha=feature_level_config.get('alpha', 0.2),
                prob=feature_level_config.get('prob', 0.5)
            )

    def apply_input_mixup(
        self,
        batch: dict,
        device: str = 'cuda'
    ) -> Tuple[dict, Optional[torch.Tensor], Optional[torch.Tensor], float]:
        """
        对输入batch应用mixup

        Args:
            batch: 输入batch字典，包含'temporal', 'frequency', 'label'等
            device: 设备

        Returns:
            mixed_batch: 混合后的batch
            labels_a: 第一个标签（用于损失计算）
            labels_b: 第二个标签（用于损失计算）
            lam: 混合系数
        """
        labels = batch['label'].to(device)
        batch_size = labels.size(0)

        # 如果都不启用，直接返回
        if not self.time_domain_enabled and not self.frequency_domain_enabled:
            return batch, labels, None, 1.0

        # 随机配对：随机打乱索引
        indices = torch.randperm(batch_size).to(device)

        # 决定是否应用mixup（至少有一个启用）
        apply_mixup = False
        lam = 1.0

        # 时域mixup
        if self.time_domain_enabled and 'temporal' in batch:
            temporal = batch['temporal'].to(device)

            # 对batch中的每对样本应用mixup
            if np.random.rand() < self.time_mixup.prob:
                apply_mixup = True

                # 采样lambda
                if self.time_mixup.alpha > 0:
                    lam = np.random.beta(self.time_mixup.alpha, self.time_mixup.alpha)
                else:
                    lam = 1.0

                # 混合
                mixed_temporal = lam * temporal + (1 - lam) * temporal[indices]
                batch['temporal'] = mixed_temporal

        # 频域mixup
        if self.frequency_domain_enabled and 'frequency' in batch:
            frequency = batch['frequency'].to(device)

            # 对batch中的每对样本应用mixup
            if np.random.rand() < self.frequency_mixup.prob:
                apply_mixup = True

                # 如果时域已经采样了lambda，使用相同的lambda保持一致性
                if lam == 1.0:
                    if self.frequency_mixup.alpha > 0:
                        lam = np.random.beta(
                            self.frequency_mixup.alpha,
                            self.frequency_mixup.alpha
                        )

                # 根据mix_mode混合
                if self.frequency_mixup.mix_mode == 'magnitude':
                    mixed_frequency = lam * frequency + (1 - lam) * frequency[indices]
                elif self.frequency_mixup.mix_mode == 'low_freq':
                    # 仅混合低频部分
                    mixed_frequency = frequency.clone()
                    if frequency.dim() == 3:  # (B, C, L)
                        cutoff = frequency.size(2) // 4
                        mixed_frequency[:, :, :cutoff] = (
                            lam * frequency[:, :, :cutoff] +
                            (1 - lam) * frequency[indices, :, :cutoff]
                        )
                    elif frequency.dim() == 4:  # (B, C, H, W)
                        h_cutoff = frequency.size(2) // 4
                        w_cutoff = frequency.size(3) // 4
                        mixed_frequency[:, :, :h_cutoff, :w_cutoff] = (
                            lam * frequency[:, :, :h_cutoff, :w_cutoff] +
                            (1 - lam) * frequency[indices, :, :h_cutoff, :w_cutoff]
                        )
                else:  # 'full'
                    mixed_frequency = lam * frequency + (1 - lam) * frequency[indices]

                batch['frequency'] = mixed_frequency

        # 返回混合后的batch和标签信息
        if apply_mixup:
            labels_a = labels
            labels_b = labels[indices]
            return batch, labels_a, labels_b, lam
        else:
            return batch, labels, None, 1.0

    def apply_feature_mixup(
        self,
        features: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        对特征层应用mixup

        Args:
            features: 特征张量 (B, D)
            labels: 标签 (B,)

        Returns:
            mixed_features: 混合后的特征
            labels_a: 第一个标签
            labels_b: 第二个标签
            lam: 混合系数
        """
        if not self.feature_level_enabled:
            return features, labels, None, 1.0

        if np.random.rand() > self.feature_mixup.prob:
            return features, labels, None, 1.0

        batch_size = features.size(0)
        indices = torch.randperm(batch_size).to(features.device)

        # 采样lambda
        if self.feature_mixup.alpha > 0:
            lam = np.random.beta(self.feature_mixup.alpha, self.feature_mixup.alpha)
        else:
            lam = 1.0

        # 混合特征
        mixed_features = lam * features + (1 - lam) * features[indices]

        # 返回混合信息
        labels_a = labels
        labels_b = labels[indices]

        return mixed_features, labels_a, labels_b, lam

if __name__ == "__main__":
    """测试代码"""

    print("测试Mixup数据增强")
    print("=" * 70)

    # 测试时域mixup
    print("\n1. 时域Mixup")
    time_mixup = TimeDomainMixup(alpha=0.2, prob=1.0)
    x1 = torch.randn(512)  # 时域信号
    x2 = torch.randn(512)
    y1 = torch.tensor(0)
    y2 = torch.tensor(1)
    x_mix, y_mix, lam = time_mixup(x1, x2, y1, y2)
    print(f"   时域信号: {x1.shape} -> {x_mix.shape}, λ={lam:.3f}")

    # 测试频域mixup
    print("\n2. 频域Mixup")
    freq_mixup = FrequencyDomainMixup(alpha=0.2, prob=1.0, mix_mode='magnitude')
    f1 = torch.randn(1, 128, 128)  # 时频图
    f2 = torch.randn(1, 128, 128)
    f_mix, y_mix, lam = freq_mixup(f1, f2, y1, y2)
    print(f"   时频图: {f1.shape} -> {f_mix.shape}, λ={lam:.3f}")

    # 测试多模态mixup
    print("\n3. 多模态Mixup")
    multi_mixup = MultiModalMixup(
        time_domain_config={'enable': True, 'alpha': 0.2, 'prob': 0.5},
        frequency_domain_config={'enable': True, 'alpha': 0.2, 'prob': 0.3}
    )

    batch = {
        'temporal': torch.randn(8, 1, 512),
        'frequency': torch.randn(8, 1, 128, 128),
        'label': torch.randint(0, 6, (8,))
    }

    mixed_batch, labels_a, labels_b, lam = multi_mixup.apply_input_mixup(batch, 'cpu')
    print(f"   混合后temporal: {mixed_batch['temporal'].shape}")
    print(f"   混合后frequency: {mixed_batch['frequency'].shape}")
    print(f"   λ={lam:.3f}")

    print("\n✓ Mixup测试完成!")
