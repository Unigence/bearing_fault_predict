"""
频域数据增强 (Frequency-Domain Augmentation)
包含:频谱遮蔽、随机滤波、相位扰动等
"""
import torch
import numpy as np
from scipy import signal as scipy_signal


class FrequencyAugmentation:
    """频域增强基类"""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, signal):
        if np.random.rand() > self.prob:
            return signal
        return self.apply(signal)

    def apply(self, signal):
        raise NotImplementedError


class FrequencyMasking(FrequencyAugmentation):
    """
    频谱遮蔽 (SpecAugment for 1D)
    随机遮蔽频率带
    """

    def __init__(self,
                 mask_param=(5, 20),
                 num_masks=(1, 2),
                 prob=0.4):
        """
        Args:
            mask_param: 遮蔽频率bin数量范围 (min, max)
            num_masks: 遮蔽段数范围 (min, max) - 注意是闭区间
            prob: 应用概率
        """
        super().__init__(prob)
        self.mask_param = mask_param
        self.num_masks = num_masks

    def apply(self, signal):
        """
        在频域进行遮蔽

        Args:
            signal: (seq_len,) 时域信号
        Returns:
            masked_signal: 遮蔽后的信号
        """
        # FFT
        fft_result = torch.fft.rfft(signal)
        freq_len = len(fft_result)

        # 🔧 修复: 处理 num_masks 的范围问题
        min_masks, max_masks = self.num_masks
        if min_masks == max_masks:
            # 如果min == max,直接使用该值
            n_masks = min_masks
        else:
            # 否则使用randint,注意max_masks是不包含的
            # randint(a, b)返回[a, b),所以需要+1
            n_masks = np.random.randint(min_masks, max_masks + 1)

        for _ in range(n_masks):
            # 随机遮蔽长度
            min_len, max_len = self.mask_param
            if min_len == max_len:
                mask_len = min_len
            else:
                mask_len = np.random.randint(min_len, max_len + 1)

            # 检查是否超出频谱长度
            if mask_len >= freq_len:
                mask_len = freq_len // 2

            if freq_len - mask_len <= 0:
                continue

            # 随机起始位置
            start = np.random.randint(0, freq_len - mask_len)
            end = start + mask_len

            # 遮蔽频率
            fft_result[start:end] = 0

        # 逆FFT
        masked_signal = torch.fft.irfft(fft_result, n=len(signal))

        return masked_signal


class MagnitudeMasking(FrequencyAugmentation):
    """
    幅度遮蔽
    随机衰减某些频率的幅度
    """

    def __init__(self,
                 attenuation=(0.3, 0.7),
                 mask_param=(10, 30),
                 num_masks=(1, 2),
                 prob=0.3):
        """
        Args:
            attenuation: 衰减因子范围 (min, max)
            mask_param: 遮蔽频率bin数量范围
            num_masks: 遮蔽段数范围
            prob: 应用概率
        """
        super().__init__(prob)
        self.attenuation = attenuation
        self.mask_param = mask_param
        self.num_masks = num_masks

    def apply(self, signal):
        """
        Args:
            signal: (seq_len,) 时域信号
        Returns:
            attenuated_signal: 衰减后的信号
        """
        # FFT
        fft_result = torch.fft.rfft(signal)
        freq_len = len(fft_result)

        # 🔧 修复: 处理 num_masks 的范围问题
        min_masks, max_masks = self.num_masks
        if min_masks == max_masks:
            n_masks = min_masks
        else:
            n_masks = np.random.randint(min_masks, max_masks + 1)

        for _ in range(n_masks):
            # 随机遮蔽长度和衰减因子
            min_len, max_len = self.mask_param
            if min_len == max_len:
                mask_len = min_len
            else:
                mask_len = np.random.randint(min_len, max_len + 1)

            atten_factor = np.random.uniform(*self.attenuation)

            # 检查是否超出频谱长度
            if mask_len >= freq_len:
                mask_len = freq_len // 2

            if freq_len - mask_len <= 0:
                continue

            # 随机起始位置
            start = np.random.randint(0, freq_len - mask_len)
            end = start + mask_len

            # 衰减幅度
            fft_result[start:end] *= atten_factor

        # 逆FFT
        attenuated_signal = torch.fft.irfft(fft_result, n=len(signal))

        return attenuated_signal


class RandomFiltering(FrequencyAugmentation):
    """
    随机滤波
    应用带通或带阻滤波器
    """

    def __init__(self,
                 filter_type='bandpass',
                 freq_range=(500, 8000),
                 sampling_rate=20480,
                 prob=0.35):
        """
        Args:
            filter_type: 'bandpass' 或 'bandstop'
            freq_range: 频率范围 (min, max) in Hz
            sampling_rate: 采样率
            prob: 应用概率
        """
        super().__init__(prob)
        self.filter_type = filter_type
        self.freq_range = freq_range
        self.sampling_rate = sampling_rate

    def apply(self, signal):
        """
        Args:
            signal: (seq_len,) 时域信号
        Returns:
            filtered: 滤波后的信号
        """
        # 转换为numpy
        if isinstance(signal, torch.Tensor):
            signal_np = signal.numpy()
            is_tensor = True
        else:
            signal_np = signal
            is_tensor = False

        # 随机频率范围
        f1 = np.random.uniform(*self.freq_range)
        f2 = np.random.uniform(f1, self.freq_range[1])

        # 归一化频率
        nyq = self.sampling_rate / 2
        low = f1 / nyq
        high = f2 / nyq

        # 设计滤波器
        if self.filter_type == 'bandpass':
            b, a = scipy_signal.butter(4, [low, high], btype='band')
        elif self.filter_type == 'bandstop':
            b, a = scipy_signal.butter(4, [low, high], btype='bandstop')
        else:
            raise ValueError(f"Unknown filter type: {self.filter_type}")

        # 应用滤波器
        filtered = scipy_signal.filtfilt(b, a, signal_np)

        # 转回tensor
        if is_tensor:
            filtered = np.ascontiguousarray(filtered)
            filtered = torch.from_numpy(filtered).float()

        return filtered


class PhaseShift(FrequencyAugmentation):
    """
    相位扰动
    在频域添加随机相位
    """

    def __init__(self,
                 phase_range=(-np.pi/8, np.pi/8),
                 prob=0.25):
        """
        Args:
            phase_range: 相位扰动范围 (min, max) in radians
            prob: 应用概率
        """
        super().__init__(prob)
        self.phase_range = phase_range

    def apply(self, signal):
        """
        Args:
            signal: (seq_len,) 时域信号
        Returns:
            phase_shifted: 相位扰动后的信号
        """
        # FFT
        fft_result = torch.fft.rfft(signal)

        # 提取幅度和相位
        magnitude = torch.abs(fft_result)
        phase = torch.angle(fft_result)

        # 添加随机相位扰动
        phase_noise = torch.empty_like(phase).uniform_(*self.phase_range)
        new_phase = phase + phase_noise

        # 重构复数频谱
        new_fft = magnitude * torch.exp(1j * new_phase)

        # 逆FFT
        phase_shifted = torch.fft.irfft(new_fft, n=len(signal))

        return phase_shifted


class FrequencyShift(FrequencyAugmentation):
    """
    频率偏移
    整体移动频谱
    """

    def __init__(self,
                 shift_range=(-10, 10),
                 prob=0.2):
        """
        Args:
            shift_range: 频率bin偏移范围
            prob: 应用概率
        """
        super().__init__(prob)
        self.shift_range = shift_range

    def apply(self, signal):
        """
        Args:
            signal: (seq_len,) 时域信号
        Returns:
            shifted: 频率偏移后的信号
        """
        # FFT
        fft_result = torch.fft.rfft(signal)

        # 随机偏移量
        min_shift, max_shift = self.shift_range
        if min_shift == max_shift:
            shift = min_shift
        else:
            shift = np.random.randint(min_shift, max_shift + 1)

        # 频谱偏移
        shifted_fft = torch.roll(fft_result, shifts=shift, dims=0)

        # 逆FFT
        shifted_signal = torch.fft.irfft(shifted_fft, n=len(signal))

        return shifted_signal


class FreqCompose:
    """组合多个频域增强"""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, signal):
        for t in self.transforms:
            signal = t(signal)
        return signal


def get_frequency_augmentation(intensity='medium'):
    """
    获取频域增强组合

    Args:
        intensity: 'weak', 'medium', 'strong'
    """
    if intensity == 'weak':
        return FreqCompose([
            FrequencyMasking(mask_param=(5, 10), num_masks=(1, 2), prob=0.2),  # 修复:(1,1)->(1,2)
            MagnitudeMasking(attenuation=(0.5, 0.8), prob=0.2),
        ])

    elif intensity == 'medium':
        return FreqCompose([
            FrequencyMasking(mask_param=(5, 15), num_masks=(1, 2), prob=0.3),
            MagnitudeMasking(attenuation=(0.4, 0.7), prob=0.3),
            RandomFiltering(filter_type='bandpass', prob=0.25),
        ])

    elif intensity == 'strong':
        return FreqCompose([
            FrequencyMasking(mask_param=(5, 20), num_masks=(1, 3), prob=0.4),  # 修复:(1,2)->(1,3)
            MagnitudeMasking(attenuation=(0.3, 0.7), prob=0.3),
            RandomFiltering(filter_type='bandpass', prob=0.35),
            PhaseShift(phase_range=(-np.pi/8, np.pi/8), prob=0.25),
        ])

    else:
        raise ValueError(f"Unknown intensity: {intensity}")


if __name__ == "__main__":
    """测试代码"""

    # 创建测试信号
    t = torch.linspace(0, 1, 512)
    signal = torch.sin(2 * np.pi * 100 * t) + 0.5 * torch.sin(2 * np.pi * 300 * t)

    print("=" * 70)
    print("测试频域数据增强 (修复版)")
    print("=" * 70)

    # 测试各种增强
    augmentations = {
        'FrequencyMasking': FrequencyMasking(prob=1.0),
        'MagnitudeMasking': MagnitudeMasking(prob=1.0),
        'RandomFiltering': RandomFiltering(prob=1.0),
        'PhaseShift': PhaseShift(prob=1.0),
        'FrequencyShift': FrequencyShift(prob=1.0),
    }

    for name, aug in augmentations.items():
        try:
            augmented = aug(signal)
            print(f"✓ {name:20s}: 输入 {signal.shape} → 输出 {augmented.shape}")
        except Exception as e:
            print(f"✗ {name:20s}: 失败 - {e}")

    # 测试组合增强
    print("\n测试组合增强:")
    for intensity in ['weak', 'medium', 'strong']:
        aug = get_frequency_augmentation(intensity)
        aug_signal = aug(signal)
        print(f"✓ {intensity.capitalize():8s}: {signal.shape} → {aug_signal.shape}")

    # 测试边界情况
    print("\n测试边界情况:")
    edge_cases = [
        ('num_masks=(1,1)', FrequencyMasking(mask_param=(5, 10), num_masks=(1, 1), prob=1.0)),
        ('num_masks=(1,2)', FrequencyMasking(mask_param=(5, 10), num_masks=(1, 2), prob=1.0)),
        ('num_masks=(2,2)', FrequencyMasking(mask_param=(5, 10), num_masks=(2, 2), prob=1.0)),
    ]

    for desc, aug in edge_cases:
        try:
            augmented = aug(signal)
            print(f"✓ {desc:20s}: 成功")
        except Exception as e:
            print(f"✗ {desc:20s}: 失败 - {e}")

    print("\n" + "=" * 70)
    print("✅ 频域数据增强测试完成!")
    print("=" * 70)
