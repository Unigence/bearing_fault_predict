"""
时域数据增强 (Time-Domain Augmentation)
包含：高斯噪声、时间偏移、幅值缩放、时间拉伸、随机掩码、添加冲击等
"""
import torch
import numpy as np
from scipy import interpolate


class TimeAugmentation:
    """时域增强基类"""
    
    def __init__(self, prob=0.5):
        """
        Args:
            prob: 应用增强的概率
        """
        self.prob = prob
    
    def __call__(self, signal):
        """
        Args:
            signal: (seq_len,) 时域信号
        Returns:
            augmented: 增强后的信号
        """
        if np.random.rand() > self.prob:
            return signal
        return self.apply(signal)
    
    def apply(self, signal):
        raise NotImplementedError


class GaussianNoise(TimeAugmentation):
    """
    高斯噪声注入
    模拟传感器噪声
    """
    
    def __init__(self, sigma_range=(0.01, 0.05), prob=0.6):
        """
        Args:
            sigma_range: 噪声标准差范围 (min, max)
            prob: 应用概率
        """
        super().__init__(prob)
        self.sigma_range = sigma_range
    
    def apply(self, signal):
        """
        Args:
            signal: (seq_len,) torch tensor
        Returns:
            noisy_signal: 添加噪声后的信号
        """
        # 随机选择噪声强度
        sigma = np.random.uniform(*self.sigma_range)
        
        # 生成噪声
        noise = torch.randn_like(signal) * sigma
        
        return signal + noise


class TimeShift(TimeAugmentation):
    """
    时间偏移
    增强时移不变性
    """
    
    def __init__(self, shift_range=(-50, 50), prob=0.5):
        """
        Args:
            shift_range: 偏移范围 (min, max) in samples
            prob: 应用概率
        """
        super().__init__(prob)
        self.shift_range = shift_range
    
    def apply(self, signal):
        """
        Args:
            signal: (seq_len,) torch tensor
        Returns:
            shifted: 偏移后的信号
        """
        # 随机偏移量
        shift = np.random.randint(*self.shift_range)
        
        # 循环移位
        shifted = torch.roll(signal, shifts=shift, dims=0)
        
        return shifted


class AmplitudeScale(TimeAugmentation):
    """
    幅值缩放
    模拟不同负载条件
    """
    
    def __init__(self, scale_range=(0.8, 1.2), prob=0.7):
        """
        Args:
            scale_range: 缩放范围 (min, max)
            prob: 应用概率
        """
        super().__init__(prob)
        self.scale_range = scale_range
    
    def apply(self, signal):
        """
        Args:
            signal: (seq_len,) torch tensor
        Returns:
            scaled: 缩放后的信号
        """
        # 随机缩放因子
        scale = np.random.uniform(*self.scale_range)
        
        return signal * scale


class TimeWarping(TimeAugmentation):
    """
    时间拉伸/压缩
    模拟转速变化
    """
    
    def __init__(self, warp_range=(0.9, 1.1), prob=0.4):
        """
        Args:
            warp_range: 拉伸比例范围 (min, max)
            prob: 应用概率
        """
        super().__init__(prob)
        self.warp_range = warp_range
    
    def apply(self, signal):
        """
        Args:
            signal: (seq_len,) torch tensor
        Returns:
            warped: 拉伸后的信号
        """
        # 转换为numpy进行插值
        if isinstance(signal, torch.Tensor):
            signal_np = signal.numpy()
            is_tensor = True
        else:
            signal_np = signal
            is_tensor = False
        
        seq_len = len(signal_np)
        
        # 随机拉伸比例
        ratio = np.random.uniform(*self.warp_range)
        
        # 原始时间轴
        x_old = np.arange(seq_len)
        
        # 新时间轴
        new_len = int(seq_len * ratio)
        x_new = np.linspace(0, seq_len - 1, new_len)
        
        # 三次样条插值
        f = interpolate.interp1d(x_old, signal_np, kind='cubic', fill_value='extrapolate')
        warped = f(x_new)
        
        # 裁剪或填充到原始长度
        if new_len > seq_len:
            warped = warped[:seq_len]
        elif new_len < seq_len:
            warped = np.pad(warped, (0, seq_len - new_len), mode='edge')
        
        # 转回tensor
        if is_tensor:
            warped = torch.from_numpy(warped).float()
        
        return warped


class RandomMasking(TimeAugmentation):
    """
    随机掩码
    将部分时间段置零，增强鲁棒性
    """
    
    def __init__(self, mask_ratio=0.1, num_masks=(1, 3), prob=0.3):
        """
        Args:
            mask_ratio: 总掩蔽比例
            num_masks: 掩蔽段数范围 (min, max)
            prob: 应用概率
        """
        super().__init__(prob)
        self.mask_ratio = mask_ratio
        self.num_masks = num_masks
    
    def apply(self, signal):
        """
        Args:
            signal: (seq_len,) torch tensor
        Returns:
            masked: 掩码后的信号
        """
        seq_len = len(signal)
        masked = signal.clone()
        
        # 随机掩蔽段数
        n_masks = np.random.randint(*self.num_masks)
        
        # 每段的长度
        total_mask_len = int(seq_len * self.mask_ratio)
        mask_len_per_segment = total_mask_len // n_masks
        
        for _ in range(n_masks):
            # 随机起始位置
            start = np.random.randint(0, seq_len - mask_len_per_segment)
            end = start + mask_len_per_segment
            
            # 掩蔽
            masked[start:end] = 0
        
        return masked


class AddImpulse(TimeAugmentation):
    """
    添加冲击
    模拟瞬态故障
    """
    
    def __init__(self, 
                 amplitude_range=(0.5, 2.0),
                 decay_range=(50, 200),
                 freq_range=(1000, 5000),
                 num_impulses=(1, 2),
                 prob=0.3):
        """
        Args:
            amplitude_range: 冲击幅度范围 (相对于信号最大值)
            decay_range: 衰减系数范围
            freq_range: 冲击频率范围 (Hz)
            num_impulses: 冲击数量范围
            prob: 应用概率
        """
        super().__init__(prob)
        self.amplitude_range = amplitude_range
        self.decay_range = decay_range
        self.freq_range = freq_range
        self.num_impulses = num_impulses
    
    def apply(self, signal):
        """
        Args:
            signal: (seq_len,) torch tensor
        Returns:
            signal_with_impulse: 添加冲击后的信号
        """
        seq_len = len(signal)
        augmented = signal.clone()
        
        # 信号最大幅值
        max_amp = torch.abs(signal).max()
        
        # 随机冲击数量
        n_impulses = np.random.randint(*self.num_impulses)
        
        for _ in range(n_impulses):
            # 随机参数
            amplitude = np.random.uniform(*self.amplitude_range) * max_amp
            decay = np.random.uniform(*self.decay_range)
            freq = np.random.uniform(*self.freq_range)
            
            # 随机位置
            pos = np.random.randint(0, seq_len - 100)
            
            # 生成冲击信号
            t = torch.arange(100).float()
            impulse = amplitude * torch.exp(-decay * t / 1000) * torch.cos(2 * np.pi * freq * t / 20480)
            
            # 添加冲击
            end_pos = min(pos + 100, seq_len)
            augmented[pos:end_pos] += impulse[:end_pos - pos]
        
        return augmented


class RandomFlip(TimeAugmentation):
    """
    随机翻转
    反转信号极性
    """
    
    def __init__(self, prob=0.2):
        super().__init__(prob)
    
    def apply(self, signal):
        """
        Args:
            signal: (seq_len,) torch tensor
        Returns:
            flipped: 翻转后的信号
        """
        return -signal


class Compose:
    """
    组合多个增强操作
    """
    
    def __init__(self, transforms):
        """
        Args:
            transforms: 增强操作列表
        """
        self.transforms = transforms
    
    def __call__(self, signal):
        """
        Args:
            signal: (seq_len,) torch tensor
        Returns:
            augmented: 增强后的信号
        """
        for t in self.transforms:
            signal = t(signal)
        return signal


def get_weak_augmentation():
    """
    弱增强策略
    用于有监督训练前期
    """
    return Compose([
        GaussianNoise(sigma_range=(0.01, 0.02), prob=0.3),
        AmplitudeScale(scale_range=(0.9, 1.1), prob=0.3),
    ])


def get_medium_augmentation():
    """
    中等增强策略
    用于有监督训练中期
    """
    return Compose([
        GaussianNoise(sigma_range=(0.01, 0.03), prob=0.5),
        TimeShift(shift_range=(-30, 30), prob=0.4),
        AmplitudeScale(scale_range=(0.8, 1.2), prob=0.5),
        RandomMasking(mask_ratio=0.05, prob=0.3),
    ])


def get_strong_augmentation():
    """
    强增强策略
    用于对比学习和有监督训练后期
    """
    return Compose([
        GaussianNoise(sigma_range=(0.03, 0.05), prob=0.6),
        TimeShift(shift_range=(-50, 50), prob=0.5),
        AmplitudeScale(scale_range=(0.7, 1.3), prob=0.6),
        TimeWarping(warp_range=(0.9, 1.1), prob=0.4),
        RandomMasking(mask_ratio=0.1, prob=0.4),
        AddImpulse(num_impulses=(1, 2), prob=0.3),
    ])


if __name__ == "__main__":
    """测试代码"""
    import matplotlib.pyplot as plt
    
    # 创建测试信号
    t = torch.linspace(0, 1, 512)
    signal = torch.sin(2 * np.pi * 100 * t) + 0.5 * torch.sin(2 * np.pi * 300 * t)
    
    print("测试时域数据增强")
    print("=" * 70)
    
    # 测试各种增强
    augmentations = {
        'GaussianNoise': GaussianNoise(prob=1.0),
        'TimeShift': TimeShift(prob=1.0),
        'AmplitudeScale': AmplitudeScale(prob=1.0),
        'TimeWarping': TimeWarping(prob=1.0),
        'RandomMasking': RandomMasking(prob=1.0),
        'AddImpulse': AddImpulse(prob=1.0),
    }
    
    for name, aug in augmentations.items():
        augmented = aug(signal)
        print(f"{name:20s}: 输入 {signal.shape} → 输出 {augmented.shape}")
    
    # 测试组合增强
    print("\n测试组合增强:")
    print("  Weak:   ", end="")
    weak_aug = get_weak_augmentation()
    aug_signal = weak_aug(signal)
    print(f"{signal.shape} → {aug_signal.shape}")
    
    print("  Medium: ", end="")
    medium_aug = get_medium_augmentation()
    aug_signal = medium_aug(signal)
    print(f"{signal.shape} → {aug_signal.shape}")
    
    print("  Strong: ", end="")
    strong_aug = get_strong_augmentation()
    aug_signal = strong_aug(signal)
    print(f"{signal.shape} → {aug_signal.shape}")
    
    print("\n✓ 时域数据增强测试完成!")
