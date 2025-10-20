"""
数据增强管道 (Augmentation Pipeline)
组合时域和频域增强，提供不同强度的增强策略
"""
import torch
import numpy as np
from .time_domain_aug import (
    GaussianNoise, TimeShift, AmplitudeScale, 
    TimeWarping, RandomMasking, AddImpulse,
    Compose as TimeCompose
)
from .frequency_aug import (
    FrequencyMasking, MagnitudeMasking, RandomFiltering,
    PhaseShift, FreqCompose
)


class AugmentationPipeline:
    """
    完整的数据增强管道
    根据训练阶段动态调整增强强度
    """
    
    def __init__(self, mode='train', intensity='medium', use_frequency_aug=True):
        """
        Args:
            mode: 'train' 或 'val'，验证集不增强
            intensity: 'weak', 'medium', 'strong'
            use_frequency_aug: 是否使用频域增强
        """
        self.mode = mode
        self.intensity = intensity
        self.use_frequency_aug = use_frequency_aug
        
        # 如果是验证模式，不使用增强
        if mode != 'train':
            self.time_aug = None
            self.freq_aug = None
            return
        
        # 根据强度配置时域增强
        self.time_aug = self._get_time_augmentation(intensity)
        
        # 频域增强（可选）
        if use_frequency_aug:
            self.freq_aug = self._get_frequency_augmentation(intensity)
        else:
            self.freq_aug = None
    
    def _get_time_augmentation(self, intensity):
        """获取时域增强"""
        if intensity == 'weak':
            return TimeCompose([
                GaussianNoise(sigma_range=(0.01, 0.02), prob=0.3),
                AmplitudeScale(scale_range=(0.9, 1.1), prob=0.3),
            ])
        
        elif intensity == 'medium':
            return TimeCompose([
                GaussianNoise(sigma_range=(0.01, 0.03), prob=0.5),
                TimeShift(shift_range=(-30, 30), prob=0.4),
                AmplitudeScale(scale_range=(0.8, 1.2), prob=0.5),
                RandomMasking(mask_ratio=0.05, prob=0.3),
            ])
        
        elif intensity == 'strong':
            return TimeCompose([
                GaussianNoise(sigma_range=(0.03, 0.05), prob=0.6),
                TimeShift(shift_range=(-50, 50), prob=0.5),
                AmplitudeScale(scale_range=(0.7, 1.3), prob=0.6),
                TimeWarping(warp_range=(0.9, 1.1), prob=0.4),
                RandomMasking(mask_ratio=0.1, prob=0.4),
                AddImpulse(num_impulses=(1, 2), prob=0.3),
            ])
        
        else:
            raise ValueError(f"Unknown intensity: {intensity}")
    
    def _get_frequency_augmentation(self, intensity):
        """获取频域增强"""
        if intensity == 'weak':
            return FreqCompose([
                FrequencyMasking(mask_param=(5, 10), num_masks=(1, 1), prob=0.2),
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
                FrequencyMasking(mask_param=(5, 20), num_masks=(1, 2), prob=0.4),
                MagnitudeMasking(attenuation=(0.3, 0.7), prob=0.3),
                RandomFiltering(filter_type='bandpass', prob=0.35),
                PhaseShift(phase_range=(-np.pi/8, np.pi/8), prob=0.25),
            ])
        
        else:
            raise ValueError(f"Unknown intensity: {intensity}")
    
    def __call__(self, signal):
        """
        应用增强
        
        Args:
            signal: (seq_len,) torch tensor
        Returns:
            augmented: 增强后的信号
        """
        # 验证模式不增强
        if self.mode != 'train':
            return signal
        
        # 应用时域增强
        if self.time_aug is not None:
            signal = self.time_aug(signal)
        
        # 应用频域增强（以一定概率）
        if self.freq_aug is not None and np.random.rand() < 0.3:
            signal = self.freq_aug(signal)
        
        return signal


class ContrastiveAugmentation:
    """
    对比学习专用的增强策略
    为每个样本生成两个不同的增强版本
    """
    
    def __init__(self, strong_aug_prob=0.5):
        """
        Args:
            strong_aug_prob: 使用强增强的概率
        """
        self.strong_aug_prob = strong_aug_prob
        
        # 强增强
        self.strong_aug = TimeCompose([
            GaussianNoise(sigma_range=(0.03, 0.05), prob=0.8),
            TimeShift(shift_range=(-50, 50), prob=0.6),
            AmplitudeScale(scale_range=(0.7, 1.3), prob=0.7),
            TimeWarping(warp_range=(0.85, 1.15), prob=0.5),
            RandomMasking(mask_ratio=0.15, prob=0.5),
            AddImpulse(num_impulses=(1, 3), prob=0.4),
        ])
        
        # 弱增强
        self.weak_aug = TimeCompose([
            GaussianNoise(sigma_range=(0.01, 0.02), prob=0.5),
            TimeShift(shift_range=(-20, 20), prob=0.3),
            AmplitudeScale(scale_range=(0.9, 1.1), prob=0.4),
        ])
    
    def __call__(self, signal):
        """
        生成两个增强版本
        
        Args:
            signal: (seq_len,) 原始信号
        Returns:
            aug1, aug2: 两个不同的增强版本
        """
        # 第一个增强：强增强或弱增强
        if np.random.rand() < self.strong_aug_prob:
            aug1 = self.strong_aug(signal.clone())
        else:
            aug1 = self.weak_aug(signal.clone())
        
        # 第二个增强：总是强增强
        aug2 = self.strong_aug(signal.clone())
        
        return aug1, aug2


class ProgressiveAugmentation:
    """
    渐进式增强策略
    随训练进行逐渐增加增强强度
    """
    
    def __init__(self, max_epochs):
        """
        Args:
            max_epochs: 总训练轮数
        """
        self.max_epochs = max_epochs
        self.current_epoch = 0
    
    def set_epoch(self, epoch):
        """设置当前epoch"""
        self.current_epoch = epoch
    
    def get_intensity(self):
        """根据当前epoch返回增强强度"""
        if self.current_epoch < self.max_epochs * 0.3:
            return 'weak'
        elif self.current_epoch < self.max_epochs * 0.7:
            return 'medium'
        else:
            return 'strong'
    
    def get_pipeline(self, mode='train'):
        """获取当前epoch的增强管道"""
        intensity = self.get_intensity()
        return AugmentationPipeline(mode=mode, intensity=intensity)


def get_augmentation_pipeline(stage='supervised', epoch=0, max_epochs=100, mode='train'):
    """
    获取增强管道的工厂函数
    
    Args:
        stage: 'pretrain', 'supervised'
        epoch: 当前epoch
        max_epochs: 总epoch数
        mode: 'train' 或 'val'
    
    Returns:
        增强管道实例
    """
    if mode != 'train':
        # 验证集不增强
        return AugmentationPipeline(mode='val')
    
    if stage == 'pretrain':
        # 对比学习预训练：使用对比学习增强
        return ContrastiveAugmentation()
    
    elif stage == 'supervised':
        # 有监督训练：使用渐进式增强
        progressive = ProgressiveAugmentation(max_epochs)
        progressive.set_epoch(epoch)
        return progressive.get_pipeline(mode='train')
    
    else:
        raise ValueError(f"Unknown stage: {stage}")


if __name__ == "__main__":
    """测试代码"""
    
    print("测试数据增强管道")
    print("=" * 70)
    
    # 创建测试信号
    signal = torch.randn(512)
    
    # 测试不同强度的增强
    print("\n1. 测试不同强度的增强管道")
    for intensity in ['weak', 'medium', 'strong']:
        pipeline = AugmentationPipeline(mode='train', intensity=intensity)
        augmented = pipeline(signal)
        print(f"   {intensity.capitalize():8s}: {signal.shape} → {augmented.shape}")
    
    # 测试对比学习增强
    print("\n2. 测试对比学习增强")
    contrastive_aug = ContrastiveAugmentation()
    aug1, aug2 = contrastive_aug(signal)
    print(f"   原始: {signal.shape}")
    print(f"   增强1: {aug1.shape}")
    print(f"   增强2: {aug2.shape}")
    
    # 测试渐进式增强
    print("\n3. 测试渐进式增强")
    progressive = ProgressiveAugmentation(max_epochs=100)
    for epoch in [10, 40, 80]:
        progressive.set_epoch(epoch)
        intensity = progressive.get_intensity()
        pipeline = progressive.get_pipeline(mode='train')
        print(f"   Epoch {epoch:3d}: intensity={intensity}")
    
    # 测试工厂函数
    print("\n4. 测试工厂函数")
    for stage in ['pretrain', 'supervised']:
        pipeline = get_augmentation_pipeline(stage=stage, epoch=50, max_epochs=100)
        print(f"   Stage={stage:10s}: {type(pipeline).__name__}")
    
    print("\n✓ 数据增强管道测试完成!")
