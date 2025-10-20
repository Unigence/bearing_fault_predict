"""
数据增强模块
包含时域、频域增强和Mixup策略
"""
from .time_domain_aug import (
    GaussianNoise,
    TimeShift,
    AmplitudeScale,
    TimeWarping,
    RandomMasking,
    AddImpulse,
    RandomFlip,
    get_weak_augmentation,
    get_medium_augmentation,
    get_strong_augmentation,
)

from .frequency_aug import (
    FrequencyMasking,
    MagnitudeMasking,
    RandomFiltering,
    PhaseShift,
    FrequencyShift,
    get_frequency_augmentation,
)

from .mixup import (
    TimeDomainMixup,
    FrequencyDomainMixup,
    MultiModalMixup,
    ManifoldMixup,
)

from .augmentation_pipeline import (
    AugmentationPipeline,
    ContrastiveAugmentation,
    ProgressiveAugmentation,
    get_augmentation_pipeline,
)

__all__ = [
    # 时域增强
    'GaussianNoise',
    'TimeShift',
    'AmplitudeScale',
    'TimeWarping',
    'RandomMasking',
    'AddImpulse',
    'RandomFlip',
    'get_weak_augmentation',
    'get_medium_augmentation',
    'get_strong_augmentation',
    
    # 频域增强
    'FrequencyMasking',
    'MagnitudeMasking',
    'RandomFiltering',
    'PhaseShift',
    'FrequencyShift',
    'get_frequency_augmentation',
    
    # Mixup
    TimeDomainMixup,
    FrequencyDomainMixup,
    MultiModalMixup,
    ManifoldMixup,
    
    # 增强管道
    'AugmentationPipeline',
    'ContrastiveAugmentation',
    'ProgressiveAugmentation',
    'get_augmentation_pipeline',
]
