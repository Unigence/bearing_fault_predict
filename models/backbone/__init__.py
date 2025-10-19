"""
三分支Backbone模块 - 重构版本
包含时域、频域和时频分支（纯特征提取，无预处理）
"""
from .temporal_branch import (
    TemporalBranch,
    create_temporal_branch
)

from .frequency_branch import (
    FrequencyBranch,
    LightweightFrequencyBranch,
    create_frequency_branch
)

from .timefreq_branch import (
    TimeFrequencyBranch,
    create_timefreq_branch
)

__all__ = [
    # Temporal branch
    'TemporalBranch',
    'create_temporal_branch',

    # Frequency branch
    'FrequencyBranch',
    'LightweightFrequencyBranch',
    'create_frequency_branch',

    # Time-frequency branch
    'TimeFrequencyBranch',
    'create_timefreq_branch',
]

