"""
模型模块
包含backbone, modules, fusion, heads等子模块
"""
from .backbone import (
    TemporalBranch,
    FrequencyBranch,
    TimeFrequencyBranch,
    create_temporal_branch,
    create_frequency_branch,
    create_timefreq_branch
)

from .modules import (
    CBAM,
    SEBlock,
    TemporalSelfAttention,
    EMA,
    MSFEB,
    InceptionBlock1D,
    ResidualBlock1D,
    ResidualBlock2D,
    GlobalPooling1D,
    GlobalPooling2D,
    MultiHeadPooling1D
)

__all__ = [
    # Backbone branches
    'TemporalBranch',
    'FrequencyBranch',
    'TimeFrequencyBranch',
    'create_temporal_branch',
    'create_frequency_branch',
    'create_timefreq_branch',
    
    # Modules
    'CBAM',
    'SEBlock',
    'TemporalSelfAttention',
    'EMA',
    'MSFEB',
    'InceptionBlock1D',
    'ResidualBlock1D',
    'ResidualBlock2D',
    'GlobalPooling1D',
    'GlobalPooling2D',
    'MultiHeadPooling1D',
]
