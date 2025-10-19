"""
可复用的网络模块
"""
from .attention import (
    CBAM,
    SEBlock,
    TemporalSelfAttention,
    EMA,
    ChannelAttention,
    SpatialAttention,
    ChannelAttention2D,
    SpatialAttention2D
)

from .msfeb import MSFEB, InceptionBlock1D

from .residual import (
    ResidualBlock1D,
    ResidualBlock2D,
    BottleneckResidualBlock1D
)

from .pooling import (
    GlobalPooling1D,
    GlobalPooling2D,
    AdaptivePooling2D,
    GeM,
    MultiHeadPooling1D
)

__all__ = [
    # Attention modules
    'CBAM',
    'SEBlock',
    'TemporalSelfAttention',
    'EMA',
    'ChannelAttention',
    'SpatialAttention',
    'ChannelAttention2D',
    'SpatialAttention2D',

    # Multi-scale feature extraction
    'MSFEB',
    'InceptionBlock1D',

    # Residual blocks
    'ResidualBlock1D',
    'ResidualBlock2D',
    'BottleneckResidualBlock1D',

    # Pooling layers
    'GlobalPooling1D',
    'GlobalPooling2D',
    'AdaptivePooling2D',
    'GeM',
    'MultiHeadPooling1D',
]
