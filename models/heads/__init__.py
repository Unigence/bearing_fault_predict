"""
分类头模块
包含Softmax、ArcFace和双头设计
"""

from .softmax_head import SoftmaxHead, SimpleSoftmaxHead
from .arcface_head import ArcFaceHead, AdaptiveArcFaceHead
from .dual_head import DualHead, DualHeadWithSharedBackbone, EnsembleDualHead

__all__ = [
    'SoftmaxHead',
    'SimpleSoftmaxHead',
    'ArcFaceHead',
    'AdaptiveArcFaceHead',
    'DualHead',
    'DualHeadWithSharedBackbone',
    'EnsembleDualHead'
]
