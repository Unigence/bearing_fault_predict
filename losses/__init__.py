"""
损失函数模块
包含Focal Loss、ArcFace Loss、对比学习Loss等
"""

# Focal Loss
from .focal_loss import (
    FocalLoss,
    AdaptiveFocalLoss,
    compute_class_weights
)

# Combined Loss (Focal + ArcFace)
from .combined_loss import (
    CombinedLoss,
    ProgressiveCombinedLoss
)

# Contrastive Learning Loss
from .contrastive_loss import (
    NTXentLoss,
    SupConLoss,
    HardNegativeNTXentLoss,
    MomentumContrastLoss
)

__all__ = [
    # Focal Loss
    'FocalLoss',
    'AdaptiveFocalLoss',
    'compute_class_weights',
    
    # Combined Loss
    'CombinedLoss',
    'ProgressiveCombinedLoss',
    
    # Contrastive Loss
    'NTXentLoss',
    'SupConLoss',
    'HardNegativeNTXentLoss',
    'MomentumContrastLoss',
]
