"""
特征融合模块
包含三级渐进式融合: 模态重要性学习 + 跨模态交互 + 多粒度整合
"""

from .adaptive_fusion import ModalImportanceLearning, AdaptiveFusionV1
from .cross_modal_attention import CrossModalAttention, CrossModalInteraction
from .hierarchical_fusion import HierarchicalFusion, HierarchicalFusionV2

__all__ = [
    'ModalImportanceLearning',
    'AdaptiveFusionV1',
    'CrossModalAttention',
    'CrossModalInteraction',
    'HierarchicalFusion',
    'HierarchicalFusionV2'
]
