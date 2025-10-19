"""
双头分类器
训练时同时使用Softmax和ArcFace，推理时只使用Softmax
联合训练提升特征判别性
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .softmax_head import SoftmaxHead
from .arcface_head import ArcFaceHead


class DualHead(nn.Module):
    """
    双头分类器
    - Head 1: Softmax - 用于训练和推理
    - Head 2: ArcFace - 仅用于训练，增强度量学习
    """
    
    def __init__(self,
                 in_features=128,
                 num_classes=6,
                 hidden_dim=256,
                 s=30.0,
                 m=0.50,
                 dropout1=0.4,
                 dropout2=0.3,
                 use_l2_norm=True):
        """
        Args:
            in_features: 输入特征维度
            num_classes: 类别数
            hidden_dim: Softmax head的隐藏层维度
            s: ArcFace的scale参数
            m: ArcFace的margin参数
            dropout1: Softmax head的第一层Dropout
            dropout2: Softmax head的第二层Dropout
            use_l2_norm: 是否使用L2归一化
        """
        super(DualHead, self).__init__()
        
        self.in_features = in_features
        self.num_classes = num_classes
        
        # Head 1: Softmax分类器
        self.softmax_head = SoftmaxHead(
            in_features=in_features,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout1=dropout1,
            dropout2=dropout2,
            use_l2_norm=use_l2_norm
        )
        
        # Head 2: ArcFace分类器
        self.arcface_head = ArcFaceHead(
            in_features=in_features,
            num_classes=num_classes,
            s=s,
            m=m
        )
        
    def forward(self, x, labels=None, mode='train'):
        """
        前向传播
        
        Args:
            x: 输入特征 (B, in_features)
            labels: 标签 (B,) - 训练时必须提供
            mode: 'train' 或 'eval'
        
        Returns:
            如果mode='train':
                (softmax_logits, arcface_logits, normalized_features)
            如果mode='eval':
                softmax_logits
        """
        if mode == 'train':
            # 训练模式：返回两个头的输出
            softmax_logits, normalized_feat = self.softmax_head(x, return_features=True)
            arcface_logits = self.arcface_head(normalized_feat, labels)
            
            return softmax_logits, arcface_logits, normalized_feat
        
        else:
            # 推理模式：只使用Softmax头
            softmax_logits = self.softmax_head(x)
            return softmax_logits


class DualHeadWithSharedBackbone(nn.Module):
    """
    共享Backbone的双头分类器
    两个头共享特征增强层，减少参数量
    """
    
    def __init__(self,
                 in_features=128,
                 num_classes=6,
                 hidden_dim=256,
                 s=30.0,
                 m=0.50,
                 dropout=0.4):
        """
        Args:
            in_features: 输入特征维度
            num_classes: 类别数
            hidden_dim: 隐藏层维度
            s: ArcFace的scale参数
            m: ArcFace的margin参数
            dropout: Dropout率
        """
        super(DualHeadWithSharedBackbone, self).__init__()
        
        self.in_features = in_features
        self.num_classes = num_classes
        
        # 共享的特征增强层
        self.shared_backbone = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, in_features),
            nn.BatchNorm1d(in_features),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.75)
        )
        
        # Head 1: Softmax
        self.softmax_classifier = nn.Linear(in_features, num_classes)
        
        # Head 2: ArcFace
        self.arcface_head = ArcFaceHead(
            in_features=in_features,
            num_classes=num_classes,
            s=s,
            m=m
        )
        
    def forward(self, x, labels=None, mode='train'):
        """前向传播"""
        # 共享特征增强
        enhanced_feat = self.shared_backbone(x)
        
        # L2归一化
        normalized_feat = F.normalize(enhanced_feat, p=2, dim=1)
        
        if mode == 'train':
            # 两个头都使用
            softmax_logits = self.softmax_classifier(normalized_feat)
            arcface_logits = self.arcface_head(normalized_feat, labels)
            
            return softmax_logits, arcface_logits, normalized_feat
        
        else:
            # 只使用Softmax
            softmax_logits = self.softmax_classifier(normalized_feat)
            return softmax_logits


class EnsembleDualHead(nn.Module):
    """
    集成双头分类器
    推理时综合两个头的输出
    """
    
    def __init__(self,
                 in_features=128,
                 num_classes=6,
                 hidden_dim=256,
                 s=30.0,
                 m=0.50,
                 ensemble_weight=0.5):
        """
        Args:
            in_features: 输入特征维度
            num_classes: 类别数
            hidden_dim: 隐藏层维度
            s: ArcFace的scale参数
            m: ArcFace的margin参数
            ensemble_weight: Softmax的权重，ArcFace的权重为1-ensemble_weight
        """
        super(EnsembleDualHead, self).__init__()
        
        self.ensemble_weight = ensemble_weight
        
        self.softmax_head = SoftmaxHead(
            in_features=in_features,
            num_classes=num_classes,
            hidden_dim=hidden_dim
        )
        
        self.arcface_head = ArcFaceHead(
            in_features=in_features,
            num_classes=num_classes,
            s=s,
            m=m
        )
        
    def forward(self, x, labels=None, mode='train'):
        """前向传播"""
        if mode == 'train':
            softmax_logits, normalized_feat = self.softmax_head(x, return_features=True)
            arcface_logits = self.arcface_head(normalized_feat, labels)
            
            return softmax_logits, arcface_logits, normalized_feat
        
        else:
            # 推理时集成两个头的输出
            softmax_logits, normalized_feat = self.softmax_head(x, return_features=True)
            arcface_logits = self.arcface_head(normalized_feat)
            
            # 加权平均
            ensemble_logits = (self.ensemble_weight * softmax_logits + 
                             (1 - self.ensemble_weight) * arcface_logits)
            
            return ensemble_logits


# 单元测试
if __name__ == '__main__':
    print("Testing Dual Head...")
    
    # 创建模块
    dual_head = DualHead(
        in_features=128,
        num_classes=6,
        s=30.0,
        m=0.50
    )
    
    # 创建测试数据
    batch_size = 8
    x = torch.randn(batch_size, 128)
    labels = torch.randint(0, 6, (batch_size,))
    
    # 测试训练模式
    print("\n1. Testing training mode...")
    dual_head.train()
    softmax_logits, arcface_logits, features = dual_head(x, labels, mode='train')
    
    print(f"✓ Softmax logits shape: {softmax_logits.shape}")  # (8, 6)
    print(f"✓ ArcFace logits shape: {arcface_logits.shape}")  # (8, 6)
    print(f"✓ Features shape: {features.shape}")  # (8, 128)
    print(f"✓ Features L2 norm: {torch.norm(features, p=2, dim=1)[:3]}")
    
    # 测试推理模式
    print("\n2. Testing inference mode...")
    dual_head.eval()
    with torch.no_grad():
        logits = dual_head(x, mode='eval')
    
    print(f"✓ Inference logits shape: {logits.shape}")
    probs = F.softmax(logits, dim=1)
    print(f"✓ Probabilities sum: {probs.sum(dim=1)[:3]}")
    
    # 测试共享Backbone版本
    print("\n3. Testing DualHeadWithSharedBackbone...")
    shared_dual_head = DualHeadWithSharedBackbone(
        in_features=128,
        num_classes=6
    )
    
    shared_dual_head.train()
    softmax_logits, arcface_logits, features = shared_dual_head(x, labels, mode='train')
    print(f"✓ Shared version - Softmax: {softmax_logits.shape}, ArcFace: {arcface_logits.shape}")
    
    # 测试集成版本
    print("\n4. Testing EnsembleDualHead...")
    ensemble_head = EnsembleDualHead(
        in_features=128,
        num_classes=6,
        ensemble_weight=0.7
    )
    
    ensemble_head.train()
    softmax_logits, arcface_logits, features = ensemble_head(x, labels, mode='train')
    
    ensemble_head.eval()
    with torch.no_grad():
        ensemble_logits = ensemble_head(x, mode='eval')
    
    print(f"✓ Ensemble logits shape: {ensemble_logits.shape}")
    
    # 参数统计
    print("\n5. Model statistics...")
    
    total_params_dual = sum(p.numel() for p in dual_head.parameters())
    print(f"✓ DualHead parameters: {total_params_dual:,}")
    
    total_params_shared = sum(p.numel() for p in shared_dual_head.parameters())
    print(f"✓ DualHeadWithSharedBackbone parameters: {total_params_shared:,}")
    print(f"  (Saved {total_params_dual - total_params_shared:,} parameters)")
    
    total_params_ensemble = sum(p.numel() for p in ensemble_head.parameters())
    print(f"✓ EnsembleDualHead parameters: {total_params_ensemble:,}")
    
    print("\n✅ All tests passed!")
