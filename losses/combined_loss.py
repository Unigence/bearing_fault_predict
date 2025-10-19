"""
组合损失函数
联合训练Focal Loss + ArcFace Loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .focal_loss import FocalLoss


class CombinedLoss(nn.Module):
    """
    组合损失函数
    L_total = λ1 * L_focal + λ2 * L_arcface
    
    用于双头分类器的联合训练
    """
    
    def __init__(self,
                 num_classes=6,
                 focal_alpha=None,
                 focal_gamma=2.0,
                 focal_weight=1.0,
                 arcface_weight=0.5,
                 label_smoothing=0.1):
        """
        Args:
            num_classes: 类别数
            focal_alpha: Focal Loss的类别权重
            focal_gamma: Focal Loss的gamma参数
            focal_weight: Focal Loss的权重λ1
            arcface_weight: ArcFace Loss的权重λ2
            label_smoothing: 标签平滑参数
        """
        super(CombinedLoss, self).__init__()
        
        self.num_classes = num_classes
        self.focal_weight = focal_weight
        self.arcface_weight = arcface_weight
        
        # Focal Loss
        self.focal_loss = FocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            label_smoothing=label_smoothing
        )
        
        # ArcFace Loss (使用CrossEntropy)
        self.arcface_loss = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing
        )
    
    def forward(self, softmax_logits, arcface_logits, targets):
        """
        前向传播
        
        Args:
            softmax_logits: Softmax头的输出 (B, num_classes)
            arcface_logits: ArcFace头的输出 (B, num_classes)
            targets: 真实标签 (B,)
        
        Returns:
            total_loss: 总损失
            loss_dict: 各项损失的字典
        """
        # 计算Focal Loss
        loss_focal = self.focal_loss(softmax_logits, targets)
        
        # 计算ArcFace Loss
        loss_arcface = self.arcface_loss(arcface_logits, targets)
        
        # 总损失
        total_loss = (self.focal_weight * loss_focal + 
                     self.arcface_weight * loss_arcface)
        
        # 返回详细信息
        loss_dict = {
            'total': total_loss.item(),
            'focal': loss_focal.item(),
            'arcface': loss_arcface.item()
        }
        
        return total_loss, loss_dict
    
    def update_weights(self, epoch, total_epochs):
        """
        动态调整损失权重
        
        策略:
        - 前期(0-30%): ArcFace权重较小,让模型先学好基础分类
        - 中期(30-60%): 逐渐增大ArcFace权重
        - 后期(60-100%): ArcFace权重达到最大,强化度量学习
        
        Args:
            epoch: 当前epoch
            total_epochs: 总epoch数
        """
        progress = epoch / total_epochs
        
        if progress < 0.3:
            self.arcface_weight = 0.3
        elif progress < 0.6:
            self.arcface_weight = 0.5
        else:
            self.arcface_weight = 0.7


class ProgressiveCombinedLoss(nn.Module):
    """
    渐进式组合损失
    权重随训练动态调整
    """
    
    def __init__(self,
                 num_classes=6,
                 focal_alpha=None,
                 focal_gamma_init=2.0,
                 focal_gamma_min=1.0,
                 arcface_weight_init=0.3,
                 arcface_weight_max=0.7,
                 label_smoothing=0.1):
        """
        Args:
            num_classes: 类别数
            focal_alpha: Focal Loss类别权重
            focal_gamma_init: Focal Loss初始gamma
            focal_gamma_min: Focal Loss最小gamma
            arcface_weight_init: ArcFace初始权重
            arcface_weight_max: ArcFace最大权重
            label_smoothing: 标签平滑
        """
        super(ProgressiveCombinedLoss, self).__init__()
        
        self.num_classes = num_classes
        self.arcface_weight_init = arcface_weight_init
        self.arcface_weight_max = arcface_weight_max
        
        # Focal Loss (自适应gamma)
        from .focal_loss import AdaptiveFocalLoss
        self.focal_loss = AdaptiveFocalLoss(
            alpha=focal_alpha,
            gamma_init=focal_gamma_init,
            gamma_min=focal_gamma_min,
            gamma_schedule='linear',
            label_smoothing=label_smoothing
        )
        
        # ArcFace Loss
        self.arcface_loss = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing
        )
        
        # 当前ArcFace权重
        self.register_buffer('current_arcface_weight', 
                           torch.tensor(arcface_weight_init))
    
    def update_schedule(self, progress):
        """
        更新训练进度相关的参数
        
        Args:
            progress: 训练进度 [0, 1]
        """
        # 更新Focal Loss的gamma
        self.focal_loss.update_gamma(progress)
        
        # 更新ArcFace权重
        weight = self.arcface_weight_init + \
                (self.arcface_weight_max - self.arcface_weight_init) * progress
        self.current_arcface_weight = torch.tensor(weight)
    
    def forward(self, softmax_logits, arcface_logits, targets):
        """前向传播"""
        # 计算Focal Loss
        loss_focal = self.focal_loss(softmax_logits, targets)
        
        # 计算ArcFace Loss
        loss_arcface = self.arcface_loss(arcface_logits, targets)
        
        # 使用当前权重
        arcface_w = self.current_arcface_weight.item()
        total_loss = loss_focal + arcface_w * loss_arcface
        
        # 返回详细信息
        loss_dict = {
            'total': total_loss.item(),
            'focal': loss_focal.item(),
            'arcface': loss_arcface.item(),
            'arcface_weight': arcface_w,
            'focal_gamma': self.focal_loss.current_gamma.item()
        }
        
        return total_loss, loss_dict


# 单元测试
if __name__ == '__main__':
    print("=" * 70)
    print("测试组合损失函数")
    print("=" * 70)
    
    # 创建测试数据
    batch_size = 16
    num_classes = 6
    softmax_logits = torch.randn(batch_size, num_classes)
    arcface_logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # 1. 测试标准组合损失
    print("\n1. 测试标准CombinedLoss...")
    combined_loss = CombinedLoss(
        num_classes=6,
        focal_weight=1.0,
        arcface_weight=0.5
    )
    
    total_loss, loss_dict = combined_loss(softmax_logits, arcface_logits, targets)
    print(f"✓ 总损失: {loss_dict['total']:.4f}")
    print(f"✓ Focal Loss: {loss_dict['focal']:.4f}")
    print(f"✓ ArcFace Loss: {loss_dict['arcface']:.4f}")
    
    # 2. 测试权重更新
    print("\n2. 测试动态权重调整...")
    for epoch in [10, 40, 80]:
        combined_loss.update_weights(epoch, total_epochs=100)
        total_loss, loss_dict = combined_loss(softmax_logits, arcface_logits, targets)
        print(f"✓ Epoch {epoch:3d}: ArcFace权重={combined_loss.arcface_weight:.1f}, "
              f"总损失={loss_dict['total']:.4f}")
    
    # 3. 测试渐进式组合损失
    print("\n3. 测试ProgressiveCombinedLoss...")
    progressive_loss = ProgressiveCombinedLoss(
        num_classes=6,
        focal_gamma_init=2.0,
        focal_gamma_min=1.0,
        arcface_weight_init=0.3,
        arcface_weight_max=0.7
    )
    
    for progress in [0.0, 0.3, 0.6, 1.0]:
        progressive_loss.update_schedule(progress)
        total_loss, loss_dict = progressive_loss(softmax_logits, arcface_logits, targets)
        
        print(f"✓ Progress={progress:.1f}: "
              f"Gamma={loss_dict['focal_gamma']:.2f}, "
              f"ArcFace权重={loss_dict['arcface_weight']:.2f}, "
              f"总损失={loss_dict['total']:.4f}")
    
    # 4. 测试梯度
    print("\n4. 测试梯度传播...")
    softmax_grad = softmax_logits.clone().requires_grad_(True)
    arcface_grad = arcface_logits.clone().requires_grad_(True)
    
    combined_loss = CombinedLoss(num_classes=6)
    total_loss, _ = combined_loss(softmax_grad, arcface_grad, targets)
    total_loss.backward()
    
    print(f"✓ Softmax梯度: [{softmax_grad.grad.min():.4f}, {softmax_grad.grad.max():.4f}]")
    print(f"✓ ArcFace梯度: [{arcface_grad.grad.min():.4f}, {arcface_grad.grad.max():.4f}]")
    
    # 5. 测试带类别权重的组合损失
    print("\n5. 测试带类别权重...")
    from .focal_loss import compute_class_weights
    
    class_counts = [160, 120, 180, 150, 100, 150]
    class_weights = compute_class_weights(class_counts, method='effective_num')
    
    weighted_combined = CombinedLoss(
        num_classes=6,
        focal_alpha=class_weights,
        focal_weight=1.0,
        arcface_weight=0.5
    )
    
    total_loss, loss_dict = weighted_combined(softmax_logits, arcface_logits, targets)
    print(f"✓ Weighted总损失: {loss_dict['total']:.4f}")
    print(f"✓ Weighted Focal: {loss_dict['focal']:.4f}")
    print(f"✓ Weighted ArcFace: {loss_dict['arcface']:.4f}")
    
    print("\n" + "=" * 70)
    print("✅ 所有测试通过!")
    print("=" * 70)
