"""
Focal Loss - 解决类别不平衡问题
参考论文: Focal Loss for Dense Object Detection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    特点:
    1. 降低易分样本的权重
    2. 聚焦于难分样本
    3. 通过α和γ平衡正负样本和难易样本
    """
    
    def __init__(self, 
                 alpha=None,
                 gamma=2.0,
                 reduction='mean',
                 label_smoothing=0.0):
        """
        Args:
            alpha: 类别权重 (Tensor or None)
                   - None: 所有类别权重相同
                   - Tensor: shape (num_classes,), 每个类别的权重
            gamma: 聚焦参数
                   - 0: 退化为标准交叉熵
                   - >0: 降低易分样本权重, 通常取2.0
            reduction: 'none' | 'mean' | 'sum'
            label_smoothing: 标签平滑参数 [0, 1]
        """
        super(FocalLoss, self).__init__()
        
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        
        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                self.alpha = torch.tensor(alpha)
            elif not isinstance(alpha, torch.Tensor):
                raise TypeError("alpha should be list, tuple or Tensor")
    
    def forward(self, inputs, targets):
        """
        前向传播
        
        Args:
            inputs: 模型输出logits (B, C)
            targets: 真实标签 (B,)
        
        Returns:
            loss: Focal Loss值
        """
        batch_size, num_classes = inputs.shape
        
        # 计算交叉熵
        ce_loss = F.cross_entropy(
            inputs, 
            targets, 
            reduction='none',
            label_smoothing=self.label_smoothing
        )
        
        # 计算pt (预测概率)
        p = F.softmax(inputs, dim=1)
        p_t = p.gather(1, targets.view(-1, 1)).squeeze(1)  # (B,)
        
        # 计算focal term: (1 - p_t)^γ
        focal_weight = (1 - p_t) ** self.gamma
        
        # 计算focal loss
        focal_loss = focal_weight * ce_loss
        
        # 应用类别权重α
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss
        
        # Reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class AdaptiveFocalLoss(nn.Module):
    """
    自适应Focal Loss
    gamma参数随训练动态调整
    """
    
    def __init__(self,
                 alpha=None,
                 gamma_init=2.0,
                 gamma_min=1.0,
                 gamma_schedule='linear',
                 label_smoothing=0.0):
        """
        Args:
            alpha: 类别权重
            gamma_init: 初始gamma值
            gamma_min: 最小gamma值
            gamma_schedule: 'linear' | 'cosine' | 'step'
            label_smoothing: 标签平滑
        """
        super(AdaptiveFocalLoss, self).__init__()
        
        self.alpha = alpha
        self.gamma_init = gamma_init
        self.gamma_min = gamma_min
        self.gamma_schedule = gamma_schedule
        self.label_smoothing = label_smoothing
        
        # 当前gamma
        self.register_buffer('current_gamma', torch.tensor(gamma_init))
        
        if alpha is not None and not isinstance(alpha, torch.Tensor):
            self.alpha = torch.tensor(alpha)
    
    def update_gamma(self, progress):
        """
        更新gamma参数
        
        Args:
            progress: 训练进度 [0, 1]
        """
        import math
        
        if self.gamma_schedule == 'linear':
            gamma = self.gamma_init - (self.gamma_init - self.gamma_min) * progress
        elif self.gamma_schedule == 'cosine':
            gamma = self.gamma_min + (self.gamma_init - self.gamma_min) * \
                    (1 + math.cos(math.pi * progress)) / 2
        elif self.gamma_schedule == 'step':
            if progress < 0.5:
                gamma = self.gamma_init
            elif progress < 0.8:
                gamma = (self.gamma_init + self.gamma_min) / 2
            else:
                gamma = self.gamma_min
        else:
            gamma = self.gamma_init
        
        self.current_gamma = torch.tensor(gamma)
    
    def forward(self, inputs, targets):
        """前向传播"""
        batch_size, num_classes = inputs.shape
        
        # 计算交叉熵
        ce_loss = F.cross_entropy(
            inputs, 
            targets, 
            reduction='none',
            label_smoothing=self.label_smoothing
        )
        
        # 计算pt
        p = F.softmax(inputs, dim=1)
        p_t = p.gather(1, targets.view(-1, 1)).squeeze(1)
        
        # 使用当前gamma
        gamma = self.current_gamma.item()
        focal_weight = (1 - p_t) ** gamma
        
        # 计算focal loss
        focal_loss = focal_weight * ce_loss
        
        # 应用类别权重
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss
        
        return focal_loss.mean()


def compute_class_weights(class_counts, method='effective_num', beta=0.9999):
    """
    计算类别权重
    
    Args:
        class_counts: 各类别样本数 (list or array)
        method: 'inverse' | 'effective_num' | 'balanced'
        beta: Effective Number的β参数
    
    Returns:
        weights: 类别权重 (Tensor)
    """
    class_counts = torch.tensor(class_counts, dtype=torch.float32)
    num_classes = len(class_counts)
    total = class_counts.sum()
    
    if method == 'inverse':
        # 逆频率权重
        weights = total / (num_classes * class_counts)
    
    elif method == 'effective_num':
        # Effective Number方法
        # CB_weight = (1-β) / (1-β^n)
        effective_num = 1.0 - torch.pow(beta, class_counts)
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * num_classes
    
    elif method == 'balanced':
        # 平衡权重
        weights = total / (2 * class_counts)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # 归一化到合理范围[0.5, 2.0]
    weights = torch.clamp(weights, 0.5, 2.0)
    
    return weights


# 单元测试
if __name__ == '__main__':
    print("=" * 70)
    print("测试Focal Loss")
    print("=" * 70)
    
    # 创建测试数据
    batch_size = 16
    num_classes = 6
    inputs = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # 1. 测试标准Focal Loss
    print("\n1. 测试标准Focal Loss...")
    focal_loss = FocalLoss(gamma=2.0)
    loss = focal_loss(inputs, targets)
    print(f"✓ Focal Loss: {loss.item():.4f}")
    
    # 2. 测试带类别权重的Focal Loss
    print("\n2. 测试带类别权重...")
    class_counts = [160, 120, 180, 150, 100, 150]  # 模拟不平衡数据
    class_weights = compute_class_weights(class_counts, method='effective_num')
    print(f"✓ 类别权重: {class_weights}")
    
    focal_loss_weighted = FocalLoss(alpha=class_weights, gamma=2.0)
    loss_weighted = focal_loss_weighted(inputs, targets)
    print(f"✓ Weighted Focal Loss: {loss_weighted.item():.4f}")
    
    # 3. 测试label smoothing
    print("\n3. 测试Label Smoothing...")
    focal_loss_smooth = FocalLoss(gamma=2.0, label_smoothing=0.1)
    loss_smooth = focal_loss_smooth(inputs, targets)
    print(f"✓ Focal Loss (smoothing=0.1): {loss_smooth.item():.4f}")
    
    # 4. 测试自适应Focal Loss
    print("\n4. 测试自适应Focal Loss...")
    adaptive_focal = AdaptiveFocalLoss(
        alpha=class_weights,
        gamma_init=2.0,
        gamma_min=1.0,
        gamma_schedule='linear'
    )
    
    for progress in [0.0, 0.5, 1.0]:
        adaptive_focal.update_gamma(progress)
        loss = adaptive_focal(inputs, targets)
        print(f"✓ Progress={progress:.1f}, Gamma={adaptive_focal.current_gamma.item():.2f}, Loss={loss.item():.4f}")
    
    # 5. 对比不同gamma值的影响
    print("\n5. 对比不同gamma值...")
    for gamma in [0.0, 1.0, 2.0, 3.0]:
        focal = FocalLoss(gamma=gamma)
        loss = focal(inputs, targets)
        print(f"✓ Gamma={gamma:.1f}, Loss={loss.item():.4f}")
    
    # 6. 测试梯度
    print("\n6. 测试梯度...")
    inputs_grad = inputs.clone().requires_grad_(True)
    focal_loss = FocalLoss(gamma=2.0)
    loss = focal_loss(inputs_grad, targets)
    loss.backward()
    print(f"✓ 梯度范围: [{inputs_grad.grad.min():.4f}, {inputs_grad.grad.max():.4f}]")
    
    print("\n" + "=" * 70)
    print("✅ 所有测试通过!")
    print("=" * 70)
