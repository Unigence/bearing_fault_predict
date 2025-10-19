"""
ArcFace分类器
基于度量学习的分类头，增大类间距离，缩小类内距离
参考论文: ArcFace: Additive Angular Margin Loss for Deep Face Recognition
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ArcFaceHead(nn.Module):
    """
    ArcFace分类器
    在角度空间添加margin，增强类别判别性
    """
    
    def __init__(self,
                 in_features=128,
                 num_classes=6,
                 s=30.0,
                 m=0.50,
                 easy_margin=False):
        """
        Args:
            in_features: 输入特征维度
            num_classes: 类别数
            s: scale参数，控制logits的缩放
            m: margin参数，添加的角度余量（弧度）
            easy_margin: 是否使用easy margin（简化版）
        """
        super(ArcFaceHead, self).__init__()
        
        self.in_features = in_features
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.easy_margin = easy_margin
        
        # 权重矩阵 (相当于类中心)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        # 预计算cos(m)和sin(m)
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)  # threshold = cos(pi - m)
        self.mm = math.sin(math.pi - m) * m  # margin adjustment
        
    def forward(self, x, labels=None):
        """
        前向传播
        
        Args:
            x: 输入特征 (B, in_features) - 必须是L2归一化的
            labels: 标签 (B,) - 训练时必须提供
        
        Returns:
            logits: ArcFace logits (B, num_classes)
        """
        # L2归一化输入特征
        x = F.normalize(x, p=2, dim=1)
        
        # L2归一化权重（类中心）
        W = F.normalize(self.weight, p=2, dim=1)
        
        # 计算余弦相似度 (角度)
        cosine = F.linear(x, W)  # (B, num_classes)
        cosine = torch.clamp(cosine, -1.0, 1.0)  # 数值稳定性
        
        # 如果没有标签（推理阶段），直接返回cosine * s
        if labels is None:
            return cosine * self.s
        
        # ==================== 训练阶段：添加margin ====================
        
        # 计算sin(theta)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        
        # 计算cos(theta + m)
        phi = cosine * self.cos_m - sine * self.sin_m
        
        if self.easy_margin:
            # Easy margin: 只在cos(theta) > 0时添加margin
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            # Hard margin: 更严格的条件
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # One-hot编码
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        # 只对正确类别添加margin
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        
        # 缩放
        output = output * self.s
        
        return output


class AdaptiveArcFaceHead(nn.Module):
    """
    自适应ArcFace分类器
    margin参数会随着训练动态调整
    """
    
    def __init__(self,
                 in_features=128,
                 num_classes=6,
                 s=30.0,
                 m_init=0.30,
                 m_max=0.50,
                 m_schedule='linear'):
        """
        Args:
            in_features: 输入特征维度
            num_classes: 类别数
            s: scale参数
            m_init: 初始margin
            m_max: 最大margin
            m_schedule: margin调度策略 ('linear', 'cosine', 'step')
        """
        super(AdaptiveArcFaceHead, self).__init__()
        
        self.in_features = in_features
        self.num_classes = num_classes
        self.s = s
        self.m_init = m_init
        self.m_max = m_max
        self.m_schedule = m_schedule
        
        # 当前margin（会动态更新）
        self.register_buffer('current_m', torch.tensor(m_init))
        
        # 权重矩阵
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        
    def update_margin(self, progress):
        """
        更新margin
        
        Args:
            progress: 训练进度 [0, 1]
        """
        if self.m_schedule == 'linear':
            m = self.m_init + (self.m_max - self.m_init) * progress
        elif self.m_schedule == 'cosine':
            m = self.m_init + (self.m_max - self.m_init) * (1 - math.cos(progress * math.pi)) / 2
        elif self.m_schedule == 'step':
            if progress < 0.5:
                m = self.m_init
            elif progress < 0.8:
                m = (self.m_init + self.m_max) / 2
            else:
                m = self.m_max
        else:
            m = self.m_max
        
        self.current_m = torch.tensor(m)
        
    def forward(self, x, labels=None):
        """前向传播"""
        # L2归一化
        x = F.normalize(x, p=2, dim=1)
        W = F.normalize(self.weight, p=2, dim=1)
        
        # 计算余弦相似度
        cosine = F.linear(x, W)
        cosine = torch.clamp(cosine, -1.0, 1.0)
        
        if labels is None:
            return cosine * self.s
        
        # 使用当前margin
        m = self.current_m.item()
        cos_m = math.cos(m)
        sin_m = math.sin(m)
        
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * cos_m - sine * sin_m
        
        # One-hot编码
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        # 添加margin
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s
        
        return output


# 单元测试
if __name__ == '__main__':
    print("Testing ArcFace Head...")
    
    # 创建模块
    arcface = ArcFaceHead(
        in_features=128,
        num_classes=6,
        s=30.0,
        m=0.50
    )
    
    # 创建测试数据
    batch_size = 8
    x = torch.randn(batch_size, 128)
    labels = torch.randint(0, 6, (batch_size,))
    
    # 前向传播 - 训练模式
    print("\n1. Testing training mode (with labels)...")
    logits_train = arcface(x, labels)
    print(f"✓ Training logits shape: {logits_train.shape}")  # (8, 6)
    print(f"✓ Logits range: [{logits_train.min():.2f}, {logits_train.max():.2f}]")
    
    # 前向传播 - 推理模式
    print("\n2. Testing inference mode (without labels)...")
    logits_infer = arcface(x)
    print(f"✓ Inference logits shape: {logits_infer.shape}")
    print(f"✓ Logits range: [{logits_infer.min():.2f}, {logits_infer.max():.2f}]")
    
    # 测试概率输出
    print("\n3. Testing softmax probabilities...")
    probs = F.softmax(logits_infer, dim=1)
    print(f"✓ Probabilities sum: {probs.sum(dim=1)}")
    print(f"✓ Sample probabilities: {probs[0]}")
    
    # 测试自适应版本
    print("\n4. Testing AdaptiveArcFaceHead...")
    adaptive_arcface = AdaptiveArcFaceHead(
        in_features=128,
        num_classes=6,
        m_init=0.30,
        m_max=0.50,
        m_schedule='linear'
    )
    
    # 模拟训练进度
    for progress in [0.0, 0.25, 0.5, 0.75, 1.0]:
        adaptive_arcface.update_margin(progress)
        logits = adaptive_arcface(x, labels)
        print(f"✓ Progress={progress:.2f}, margin={adaptive_arcface.current_m:.3f}, "
              f"logits_max={logits.max():.2f}")
    
    # 参数统计
    print("\n5. Model statistics...")
    total_params = sum(p.numel() for p in arcface.parameters())
    print(f"✓ ArcFaceHead parameters: {total_params:,}")
    
    total_params_adaptive = sum(p.numel() for p in adaptive_arcface.parameters())
    print(f"✓ AdaptiveArcFaceHead parameters: {total_params_adaptive:,}")
    
    # 测试权重归一化
    print("\n6. Testing weight normalization...")
    with torch.no_grad():
        W = F.normalize(arcface.weight, p=2, dim=1)
        norms = torch.norm(W, p=2, dim=1)
        print(f"✓ Weight norms: {norms}")  # 应该全是1
    
    print("\n✅ All tests passed!")
