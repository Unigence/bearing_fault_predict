"""
Softmax分类器
标准的全连接分类头，用于推理阶段
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftmaxHead(nn.Module):
    """
    Softmax分类器
    包含特征解耦与增强层 + Softmax输出层
    """
    
    def __init__(self,
                 in_features=128,
                 num_classes=6,
                 hidden_dim=256,
                 dropout1=0.4,
                 dropout2=0.3,
                 use_l2_norm=True):
        """
        Args:
            in_features: 输入特征维度
            num_classes: 类别数
            hidden_dim: 隐藏层维度
            dropout1: 第一层Dropout率
            dropout2: 第二层Dropout率
            use_l2_norm: 是否对特征进行L2归一化
        """
        super(SoftmaxHead, self).__init__()
        
        self.in_features = in_features
        self.num_classes = num_classes
        self.use_l2_norm = use_l2_norm
        
        # 特征解耦与增强
        self.feature_enhance = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout1),
            
            nn.Linear(hidden_dim, in_features),
            nn.BatchNorm1d(in_features),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout2)
        )
        
        # 分类器
        self.classifier = nn.Linear(in_features, num_classes)
        
    def forward(self, x, return_features=False):
        """
        前向传播
        
        Args:
            x: 输入特征 (B, in_features)
            return_features: 是否返回归一化后的特征
        
        Returns:
            logits: 分类logits (B, num_classes)
            features: 归一化特征 (B, in_features) - 如果return_features=True
        """
        # 特征增强
        enhanced_feat = self.feature_enhance(x)
        
        # L2归一化（可选）
        if self.use_l2_norm:
            normalized_feat = F.normalize(enhanced_feat, p=2, dim=1)
        else:
            normalized_feat = enhanced_feat
        
        # 分类
        logits = self.classifier(normalized_feat)
        
        if return_features:
            return logits, normalized_feat
        
        return logits


class SimpleSoftmaxHead(nn.Module):
    """
    简化版Softmax分类器
    只包含一层全连接
    """
    
    def __init__(self, in_features=128, num_classes=6):
        """
        Args:
            in_features: 输入特征维度
            num_classes: 类别数
        """
        super(SimpleSoftmaxHead, self).__init__()
        
        self.classifier = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征 (B, in_features)
        
        Returns:
            logits: 分类logits (B, num_classes)
        """
        return self.classifier(x)


# 单元测试
if __name__ == '__main__':
    print("Testing Softmax Head...")
    
    # 创建模块
    head = SoftmaxHead(
        in_features=128,
        num_classes=6,
        hidden_dim=256
    )
    
    # 创建测试数据
    batch_size = 8
    x = torch.randn(batch_size, 128)
    
    # 前向传播
    print("\n1. Testing basic forward...")
    logits = head(x)
    print(f"✓ Logits shape: {logits.shape}")  # (8, 6)
    
    # 测试返回特征
    print("\n2. Testing with features...")
    logits, features = head(x, return_features=True)
    print(f"✓ Logits shape: {logits.shape}")
    print(f"✓ Features shape: {features.shape}")  # (8, 128)
    print(f"✓ Features L2 norm: {torch.norm(features, p=2, dim=1)}")  # 应该全是1
    
    # 测试Softmax输出
    print("\n3. Testing softmax probabilities...")
    probs = F.softmax(logits, dim=1)
    print(f"✓ Probabilities shape: {probs.shape}")
    print(f"✓ Probabilities sum: {probs.sum(dim=1)}")  # 应该全是1
    print(f"✓ Sample probabilities: {probs[0]}")
    
    # 测试简化版
    print("\n4. Testing SimpleSoftmaxHead...")
    simple_head = SimpleSoftmaxHead(in_features=128, num_classes=6)
    logits_simple = simple_head(x)
    print(f"✓ Simple logits shape: {logits_simple.shape}")
    
    # 参数统计
    print("\n5. Model statistics...")
    total_params = sum(p.numel() for p in head.parameters())
    print(f"✓ SoftmaxHead parameters: {total_params:,}")
    
    total_params_simple = sum(p.numel() for p in simple_head.parameters())
    print(f"✓ SimpleSoftmaxHead parameters: {total_params_simple:,}")
    
    print("\n✅ All tests passed!")
