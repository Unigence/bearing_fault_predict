"""
Mixup数据增强
包含标准Mixup和Manifold Mixup
"""
import torch
import numpy as np


class Mixup:
    """
    标准Mixup增强
    在输入空间混合两个样本
    
    Reference:
        mixup: Beyond Empirical Risk Minimization (ICLR 2018)
    """
    
    def __init__(self, alpha=0.2, prob=0.5):
        """
        Args:
            alpha: Beta分布参数，控制混合强度
            prob: 应用Mixup的概率
        """
        self.alpha = alpha
        self.prob = prob
    
    def __call__(self, x1, x2, y1, y2):
        """
        Args:
            x1, x2: 两个输入样本
            y1, y2: 对应的标签（one-hot或标量）
        
        Returns:
            x_mix: 混合后的输入
            y_mix: 混合后的标签
            lam: 混合系数
        """
        # 判断是否应用
        if np.random.rand() > self.prob:
            return x1, y1, 1.0
        
        # 从Beta分布采样混合系数
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
        
        # 混合输入
        x_mix = lam * x1 + (1 - lam) * x2
        
        # 混合标签
        y_mix = lam * y1 + (1 - lam) * y2
        
        return x_mix, y_mix, lam
    
    def batch_mixup(self, batch_x, batch_y):
        """
        批量Mixup：随机配对batch内的样本
        
        Args:
            batch_x: (B, ...) 输入batch
            batch_y: (B, num_classes) 标签batch (one-hot)
        
        Returns:
            mixed_x: 混合后的输入
            mixed_y: 混合后的标签
        """
        if np.random.rand() > self.prob:
            return batch_x, batch_y
        
        batch_size = batch_x.size(0)
        
        # 随机打乱索引
        indices = torch.randperm(batch_size)
        
        # 从Beta分布采样
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
        
        # 混合
        mixed_x = lam * batch_x + (1 - lam) * batch_x[indices]
        mixed_y = lam * batch_y + (1 - lam) * batch_y[indices]
        
        return mixed_x, mixed_y


class CutMix:
    """
    CutMix增强
    随机裁剪并粘贴两个样本的片段
    
    适用于1D信号
    """
    
    def __init__(self, alpha=1.0, prob=0.5):
        """
        Args:
            alpha: Beta分布参数
            prob: 应用概率
        """
        self.alpha = alpha
        self.prob = prob
    
    def __call__(self, x1, x2, y1, y2):
        """
        Args:
            x1, x2: (seq_len,) or (C, seq_len)
            y1, y2: 标签
        
        Returns:
            x_cut: 混合后的输入
            y_cut: 混合后的标签
            lam: 混合系数
        """
        if np.random.rand() > self.prob:
            return x1, y1, 1.0
        
        # 从Beta分布采样
        lam = np.random.beta(self.alpha, self.alpha)
        
        # 确定形状
        if x1.dim() == 1:
            seq_len = x1.size(0)
        else:
            seq_len = x1.size(1)
        
        # 计算裁剪长度
        cut_len = int(seq_len * (1 - lam))
        
        # 随机裁剪位置
        cut_start = np.random.randint(0, seq_len - cut_len + 1)
        cut_end = cut_start + cut_len
        
        # 复制x1
        x_cut = x1.clone()
        
        # 替换片段
        if x1.dim() == 1:
            x_cut[cut_start:cut_end] = x2[cut_start:cut_end]
        else:
            x_cut[:, cut_start:cut_end] = x2[:, cut_start:cut_end]
        
        # 调整lambda（实际混合比例）
        lam = 1 - (cut_len / seq_len)
        
        # 混合标签
        y_cut = lam * y1 + (1 - lam) * y2
        
        return x_cut, y_cut, lam


class ManifoldMixup:
    """
    Manifold Mixup
    在特征空间（中间层）进行混合
    
    需要在训练循环中使用，不在Dataset中应用
    """
    
    def __init__(self, alpha=0.2, prob=0.5):
        """
        Args:
            alpha: Beta分布参数
            prob: 应用概率
        """
        self.alpha = alpha
        self.prob = prob
    
    def __call__(self, features1, features2, y1, y2):
        """
        在特征空间混合
        
        Args:
            features1, features2: 中间层特征
            y1, y2: 标签
        
        Returns:
            mixed_features: 混合特征
            mixed_y: 混合标签
            lam: 混合系数
        """
        if np.random.rand() > self.prob:
            return features1, y1, 1.0
        
        # 从Beta分布采样
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
        
        # 混合特征
        mixed_features = lam * features1 + (1 - lam) * features2
        
        # 混合标签
        mixed_y = lam * y1 + (1 - lam) * y2
        
        return mixed_features, mixed_y, lam


class MixupCollator:
    """
    用于DataLoader的collate_fn
    在batch拼接时应用Mixup
    """
    
    def __init__(self, mixup_fn, num_classes):
        """
        Args:
            mixup_fn: Mixup实例
            num_classes: 类别数
        """
        self.mixup_fn = mixup_fn
        self.num_classes = num_classes
    
    def __call__(self, batch):
        """
        Args:
            batch: list of (data, label)
        
        Returns:
            mixed_data: (B, ...)
            mixed_labels: (B, num_classes)
        """
        # 拆分数据和标签
        data_list = []
        label_list = []
        
        for item in batch:
            if isinstance(item, dict):
                # 如果是字典格式
                data_list.append(item['data'])
                label_list.append(item['label'])
            else:
                # 如果是tuple格式
                data, label = item
                data_list.append(data)
                label_list.append(label)
        
        # 堆叠
        batch_data = torch.stack(data_list, dim=0)
        batch_labels = torch.tensor(label_list)
        
        # 转换为one-hot
        batch_labels_onehot = torch.nn.functional.one_hot(
            batch_labels, 
            num_classes=self.num_classes
        ).float()
        
        # 应用Mixup
        if hasattr(self.mixup_fn, 'batch_mixup'):
            mixed_data, mixed_labels = self.mixup_fn.batch_mixup(
                batch_data, 
                batch_labels_onehot
            )
        else:
            # 如果没有batch_mixup方法，逐对混合
            mixed_data = batch_data.clone()
            mixed_labels = batch_labels_onehot.clone()
            
            for i in range(0, len(batch_data) - 1, 2):
                mixed_data[i], mixed_labels[i], _ = self.mixup_fn(
                    batch_data[i], 
                    batch_data[i+1],
                    batch_labels_onehot[i],
                    batch_labels_onehot[i+1]
                )
        
        return mixed_data, mixed_labels


def get_mixup_fn(mixup_type='mixup', alpha=0.2, prob=0.5):
    """
    获取Mixup函数的工厂函数
    
    Args:
        mixup_type: 'mixup', 'cutmix', 'manifold'
        alpha: Beta分布参数
        prob: 应用概率
    
    Returns:
        Mixup实例
    """
    mixup_types = {
        'mixup': Mixup,
        'cutmix': CutMix,
        'manifold': ManifoldMixup,
    }
    
    if mixup_type not in mixup_types:
        raise ValueError(f"Unknown mixup type: {mixup_type}")
    
    return mixup_types[mixup_type](alpha=alpha, prob=prob)


if __name__ == "__main__":
    """测试代码"""
    
    print("测试Mixup数据增强")
    print("=" * 70)
    
    # 创建测试数据
    x1 = torch.randn(512)
    x2 = torch.randn(512)
    y1 = torch.tensor([1, 0, 0, 0, 0, 0]).float()  # one-hot, 类别0
    y2 = torch.tensor([0, 1, 0, 0, 0, 0]).float()  # one-hot, 类别1
    
    # 测试标准Mixup
    print("\n1. 标准Mixup")
    mixup = Mixup(alpha=0.2, prob=1.0)
    x_mix, y_mix, lam = mixup(x1, x2, y1, y2)
    print(f"   输入1: {x1.shape}, 标签1: {y1}")
    print(f"   输入2: {x2.shape}, 标签2: {y2}")
    print(f"   混合后: {x_mix.shape}, 标签: {y_mix}, λ={lam:.3f}")
    
    # 测试CutMix
    print("\n2. CutMix")
    cutmix = CutMix(alpha=1.0, prob=1.0)
    x_cut, y_cut, lam = cutmix(x1, x2, y1, y2)
    print(f"   混合后: {x_cut.shape}, 标签: {y_cut}, λ={lam:.3f}")
    
    # 测试批量Mixup
    print("\n3. 批量Mixup")
    batch_x = torch.randn(8, 512)
    batch_y = torch.nn.functional.one_hot(torch.randint(0, 6, (8,)), 6).float()
    
    mixup = Mixup(alpha=0.2, prob=1.0)
    mixed_x, mixed_y = mixup.batch_mixup(batch_x, batch_y)
    print(f"   输入batch: {batch_x.shape}")
    print(f"   混合后: {mixed_x.shape}, 标签: {mixed_y.shape}")
    
    # 测试Manifold Mixup
    print("\n4. Manifold Mixup")
    feat1 = torch.randn(256)
    feat2 = torch.randn(256)
    
    manifold_mixup = ManifoldMixup(alpha=0.2, prob=1.0)
    mixed_feat, mixed_y, lam = manifold_mixup(feat1, feat2, y1, y2)
    print(f"   特征1: {feat1.shape}")
    print(f"   混合后: {mixed_feat.shape}, λ={lam:.3f}")
    
    print("\n✓ Mixup数据增强测试完成!")
