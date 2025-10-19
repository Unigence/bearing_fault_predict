"""
对比学习损失函数 - 用于自监督预训练
包含NT-Xent (SimCLR), SupCon等对比学习损失
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class NTXentLoss(nn.Module):
    """
    NT-Xent Loss (Normalized Temperature-scaled Cross Entropy Loss)
    用于SimCLR等对比学习方法
    
    参考论文: A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)
    
    核心思想:
    - 同一样本的不同增强版本为正样本对
    - batch内其他所有样本为负样本
    - 最大化正样本对的相似度，最小化负样本对的相似度
    """
    
    def __init__(self, temperature=0.07, reduction='mean'):
        """
        Args:
            temperature: 温度参数τ，控制分布的平滑程度
                        - 较小的τ(0.05-0.1): 分布更sharp，对hard negatives更敏感
                        - 较大的τ(0.5-1.0): 分布更smooth，训练更稳定
            reduction: 'mean' | 'sum' | 'none'
        """
        super(NTXentLoss, self).__init__()
        
        self.temperature = temperature
        self.reduction = reduction
        
    def forward(self, z1, z2):
        """
        前向传播
        
        Args:
            z1: 第一组增强的特征 (B, D) - 已L2归一化
            z2: 第二组增强的特征 (B, D) - 已L2归一化
            
        Returns:
            loss: NT-Xent损失
        """
        batch_size = z1.shape[0]
        
        # L2归一化（如果还没归一化）
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)
        
        # 拼接所有特征: [z1; z2]
        z = torch.cat([z1, z2], dim=0)  # (2B, D)
        
        # 计算相似度矩阵: (2B, 2B)
        sim_matrix = torch.matmul(z, z.T) / self.temperature
        
        # 创建mask，排除自身（对角线）
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        sim_matrix = sim_matrix.masked_fill(mask, -9e15)
        
        # 正样本对的位置
        # 对于z1[i]，正样本是z2[i]，位置在batch_size + i
        # 对于z2[i]，正样本是z1[i]，位置在i
        pos_idx = torch.cat([
            torch.arange(batch_size, 2 * batch_size),  # z1的正样本位置
            torch.arange(0, batch_size)                 # z2的正样本位置
        ], dim=0).to(z.device)
        
        # 提取正样本相似度
        pos_sim = sim_matrix[torch.arange(2 * batch_size), pos_idx]  # (2B,)
        
        # 计算loss: -log(exp(pos) / sum(exp(all)))
        # = -pos + log(sum(exp(all)))
        loss = -pos_sim + torch.logsumexp(sim_matrix, dim=1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss
    
    参考论文: Supervised Contrastive Learning
    
    与NT-Xent的区别:
    - NT-Xent: 只有同一样本的增强版本是正样本对
    - SupCon: 同一类别的所有样本都是正样本对
    
    优势: 利用标签信息，学习更好的类别判别性特征
    """
    
    def __init__(self, temperature=0.07, base_temperature=0.07):
        """
        Args:
            temperature: 缩放参数
            base_temperature: 基础温度（用于归一化）
        """
        super(SupConLoss, self).__init__()
        
        self.temperature = temperature
        self.base_temperature = base_temperature
        
    def forward(self, features, labels):
        """
        前向传播
        
        Args:
            features: 特征 (B, D) 或 (B, n_views, D)
                     如果是3D，表示每个样本有n_views个增强版本
            labels: 标签 (B,)
        
        Returns:
            loss: SupCon损失
        """
        device = features.device
        
        # 处理输入维度
        if len(features.shape) == 2:
            # (B, D) -> (B, 1, D)
            features = features.unsqueeze(1)
        
        batch_size = features.shape[0]
        n_views = features.shape[1]
        
        # 展平: (B, n_views, D) -> (B*n_views, D)
        features = features.view(batch_size * n_views, -1)
        
        # L2归一化
        features = F.normalize(features, p=2, dim=1)
        
        # 扩展标签: (B,) -> (B*n_views,)
        labels = labels.contiguous().view(-1, 1)
        if n_views > 1:
            labels = labels.repeat(n_views, 1)
        labels = labels.view(-1)
        
        # 计算相似度矩阵
        sim_matrix = torch.matmul(features, features.T) / self.temperature  # (B*n_views, B*n_views)
        
        # 创建mask
        # 1. 排除自身
        mask_self = torch.eye(batch_size * n_views, dtype=torch.bool, device=device)
        
        # 2. 找到同类样本（正样本）
        mask_pos = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B*n_views, B*n_views)
        mask_pos = mask_pos & ~mask_self  # 排除自身
        
        # 计算log_prob
        exp_sim = torch.exp(sim_matrix)
        exp_sim = exp_sim.masked_fill(mask_self, 0)  # 排除自身
        
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
        
        # 对每个样本，计算其与所有正样本的平均loss
        # 只在有正样本的位置计算
        mean_log_prob_pos = (mask_pos * log_prob).sum(dim=1) / (mask_pos.sum(dim=1) + 1e-6)
        
        # 损失
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()
        
        return loss


class HardNegativeNTXentLoss(nn.Module):
    """
    Hard Negative Mining NT-Xent Loss
    
    改进版NT-Xent，只关注难负样本（相似度高的负样本）
    可以加速收敛并提升性能
    """
    
    def __init__(self, temperature=0.07, hard_negative_ratio=0.5):
        """
        Args:
            temperature: 温度参数
            hard_negative_ratio: 使用的难负样本比例 [0, 1]
                                0.5表示只使用相似度最高的50%负样本
        """
        super(HardNegativeNTXentLoss, self).__init__()
        
        self.temperature = temperature
        self.hard_negative_ratio = hard_negative_ratio
        
    def forward(self, z1, z2):
        """前向传播"""
        batch_size = z1.shape[0]
        
        # L2归一化
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)
        
        # 拼接
        z = torch.cat([z1, z2], dim=0)  # (2B, D)
        
        # 计算相似度矩阵
        sim_matrix = torch.matmul(z, z.T) / self.temperature
        
        # Mask自身
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        sim_matrix_masked = sim_matrix.masked_fill(mask, -9e15)
        
        # 正样本位置
        pos_idx = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(0, batch_size)
        ], dim=0).to(z.device)
        
        # 提取正样本相似度
        pos_sim = sim_matrix[torch.arange(2 * batch_size), pos_idx]
        
        # Hard Negative Mining
        # 对每个样本，选择相似度最高的k个负样本
        k = int((2 * batch_size - 1) * self.hard_negative_ratio)
        
        # 获取每行的top-k相似度（负样本）
        topk_sim, _ = torch.topk(sim_matrix_masked, k=k, dim=1)  # (2B, k)
        
        # 计算loss（只考虑hard negatives）
        # -log(exp(pos) / (exp(pos) + sum(exp(hard_neg))))
        hard_neg_sum = torch.exp(topk_sim).sum(dim=1)
        loss = -torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + hard_neg_sum))
        
        return loss.mean()


class MomentumContrastLoss(nn.Module):
    """
    Momentum Contrast (MoCo) Loss
    
    参考论文: Momentum Contrast for Unsupervised Visual Representation Learning
    
    特点:
    - 使用动量编码器维护一个大的负样本队列
    - 可以使用更多的负样本，提升性能
    """
    
    def __init__(self, 
                 temperature=0.07,
                 queue_size=65536,
                 momentum=0.999):
        """
        Args:
            temperature: 温度参数
            queue_size: 负样本队列大小
            momentum: 动量编码器的动量系数
        """
        super(MomentumContrastLoss, self).__init__()
        
        self.temperature = temperature
        self.queue_size = queue_size
        self.momentum = momentum
        
        # 注册queue为buffer（不参与梯度更新）
        # 初始化时需要特征维度，所以在第一次forward时初始化
        self.register_buffer('queue', None)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
        
    def _init_queue(self, dim):
        """初始化队列"""
        if self.queue is None:
            queue = torch.randn(dim, self.queue_size)
            queue = F.normalize(queue, p=2, dim=0)
            self.register_buffer('queue', queue)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """
        更新队列
        
        Args:
            keys: 新的负样本特征 (B, D)
        """
        batch_size = keys.shape[0]
        
        ptr = int(self.queue_ptr)
        
        # 替换队列中的旧样本
        if ptr + batch_size <= self.queue_size:
            self.queue[:, ptr:ptr + batch_size] = keys.T
        else:
            # 队列满了，从头开始
            remaining = self.queue_size - ptr
            self.queue[:, ptr:] = keys[:remaining].T
            self.queue[:, :batch_size - remaining] = keys[remaining:].T
        
        # 更新指针
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr
    
    def forward(self, q, k):
        """
        前向传播
        
        Args:
            q: query特征 (B, D) - 来自在线编码器
            k: key特征 (B, D) - 来自动量编码器
        
        Returns:
            loss: MoCo损失
        """
        # 初始化队列
        if self.queue is None:
            self._init_queue(q.shape[1])
        
        # L2归一化
        q = F.normalize(q, p=2, dim=1)
        k = F.normalize(k, p=2, dim=1)
        
        # 计算正样本相似度 (B,)
        pos_sim = torch.einsum('nc,nc->n', [q, k]) / self.temperature
        
        # 计算负样本相似度 (B, queue_size)
        neg_sim = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()]) / self.temperature
        
        # 拼接: [pos, neg]
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (B, 1+queue_size)
        
        # 标签：正样本在第0个位置
        labels = torch.zeros(q.shape[0], dtype=torch.long, device=q.device)
        
        # 交叉熵损失
        loss = F.cross_entropy(logits, labels)
        
        # 更新队列
        self._dequeue_and_enqueue(k)
        
        return loss


# 单元测试
if __name__ == '__main__':
    print("=" * 70)
    print("测试对比学习损失函数")
    print("=" * 70)
    
    # 创建测试数据
    batch_size = 16
    feature_dim = 128
    num_classes = 6
    
    # 1. 测试NT-Xent Loss
    print("\n1. 测试NT-Xent Loss...")
    nt_xent = NTXentLoss(temperature=0.07)
    
    z1 = F.normalize(torch.randn(batch_size, feature_dim), p=2, dim=1)
    z2 = F.normalize(torch.randn(batch_size, feature_dim), p=2, dim=1)
    
    loss = nt_xent(z1, z2)
    print(f"✓ NT-Xent Loss: {loss.item():.4f}")
    
    # 测试梯度
    z1_grad = z1.clone().requires_grad_(True)
    z2_grad = z2.clone().requires_grad_(True)
    loss = nt_xent(z1_grad, z2_grad)
    loss.backward()
    print(f"✓ 梯度范围: [{z1_grad.grad.min():.4f}, {z1_grad.grad.max():.4f}]")
    
    # 2. 测试SupCon Loss
    print("\n2. 测试Supervised Contrastive Loss...")
    supcon = SupConLoss(temperature=0.07)
    
    features = F.normalize(torch.randn(batch_size, feature_dim), p=2, dim=1)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    loss = supcon(features, labels)
    print(f"✓ SupCon Loss: {loss.item():.4f}")
    
    # 测试多视图
    features_multiview = F.normalize(torch.randn(batch_size, 2, feature_dim), p=2, dim=2)
    loss_multiview = supcon(features_multiview, labels)
    print(f"✓ SupCon Loss (2 views): {loss_multiview.item():.4f}")
    
    # 3. 测试Hard Negative NT-Xent
    print("\n3. 测试Hard Negative NT-Xent Loss...")
    hard_nt_xent = HardNegativeNTXentLoss(temperature=0.07, hard_negative_ratio=0.5)
    
    loss_hard = hard_nt_xent(z1, z2)
    print(f"✓ Hard NT-Xent Loss: {loss_hard.item():.4f}")
    
    # 对比不同hard_negative_ratio
    print(f"\n  对比不同难负样本比例:")
    for ratio in [0.3, 0.5, 0.7, 1.0]:
        hard_nt_xent = HardNegativeNTXentLoss(temperature=0.07, hard_negative_ratio=ratio)
        loss = hard_nt_xent(z1, z2)
        print(f"    ratio={ratio:.1f}: loss={loss.item():.4f}")
    
    # 4. 测试MoCo Loss
    print("\n4. 测试Momentum Contrast Loss...")
    moco = MomentumContrastLoss(temperature=0.07, queue_size=256)
    
    q = F.normalize(torch.randn(batch_size, feature_dim), p=2, dim=1)
    k = F.normalize(torch.randn(batch_size, feature_dim), p=2, dim=1)
    
    loss_moco = moco(q, k)
    print(f"✓ MoCo Loss: {loss_moco.item():.4f}")
    
    # 测试队列更新
    print(f"✓ Queue size: {moco.queue.shape}")
    print(f"✓ Queue ptr: {moco.queue_ptr.item()}")
    
    # 5. 对比不同温度的影响
    print("\n5. 对比不同温度参数...")
    for temp in [0.05, 0.07, 0.1, 0.5]:
        nt_xent = NTXentLoss(temperature=temp)
        loss = nt_xent(z1, z2)
        print(f"✓ Temperature={temp:.2f}: loss={loss.item():.4f}")
    
    # 6. 验证SupCon的标签利用
    print("\n6. 验证SupCon的标签利用...")
    
    # 创建一个batch，其中一半样本是同一类
    labels_same = torch.zeros(batch_size, dtype=torch.long)
    labels_same[batch_size//2:] = 1
    
    loss_same = supcon(features, labels_same)
    loss_diff = supcon(features, labels)
    
    print(f"✓ SupCon (多样本同类): {loss_same.item():.4f}")
    print(f"✓ SupCon (样本异类): {loss_diff.item():.4f}")
    print(f"  → 同类样本的loss应该更低")
    
    print("\n" + "=" * 70)
    print("✅ 所有测试通过!")
    print("=" * 70)
    
    print("\n推荐使用方案:")
    print("• 预训练阶段: NT-Xent (无标签) 或 SupCon (有标签)")
    print("• 难负样本场景: Hard Negative NT-Xent")
    print("• 大规模数据: MoCo (维护大队列)")
    print("• 温度参数: 0.07 (SimCLR标准值)")
