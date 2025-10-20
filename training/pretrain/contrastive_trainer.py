"""
对比学习训练器
用于预训练阶段的自监督学习
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Dict, Any, Tuple
from tqdm import tqdm
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.trainer_base import TrainerBase
from losses import NTXentLoss, SupConLoss
from training.scheduler_factory import WarmupScheduler


class ContrastiveTrainer(TrainerBase):
    """对比学习训练器"""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        loss_type: str = 'ntxent',
        temperature: float = 0.07,
        device: str = 'cuda',
        experiment_dir: str = 'experiments/runs',
        use_amp: bool = False,
        gradient_clip_max_norm: float = 1.0
    ):
        """
        Args:
            model: 模型
            train_loader: 训练数据加载器(对比学习数据集)
            val_loader: 验证数据加载器
            optimizer: 优化器
            scheduler: 学习率调度器
            loss_type: 损失类型 'ntxent' | 'supcon'
            temperature: 温度参数
            device: 训练设备
            experiment_dir: 实验保存目录
            use_amp: 是否使用混合精度训练
            gradient_clip_max_norm: 梯度裁剪的最大范数
        """
        super().__init__(
            model, train_loader, val_loader, optimizer, scheduler,
            device, experiment_dir, use_amp
        )

        self.loss_type = loss_type
        self.temperature = temperature
        self.gradient_clip_max_norm = gradient_clip_max_norm

        # 创建对比学习损失
        if loss_type == 'ntxent':
            self.criterion = NTXentLoss(temperature=temperature)
        elif loss_type == 'supcon':
            self.criterion = SupConLoss(temperature=temperature)
        else:
            raise ValueError(f"不支持的损失类型: {loss_type}")

        # 检查scheduler是否需要metric
        self.scheduler_needs_metric = self._check_scheduler_needs_metric()

        print(f"ContrastiveTrainer初始化完成")
        print(f"  - 损失类型: {loss_type}")
        print(f"  - 温度参数: {temperature}")
        print(f"  - 梯度裁剪: {gradient_clip_max_norm}")
        print(f"  - Scheduler需要metric: {self.scheduler_needs_metric}")

    def _check_scheduler_needs_metric(self) -> bool:
        """检查scheduler是否需要metric(用于ReduceLROnPlateau)"""
        if self.scheduler is None:
            return False

        # 检查scheduler本身
        if isinstance(self.scheduler, ReduceLROnPlateau):
            return True

        # 检查WarmupScheduler的base_scheduler
        if isinstance(self.scheduler, WarmupScheduler):
            if self.scheduler.base_scheduler is not None:
                if isinstance(self.scheduler.base_scheduler, ReduceLROnPlateau):
                    return True

        return False

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算对比学习损失

        Args:
            batch: 数据batch,应包含'view1'和'view2'两个增强版本

        Returns:
            loss: 总损失
            loss_dict: 损失详情字典
        """
        # 确保batch格式正确
        if 'view1' not in batch or 'view2' not in batch:
            raise ValueError(
                "对比学习数据集应返回两个增强版本(view1, view2)\n"
                "请检查ContrastiveDataset.__getitem__是否正确实现"
            )

        # 将数据移到设备
        view1 = {k: v.to(self.device) for k, v in batch['view1'].items()}
        view2 = {k: v.to(self.device) for k, v in batch['view2'].items()}

        # 前向传播 - 对比学习模式
        z1, z2 = self.model({'view1': view1, 'view2': view2}, mode='contrastive')

        # 计算对比学习损失
        if self.loss_type == 'ntxent':
            loss = self.criterion(z1, z2)
            loss_dict = {
                'contrastive_loss': loss.item(),
                'total': loss.item()
            }
        else:  # supcon
            # 有监督对比学习需要标签
            if 'label' not in batch:
                raise ValueError("SupCon需要标签,但batch中没有'label'")
            labels = batch['label'].to(self.device)
            loss = self.criterion(z1, z2, labels)
            loss_dict = {
                'contrastive_loss': loss.item(),
                'total': loss.item()
            }

        return loss, loss_dict

    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()

        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc='Training')
        for batch in pbar:
            # 🔧 修复: 确保batch包含view1和view2
            if 'view1' not in batch or 'view2' not in batch:
                raise ValueError(
                    "对比学习数据集应返回两个增强版本(view1, view2)\n"
                    "请检查ContrastiveDataset.__getitem__是否正确实现"
                )

            # 将数据移到设备
            view1 = {k: v.to(self.device) for k, v in batch['view1'].items()}
            view2 = {k: v.to(self.device) for k, v in batch['view2'].items()}

            # 有监督对比学习需要标签
            labels = None
            if 'label' in batch:
                labels = batch['label'].to(self.device)

            # 混合精度训练
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    # 前向传播 - 对比学习模式
                    z1, z2 = self.model({'view1': view1, 'view2': view2}, mode='contrastive')

                    # 计算对比学习损失
                    if self.loss_type == 'ntxent':
                        loss = self.criterion(z1, z2)
                    else:  # supcon
                        if labels is None:
                            raise ValueError("SupCon需要标签,但batch中没有'label'")
                        loss = self.criterion(z1, z2, labels)

                # 反向传播
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()

                # 梯度裁剪
                if self.gradient_clip_max_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip_max_norm
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # 标准训练
                z1, z2 = self.model({'view1': view1, 'view2': view2}, mode='contrastive')

                if self.loss_type == 'ntxent':
                    loss = self.criterion(z1, z2)
                else:
                    if labels is None:
                        raise ValueError("SupCon需要标签,但batch中没有'label'")
                    loss = self.criterion(z1, z2, labels)

                self.optimizer.zero_grad()
                loss.backward()

                # 梯度裁剪
                if self.gradient_clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip_max_norm
                    )

                self.optimizer.step()

            # 累计损失
            total_loss += loss.item()
            num_batches += 1

            # 更新进度条
            pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / num_batches
        return {'loss': avg_loss}

    def validate_epoch(self) -> Dict[str, float]:
        """验证一个epoch"""
        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # 确保batch格式正确
                if 'view1' not in batch or 'view2' not in batch:
                    raise ValueError("对比学习数据集应返回两个增强版本(view1, view2)")

                # 将数据移到设备
                view1 = {k: v.to(self.device) for k, v in batch['view1'].items()}
                view2 = {k: v.to(self.device) for k, v in batch['view2'].items()}

                # 获取标签(如果有)
                labels = None
                if 'label' in batch:
                    labels = batch['label'].to(self.device)

                # 前向传播
                z1, z2 = self.model({'view1': view1, 'view2': view2}, mode='contrastive')

                # 计算损失
                if self.loss_type == 'ntxent':
                    loss = self.criterion(z1, z2)
                else:
                    if labels is None:
                        raise ValueError("SupCon需要标签,但batch中没有'label'")
                    loss = self.criterion(z1, z2, labels)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        return {'loss': avg_loss}


# 测试代码
if __name__ == '__main__':
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from models import create_model
    from datasets import ContrastiveDataset
    from torch.utils.data import DataLoader
    from training.optimizer_factory import create_optimizer_from_config
    from training.scheduler_factory import create_scheduler_from_config

    print("=" * 80)
    print("测试ContrastiveTrainer")
    print("=" * 80)

    # 创建模型
    model = create_model(config='small', enable_contrastive=True)
    print(f"✓ 模型创建成功")

    # 创建数据集(使用少量数据测试)
    print("\n创建数据集...")
    train_dataset = ContrastiveDataset(
        data_dir='raw_datasets/train',
        mode='train',
        fold=0,
        n_folds=5,
        use_labels=False,
        cache_data=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0
    )

    print(f"✓ 数据加载器创建成功")

    # 创建优化器和调度器
    optimizer_config = {'type': 'adam', 'lr': 0.001}
    optimizer = create_optimizer_from_config(model, optimizer_config)

    scheduler_config = {'type': 'cosine', 'T_max': 10, 'eta_min': 1e-6}
    scheduler, needs_metric = create_scheduler_from_config(
        optimizer,
        scheduler_config,
        total_epochs=10
    )

    print(f"✓ 优化器和调度器创建成功")
    print(f"  - Scheduler需要metric: {needs_metric}")

    # 创建训练器
    trainer = ContrastiveTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_type='ntxent',
        temperature=0.07,
        device='cpu',
        experiment_dir='/tmp/test_contrastive',
        use_amp=False
    )

    print(f"✓ Trainer创建成功")

    # 测试一个epoch
    print("\n测试训练一个epoch...")
    train_metrics = trainer.train_epoch()
    print(f"✓ 训练完成: loss={train_metrics['loss']:.4f}")

    print("\n测试验证一个epoch...")
    val_metrics = trainer.validate_epoch()
    print(f"✓ 验证完成: loss={val_metrics['loss']:.4f}")

    print("\n" + "=" * 80)
    print("✓ 所有测试通过!")
    print("=" * 80)
