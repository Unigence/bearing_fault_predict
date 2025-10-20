"""
有监督训练器
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple, Optional
from tqdm import tqdm
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.trainer_base import TrainerBase
from losses import CombinedLoss, ProgressiveCombinedLoss, compute_class_weights
from augmentation.mixup import MultiModalMixup, ManifoldMixup


class SupervisedTrainer(TrainerBase):
    """有监督训练器"""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        loss_config: Dict[str, Any],
        device: str = 'cuda',
        experiment_dir: str = 'experiments/runs',
        use_amp: bool = False,
        mixup_config: Optional[Dict[str, Any]] = None,
        gradient_clip_max_norm: float = 1.0
    ):
        """
        Args:
            model: 模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            optimizer: 优化器
            scheduler: 学习率调度器
            loss_config: 损失函数配置
            device: 训练设备
            experiment_dir: 实验保存目录
            use_amp: 是否使用混合精度训练
            mixup_config: Mixup配置字典，包含time_domain, frequency_domain, feature_level配置
            gradient_clip_max_norm: 梯度裁剪的最大范数
        """
        super().__init__(
            model, train_loader, val_loader, optimizer, scheduler,
            device, experiment_dir, use_amp
        )

        self.gradient_clip_max_norm = gradient_clip_max_norm

        # 创建损失函数
        self.criterion = self._create_criterion(loss_config)

        # 创建Mixup管理器
        self.mixup_manager = None
        if mixup_config:
            self.mixup_manager = MultiModalMixup(
                time_domain_config=mixup_config.get('time_domain'),
                frequency_domain_config=mixup_config.get('frequency_domain'),
                feature_level_config=mixup_config.get('feature_level')
            )

        print(f"SupervisedTrainer初始化完成")
        if self.mixup_manager:
            print(f"  - 时域Mixup: {self.mixup_manager.time_domain_enabled}")
            print(f"  - 频域Mixup: {self.mixup_manager.frequency_domain_enabled}")
            print(f"  - 特征层Mixup: {self.mixup_manager.feature_level_enabled}")
            print(f"  ⚠️  注意: 使用输入层Mixup时将禁用ArcFace损失(仅使用Softmax)")
        else:
            print(f"  - Mixup: 未启用")
        print(f"  - 梯度裁剪: {gradient_clip_max_norm}")

    def _create_criterion(self, loss_config: Dict[str, Any]):
        """
        创建损失函数

        Args:
            loss_config: 损失函数配置

        Returns:
            criterion: 损失函数实例
        """
        if loss_config.get('use_progressive', False):
            # 渐进式组合损失
            criterion = ProgressiveCombinedLoss(
                focal_alpha=loss_config['focal'].get('alpha', None),
                focal_gamma_init=loss_config['focal'].get('gamma_init', None),
                focal_gamma_min=loss_config['focal'].get('gamma_min', None),
                arcface_weight_init=loss_config['arcface'].get('weight_init', None),
                arcface_weight_max=loss_config['arcface'].get('weight_max', None),
                label_smoothing=loss_config.get('label_smoothing', 0.0)
            )
        else:
            # 固定权重组合损失
            criterion = CombinedLoss(
                focal_alpha=loss_config['focal'].get('alpha', None),
                focal_gamma=loss_config['focal'].get('gamma_init', None),
                focal_weight=loss_config['focal'].get('weight', None),
                arcface_weight=loss_config['arcface'].get('weight_init', 0.5),
                label_smoothing=loss_config.get('label_smoothing', 0.0)
            )

        return criterion

    def train_epoch(self) -> Dict[str, float]:
        """
        训练一个epoch

        Returns:
            metrics: 训练指标字典 {'train_loss': ..., 'train_acc': ...}
        """
        self.model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        pbar = tqdm(self.train_loader, desc=f"Training Epoch {self.current_epoch+1}")

        for batch_idx, batch in enumerate(pbar):
            # 应用输入层mixup（如果启用）
            if self.mixup_manager:
                mixed_batch, labels_a, labels_b, lam = self.mixup_manager.apply_input_mixup(
                    batch, self.device
                )
                use_mixup = labels_b is not None
            else:
                mixed_batch = batch
                labels_a = batch['label'].to(self.device)
                labels_b = None
                lam = 1.0
                use_mixup = False

            # 前向传播 - 根据是否使用Mixup决定计算方式
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    softmax_logits, arcface_logits, features = self.model(
                        mixed_batch, mode='supervised'
                    )

                    #  Mixup时只使用Softmax损失,不使用ArcFace
                    if use_mixup:
                        # Mixup情况: 只用Softmax分类损失
                        # 原因: Mixup后的特征不在任何类的流形上,无法计算有意义的角度边界
                        loss_a = F.cross_entropy(softmax_logits, labels_a, reduction='mean')
                        loss_b = F.cross_entropy(softmax_logits, labels_b, reduction='mean')
                        loss = lam * loss_a + (1 - lam) * loss_b
                    else:
                        # 正常情况: 使用完整的组合损失(Focal + ArcFace)
                        loss, loss_dict = self.criterion(softmax_logits, arcface_logits, labels_a)
            else:
                softmax_logits, arcface_logits, features = self.model(
                    mixed_batch, mode='supervised'
                )

                if use_mixup:
                    loss_a = F.cross_entropy(softmax_logits, labels_a, reduction='mean')
                    loss_b = F.cross_entropy(softmax_logits, labels_b, reduction='mean')
                    loss = lam * loss_a + (1 - lam) * loss_b
                else:
                    loss, loss_dict = self.criterion(softmax_logits, arcface_logits, labels_a)

            # 反向传播
            self.optimizer.zero_grad()

            if self.use_amp:
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
                loss.backward()
                if self.gradient_clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip_max_norm
                    )
                self.optimizer.step()

            # 统计准确率
            with torch.no_grad():
                predictions = torch.argmax(softmax_logits, dim=1)

                if use_mixup:
                    # Mixup情况：使用硬标签计算准确率（取lambda较大的那个）
                    if lam > 0.5:
                        correct = (predictions == labels_a).sum().item()
                    else:
                        correct = (predictions == labels_b).sum().item()
                else:
                    correct = (predictions == labels_a).sum().item()

                total_correct += correct
                total_samples += labels_a.size(0)

            # 统计loss
            total_loss += loss.item()
            self.global_step += 1

            # 更新进度条
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': total_correct / total_samples,
                'mixup': use_mixup
            })

        # 计算平均指标
        avg_loss = total_loss / len(self.train_loader)
        avg_acc = total_correct / total_samples

        return {
            'train_loss': avg_loss,
            'train_acc': avg_acc
        }

    def validate_epoch(self) -> Dict[str, float]:
        """
        验证一个epoch

        🔧 修复2: 验证时使用与训练相同的损失函数计算

        Returns:
            metrics: 验证指标字典 {'val_loss': ..., 'val_acc': ...}
        """
        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        pbar = tqdm(self.val_loader, desc=f"Validation Epoch {self.current_epoch+1}")

        with torch.no_grad():
            for batch in pbar:
                labels = batch['label'].to(self.device)

                # 🔧 关键修复: 验证时也使用双头输出和完整损失
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        # 验证模式仍然返回双头输出
                        softmax_logits, arcface_logits, features = self.model(
                            batch, mode='supervised'
                        )

                        # 使用与训练相同的损失函数（但不使用Mixup）
                        loss, loss_dict = self.criterion(softmax_logits, arcface_logits, labels)
                else:
                    softmax_logits, arcface_logits, features = self.model(
                        batch, mode='supervised'
                    )
                    loss, loss_dict = self.criterion(softmax_logits, arcface_logits, labels)

                # 统计准确率 - 使用softmax_logits
                predictions = torch.argmax(softmax_logits, dim=1)
                correct = (predictions == labels).sum().item()

                total_correct += correct
                total_samples += labels.size(0)
                total_loss += loss.item()

                # 更新进度条
                pbar.set_postfix({
                    'loss': loss.item(),
                    'acc': total_correct / total_samples
                })

        # 计算平均指标
        avg_loss = total_loss / len(self.val_loader)
        avg_acc = total_correct / total_samples

        return {
            'val_loss': avg_loss,
            'val_acc': avg_acc
        }

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算损失（用于其他用途，如分析）

        Args:
            batch: 输入batch

        Returns:
            loss: 总损失
            loss_dict: 损失字典
        """
        labels = batch['label'].to(self.device)

        # 前向传播
        softmax_logits, arcface_logits, features = self.model(batch, mode='supervised')

        # 计算损失
        loss, loss_dict = self.criterion(softmax_logits, arcface_logits, labels)

        return loss, loss_dict

    def freeze_backbone(self, freeze_ratio: float = 0.5):
        """
        冻结部分backbone参数

        Args:
            freeze_ratio: 冻结比例(0.0-1.0)
        """
        print(f"\n冻结 {freeze_ratio*100:.0f}% 的backbone参数")
        self.model.freeze_backbone(freeze_ratio=freeze_ratio)

    def unfreeze_all(self):
        """解冻所有参数"""
        print("\n解冻所有参数")
        self.model.unfreeze_all()

    def update_epoch(self, epoch: int, max_epochs: int):
        """
        更新epoch（用于渐进式损失）

        Args:
            epoch: 当前epoch
            max_epochs: 最大epoch数
        """
        if hasattr(self.criterion, 'update_epoch'):
            self.criterion.update_epoch(epoch, max_epochs)


if __name__ == '__main__':
    """测试代码"""
    print("=" * 70)
    print("SupervisedTrainer测试（已修复版本）")
    print("=" * 70)

    # 测试Mixup配置
    mixup_config = {
        'time_domain': {
            'enable': True,
            'alpha': 0.2,
            'prob': 0.5
        },
        'frequency_domain': {
            'enable': False,
            'alpha': 0.2,
            'prob': 0.3,
            'mix_mode': 'magnitude'
        },
        'feature_level': {
            'enable': False,
            'alpha': 0.2,
            'prob': 0.5
        }
    }

    print("\nMixup配置示例:")
    print(f"  - 时域: {mixup_config['time_domain']['enable']}")
    print(f"  - 频域: {mixup_config['frequency_domain']['enable']}")
    print(f"  - 特征层: {mixup_config['feature_level']['enable']}")
    print("\n✅ 修复说明:")
    print("  1. 使用输入层Mixup时,只使用Softmax损失")
    print("  2. 验证时使用与训练相同的完整损失函数")

    print("\n✓ SupervisedTrainer模块加载成功")
    print("=" * 70)
