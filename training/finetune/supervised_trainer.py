"""
有监督训练器
用于微调阶段的有监督学习
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
from augmentation.mixup import mixup_data, mixup_criterion


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
        use_mixup: bool = False,
        mixup_alpha: float = 0.2,
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
            use_mixup: 是否使用mixup
            mixup_alpha: mixup的alpha参数
            gradient_clip_max_norm: 梯度裁剪的最大范数
        """
        super().__init__(
            model, train_loader, val_loader, optimizer, scheduler,
            device, experiment_dir, use_amp
        )
        
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        self.gradient_clip_max_norm = gradient_clip_max_norm
        
        # 创建损失函数
        self.criterion = self._create_criterion(loss_config)
        
        print(f"SupervisedTrainer初始化完成")
        print(f"  - 使用Mixup: {use_mixup}")
        print(f"  - 梯度裁剪: {gradient_clip_max_norm}")
        print(f"  - 损失函数: {type(self.criterion).__name__}")
    
    def _create_criterion(self, loss_config: Dict[str, Any]):
        """创建损失函数"""
        use_progressive = loss_config.get('use_progressive', True)
        num_classes = loss_config['focal'].get('num_classes', 6)
        
        # 计算类别权重
        focal_config = loss_config['focal']
        class_weight_method = focal_config.get('class_weight_method')
        class_counts = focal_config.get('class_counts')
        
        if class_weight_method and class_counts:
            focal_alpha = compute_class_weights(class_counts, method=class_weight_method)
        else:
            focal_alpha = None
        
        if use_progressive:
            # 使用渐进式损失
            criterion = ProgressiveCombinedLoss(
                num_classes=num_classes,
                focal_alpha=focal_alpha,
                focal_gamma_init=focal_config.get('gamma', 2.0),
                focal_gamma_min=loss_config.get('progressive', {}).get('focal_gamma_schedule', {}).get('min', 1.0),
                arcface_weight_init=loss_config['arcface'].get('weight_init', 0.3),
                arcface_weight_max=loss_config['arcface'].get('weight_max', 0.7),
                label_smoothing=loss_config.get('label_smoothing', 0.1)
            )
        else:
            # 使用固定权重损失
            criterion = CombinedLoss(
                num_classes=num_classes,
                focal_alpha=focal_alpha,
                focal_gamma=focal_config.get('gamma', 2.0),
                focal_weight=focal_config.get('weight', 1.0),
                arcface_weight=loss_config['arcface'].get('weight_init', 0.5),
                label_smoothing=loss_config.get('label_smoothing', 0.1)
            )
        
        return criterion.to(self.device)
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        # 更新渐进式损失的调度
        if isinstance(self.criterion, ProgressiveCombinedLoss):
            # 假设总共100个epoch
            progress = self.current_epoch / 100.0
            self.criterion.update_schedule(progress)
        
        pbar = tqdm(self.train_loader, desc=f'Train Epoch {self.current_epoch+1}')
        
        for batch_idx, batch in enumerate(pbar):
            # 移动数据到设备
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # 计算损失
            loss, loss_dict = self.compute_loss(batch)
            
            # 反向传播
            self.optimizer.zero_grad()
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_max_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_max_norm)
                self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            
            # 计算准确率(使用softmax logits)
            if 'predictions' in batch:
                preds = batch['predictions']
                correct += (preds == batch['label']).sum().item()
                total += batch['label'].size(0)
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100.0 * correct / total:.2f}%" if total > 0 else "N/A"
            })
            
            self.global_step += 1
        
        # 计算平均指标
        avg_loss = total_loss / len(self.train_loader)
        avg_acc = correct / total if total > 0 else 0.0
        
        return {
            'train_loss': avg_loss,
            'train_acc': avg_acc
        }
    
    def validate_epoch(self) -> Dict[str, float]:
        """验证一个epoch"""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Val Epoch {self.current_epoch+1}')
            
            for batch in pbar:
                # 移动数据到设备
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # 前向传播(评估模式)
                logits = self.model(batch, mode='eval')
                labels = batch['label']
                
                # 计算损失(仅用于监控)
                loss = F.cross_entropy(logits, labels)
                total_loss += loss.item()
                
                # 计算准确率
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
                # 更新进度条
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{100.0 * correct / total:.2f}%"
                })
        
        # 计算平均指标
        avg_loss = total_loss / len(self.val_loader)
        avg_acc = correct / total
        
        return {
            'val_loss': avg_loss,
            'val_acc': avg_acc
        }
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算损失
        
        Args:
            batch: 数据batch
        
        Returns:
            loss: 总损失
            loss_dict: 损失详情字典
        """
        # Mixup增强
        if self.use_mixup and self.model.training:
            # 对输入进行mixup
            temporal = batch['temporal']
            labels = batch['label']
            
            temporal, labels_a, labels_b, lam = mixup_data(
                temporal, labels, alpha=self.mixup_alpha, device=self.device
            )
            
            # 更新batch
            batch['temporal'] = temporal
            
            # 前向传播
            softmax_logits, arcface_logits, _ = self.model(batch, mode='train')
            
            # 计算mixup损失
            loss, loss_dict = self.criterion(softmax_logits, arcface_logits, labels_a)
            loss_b, _ = self.criterion(softmax_logits, arcface_logits, labels_b)
            loss = lam * loss + (1 - lam) * loss_b
            
            # 记录预测(用于计算准确率)
            batch['predictions'] = torch.argmax(softmax_logits, dim=1)
        
        else:
            # 不使用mixup的正常训练
            softmax_logits, arcface_logits, _ = self.model(batch, mode='train')
            labels = batch['label']
            
            # 计算损失
            loss, loss_dict = self.criterion(softmax_logits, arcface_logits, labels)
            
            # 记录预测
            batch['predictions'] = torch.argmax(softmax_logits, dim=1)
        
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


if __name__ == '__main__':
    """测试代码"""
    print("=" * 70)
    print("SupervisedTrainer测试")
    print("=" * 70)
    
    # 这里只是基本的结构测试
    # 实际使用时需要通过launcher来创建trainer
    
    print("✓ SupervisedTrainer模块加载成功")
    print("=" * 70)
