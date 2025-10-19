"""
对比学习训练器
用于预训练阶段的自监督学习
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple
from tqdm import tqdm
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.trainer_base import TrainerBase
from losses import NTXentLoss, SupConLoss


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
        
        print(f"ContrastiveTrainer初始化完成")
        print(f"  - 损失类型: {loss_type}")
        print(f"  - 温度参数: {temperature}")
        print(f"  - 梯度裁剪: {gradient_clip_max_norm}")
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f'Pretrain Epoch {self.current_epoch+1}')
        
        for batch_idx, batch in enumerate(pbar):
            # 移动数据到设备
            # 对比学习数据集返回两个增强版本
            if 'aug1' in batch and 'aug2' in batch:
                # 两个增强版本
                aug1_batch = {k.replace('aug1_', ''): v.to(self.device) if isinstance(v, torch.Tensor) else v
                             for k, v in batch.items() if 'aug1_' in k}
                aug2_batch = {k.replace('aug2_', ''): v.to(self.device) if isinstance(v, torch.Tensor) else v
                             for k, v in batch.items() if 'aug2_' in k}
            else:
                raise ValueError("对比学习数据集应返回两个增强版本(aug1, aug2)")
            
            # 计算损失
            loss, loss_dict = self.compute_loss(aug1_batch, aug2_batch)
            
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
            
            # 更新进度条
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            self.global_step += 1
        
        # 计算平均指标
        avg_loss = total_loss / len(self.train_loader)
        
        return {
            'train_loss': avg_loss,
            'contrastive_loss': avg_loss
        }
    
    def validate_epoch(self) -> Dict[str, float]:
        """
        验证一个epoch
        对比学习的验证比较特殊,主要监控loss
        """
        self.model.eval()
        
        total_loss = 0.0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Val Epoch {self.current_epoch+1}')
            
            for batch in pbar:
                # 移动数据到设备
                if 'aug1' in batch and 'aug2' in batch:
                    aug1_batch = {k.replace('aug1_', ''): v.to(self.device) if isinstance(v, torch.Tensor) else v
                                 for k, v in batch.items() if 'aug1_' in k}
                    aug2_batch = {k.replace('aug2_', ''): v.to(self.device) if isinstance(v, torch.Tensor) else v
                                 for k, v in batch.items() if 'aug2_' in k}
                    
                    # 计算损失
                    loss, _ = self.compute_loss(aug1_batch, aug2_batch)
                    total_loss += loss.item()
                
                # 更新进度条
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # 计算平均指标
        avg_loss = total_loss / len(self.val_loader)
        
        return {
            'val_loss': avg_loss
        }
    
    def compute_loss(
        self,
        aug1_batch: Dict[str, torch.Tensor],
        aug2_batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算对比学习损失
        
        Args:
            aug1_batch: 第一个增强版本的batch
            aug2_batch: 第二个增强版本的batch
        
        Returns:
            loss: 对比损失
            loss_dict: 损失详情字典
        """
        # 获取两个增强版本的特征
        # 对比学习只需要特征向量,不需要分类logits
        _, _, features1 = self.model(aug1_batch, mode='train')  # (B, D)
        _, _, features2 = self.model(aug2_batch, mode='train')  # (B, D)
        
        # L2归一化特征
        features1 = F.normalize(features1, p=2, dim=1)
        features2 = F.normalize(features2, p=2, dim=1)
        
        # 计算对比损失
        if self.loss_type == 'ntxent':
            # NT-Xent损失
            loss = self.criterion(features1, features2)
        elif self.loss_type == 'supcon':
            # SupCon损失(需要标签)
            if 'label' in aug1_batch:
                labels = aug1_batch['label']
                # 合并特征
                features = torch.stack([features1, features2], dim=1)  # (B, 2, D)
                loss = self.criterion(features, labels)
            else:
                # 如果没有标签,退化为NT-Xent
                loss = NTXentLoss(temperature=self.temperature)(features1, features2)
        
        loss_dict = {
            'contrastive_loss': loss.item()
        }
        
        return loss, loss_dict
    
    def extract_features(self, dataloader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        提取特征用于可视化
        
        Args:
            dataloader: 数据加载器
        
        Returns:
            features: 特征矩阵 (N, D)
            labels: 标签向量 (N,)
        """
        self.model.eval()
        
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Extracting features'):
                # 移动数据到设备
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # 获取特征
                _, _, features = self.model(batch, mode='train')
                
                all_features.append(features.cpu())
                if 'label' in batch:
                    all_labels.append(batch['label'].cpu())
        
        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0) if all_labels else None
        
        return all_features, all_labels
    
    def save_pretrained_weights(self, save_path: str):
        """
        保存预训练权重
        
        Args:
            save_path: 保存路径
        """
        # 只保存backbone的权重
        backbone_state_dict = {}
        
        for name, param in self.model.named_parameters():
            # 只保存backbone参数
            if any(x in name for x in ['temporal_branch', 'frequency_branch', 'timefreq_branch', 'fusion']):
                backbone_state_dict[name] = param.data
        
        checkpoint = {
            'epoch': self.current_epoch,
            'backbone_state_dict': backbone_state_dict,
            'model_state_dict': self.model.state_dict()  # 也保存完整模型
        }
        
        torch.save(checkpoint, save_path)
        print(f"预训练权重已保存: {save_path}")


if __name__ == '__main__':
    """测试代码"""
    print("=" * 70)
    print("ContrastiveTrainer测试")
    print("=" * 70)
    
    # 这里只是基本的结构测试
    # 实际使用时需要通过launcher来创建trainer
    
    print("✓ ContrastiveTrainer模块加载成功")
    print("=" * 70)
