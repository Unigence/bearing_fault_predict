"""
Contrastive Learning Trainer
支持对比学习预训练（NT-Xent / SupCon）
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Callable, List
import time
from tqdm import tqdm
import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from models.losses.contrastive_loss import NTXentLoss, SupConLoss
from training.optimizer_factory import OptimizerFactory
from training.scheduler_factory import SchedulerFactory


class ContrastiveTrainer:
    """对比学习训练器"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        loss_type: str = 'ntxent',  # 'ntxent' or 'supcon'
        temperature: float = 0.5,
        device: str = 'cuda',
        optimizer_config: Optional[Dict[str, Any]] = None,
        scheduler_config: Optional[Dict[str, Any]] = None,
        max_epochs: int = 100,
        save_dir: str = './checkpoints',
        callbacks: Optional[List] = None,
        log_interval: int = 10,
        use_amp: bool = False,  # 混合精度训练
    ):
        """
        Args:
            model: 模型（通常是编码器）
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            loss_type: 损失函数类型
            temperature: 温度参数
            device: 设备
            optimizer_config: 优化器配置
            scheduler_config: 调度器配置
            max_epochs: 最大训练轮数
            save_dir: 保存目录
            callbacks: 回调函数列表
            log_interval: 日志打印间隔
            use_amp: 是否使用混合精度训练
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.max_epochs = max_epochs
        self.save_dir = save_dir
        self.callbacks = callbacks or []
        self.log_interval = log_interval
        self.use_amp = use_amp
        
        # 创建损失函数
        if loss_type.lower() == 'ntxent':
            self.criterion = NTXentLoss(temperature=temperature)
        elif loss_type.lower() == 'supcon':
            self.criterion = SupConLoss(temperature=temperature)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
        
        self.loss_type = loss_type.lower()
        
        # 创建优化器
        if optimizer_config is None:
            optimizer_config = {'name': 'adam', 'learning_rate': 1e-3}
        self.optimizer = OptimizerFactory.create_optimizer(
            model=self.model,
            **optimizer_config
        )
        
        # 创建学习率调度器
        self.scheduler = None
        if scheduler_config is not None:
            self.scheduler = SchedulerFactory.create_scheduler(
                optimizer=self.optimizer,
                **scheduler_config
            )
        
        # 混合精度训练
        self.scaler = None
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        # 当前状态
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        
        # 创建保存目录
        os.makedirs(self.save_dir, exist_ok=True)
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch+1}/{self.max_epochs}')
        
        for batch_idx, batch in enumerate(pbar):
            # 获取数据
            # 对于对比学习，通常batch包含两个增强视图
            if isinstance(batch, (tuple, list)):
                if len(batch) == 3:  # (view1, view2, labels)
                    view1, view2, labels = batch
                    view1 = view1.to(self.device)
                    view2 = view2.to(self.device)
                    labels = labels.to(self.device) if labels is not None else None
                else:  # (view1, view2)
                    view1, view2 = batch
                    view1 = view1.to(self.device)
                    view2 = view2.to(self.device)
                    labels = None
            else:
                raise ValueError("Batch format not supported")
            
            # 前向传播
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    loss = self._compute_loss(view1, view2, labels)
            else:
                loss = self._compute_loss(view1, view2, labels)
            
            # 反向传播
            self.optimizer.zero_grad()
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            # 记录损失
            total_loss += loss.item()
            num_batches += 1
            
            # 更新进度条
            if (batch_idx + 1) % self.log_interval == 0:
                avg_loss = total_loss / num_batches
                pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        avg_loss = total_loss / num_batches
        return {'train_loss': avg_loss}
    
    def _compute_loss(
        self,
        view1: torch.Tensor,
        view2: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """计算对比损失"""
        # 获取特征表示
        z1 = self.model(view1)
        z2 = self.model(view2)
        
        # 计算损失
        if self.loss_type == 'ntxent':
            # NT-Xent loss
            loss = self.criterion(z1, z2)
        else:
            # SupCon loss
            if labels is None:
                raise ValueError("SupCon loss requires labels")
            
            # 拼接特征和标签
            features = torch.cat([z1.unsqueeze(1), z2.unsqueeze(1)], dim=1)
            loss = self.criterion(features, labels)
        
        return loss
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """验证"""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.val_loader:
            # 获取数据
            if isinstance(batch, (tuple, list)):
                if len(batch) == 3:
                    view1, view2, labels = batch
                    view1 = view1.to(self.device)
                    view2 = view2.to(self.device)
                    labels = labels.to(self.device) if labels is not None else None
                else:
                    view1, view2 = batch
                    view1 = view1.to(self.device)
                    view2 = view2.to(self.device)
                    labels = None
            else:
                raise ValueError("Batch format not supported")
            
            # 前向传播
            loss = self._compute_loss(view1, view2, labels)
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        return {'val_loss': avg_loss}
    
    def train(self):
        """完整训练流程"""
        print(f"Starting contrastive pretraining with {self.loss_type.upper()} loss")
        print(f"Device: {self.device}")
        print(f"Max epochs: {self.max_epochs}")
        print(f"Use AMP: {self.use_amp}")
        
        # 调用on_train_begin回调
        for callback in self.callbacks:
            if hasattr(callback, 'on_train_begin'):
                callback.on_train_begin(self)
        
        for epoch in range(self.max_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # 调用on_epoch_begin回调
            for callback in self.callbacks:
                if hasattr(callback, 'on_epoch_begin'):
                    callback.on_epoch_begin(self, epoch)
            
            # 训练一个epoch
            train_metrics = self.train_epoch()
            
            # 验证
            val_metrics = self.validate()
            
            # 更新学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if 'val_loss' in val_metrics:
                        self.scheduler.step(val_metrics['val_loss'])
                else:
                    self.scheduler.step()
            
            # 记录历史
            self.history['train_loss'].append(train_metrics['train_loss'])
            if 'val_loss' in val_metrics:
                self.history['val_loss'].append(val_metrics['val_loss'])
            self.history['learning_rate'].append(current_lr)
            
            # 打印日志
            epoch_time = time.time() - epoch_start_time
            log_msg = f"Epoch {epoch+1}/{self.max_epochs} - {epoch_time:.2f}s - "
            log_msg += f"train_loss: {train_metrics['train_loss']:.4f}"
            if 'val_loss' in val_metrics:
                log_msg += f" - val_loss: {val_metrics['val_loss']:.4f}"
            log_msg += f" - lr: {current_lr:.6f}"
            print(log_msg)
            
            # 保存最佳模型
            if 'val_loss' in val_metrics:
                if val_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_loss']
                    self.save_checkpoint('best_model.pth')
            
            # 调用on_epoch_end回调
            metrics = {**train_metrics, **val_metrics, 'lr': current_lr}
            for callback in self.callbacks:
                if hasattr(callback, 'on_epoch_end'):
                    stop_training = callback.on_epoch_end(self, epoch, metrics)
                    if stop_training:
                        print(f"Early stopping triggered at epoch {epoch+1}")
                        break
            
            # 检查是否需要停止
            if hasattr(self, 'stop_training') and self.stop_training:
                break
        
        # 调用on_train_end回调
        for callback in self.callbacks:
            if hasattr(callback, 'on_train_end'):
                callback.on_train_end(self)
        
        print("Training completed!")
        return self.history
    
    def save_checkpoint(self, filename: str):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        filepath = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filename: str):
        """加载检查点"""
        filepath = os.path.join(self.save_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Checkpoint loaded: {filepath}")
    
    def get_embeddings(
        self,
        data_loader: DataLoader
    ) -> tuple:
        """
        提取嵌入表示
        
        Returns:
            embeddings: 嵌入张量
            labels: 标签（如果有）
        """
        self.model.eval()
        embeddings_list = []
        labels_list = []
        
        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, (tuple, list)):
                    if len(batch) >= 2:
                        data = batch[0].to(self.device)
                        if len(batch) >= 3:
                            labels = batch[2]
                            labels_list.append(labels)
                    else:
                        data = batch[0].to(self.device)
                else:
                    data = batch.to(self.device)
                
                embeddings = self.model(data)
                embeddings_list.append(embeddings.cpu())
        
        embeddings = torch.cat(embeddings_list, dim=0)
        
        if labels_list:
            labels = torch.cat(labels_list, dim=0)
            return embeddings, labels
        else:
            return embeddings, None


if __name__ == '__main__':
    # 测试代码
    import torch
    from torch.utils.data import TensorDataset
    
    # 创建虚拟数据
    num_samples = 100
    num_features = 128
    num_classes = 10
    
    # 模拟两个增强视图
    view1 = torch.randn(num_samples, num_features)
    view2 = torch.randn(num_samples, num_features)
    labels = torch.randint(0, num_classes, (num_samples,))
    
    train_dataset = TensorDataset(view1, view2, labels)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # 创建简单的编码器模型
    encoder = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Linear(256, 128)
    )
    
    # 创建训练器（NT-Xent）
    trainer = ContrastiveTrainer(
        model=encoder,
        train_loader=train_loader,
        loss_type='ntxent',
        temperature=0.5,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        optimizer_config={'name': 'adam', 'learning_rate': 1e-3},
        scheduler_config={'name': 'cosine', 'T_max': 50},
        max_epochs=5,
        save_dir='./test_checkpoints',
        use_amp=False
    )
    
    print("Testing NT-Xent trainer...")
    history = trainer.train()
    print("Training history:", history)
