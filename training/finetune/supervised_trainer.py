"""
Supervised Learning Trainer
支持有监督微调/训练
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

from training.optimizer_factory import OptimizerFactory
from training.scheduler_factory import SchedulerFactory


class SupervisedTrainer:
    """有监督学习训练器"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        criterion: Optional[nn.Module] = None,
        device: str = 'cuda',
        optimizer_config: Optional[Dict[str, Any]] = None,
        scheduler_config: Optional[Dict[str, Any]] = None,
        max_epochs: int = 100,
        save_dir: str = './checkpoints',
        callbacks: Optional[List] = None,
        log_interval: int = 10,
        use_amp: bool = False,
        gradient_clip_val: Optional[float] = None,
        accumulation_steps: int = 1,
        metrics_fn: Optional[Callable] = None,
    ):
        """
        Args:
            model: 模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            criterion: 损失函数
            device: 设备
            optimizer_config: 优化器配置
            scheduler_config: 调度器配置
            max_epochs: 最大训练轮数
            save_dir: 保存目录
            callbacks: 回调函数列表
            log_interval: 日志打印间隔
            use_amp: 是否使用混合精度训练
            gradient_clip_val: 梯度裁剪阈值
            accumulation_steps: 梯度累积步数
            metrics_fn: 自定义指标计算函数
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
        self.gradient_clip_val = gradient_clip_val
        self.accumulation_steps = accumulation_steps
        self.metrics_fn = metrics_fn
        
        # 创建损失函数
        if criterion is None:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = criterion
        
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
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
        
        # 当前状态
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.stop_training = False
        
        # 创建保存目录
        os.makedirs(self.save_dir, exist_ok=True)
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        num_batches = 0
        
        # 存储所有预测和标签用于计算更多指标
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch+1}/{self.max_epochs}')
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            # 获取数据
            if isinstance(batch, (tuple, list)):
                inputs, labels = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
            else:
                raise ValueError("Batch must contain (inputs, labels)")
            
            # 前向传播
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    loss = loss / self.accumulation_steps
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss = loss / self.accumulation_steps
            
            # 反向传播
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # 梯度累积
            if (batch_idx + 1) % self.accumulation_steps == 0:
                # 梯度裁剪
                if self.gradient_clip_val is not None:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                
                # 更新参数
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            
            # 记录损失（恢复真实损失）
            total_loss += loss.item() * self.accumulation_steps
            num_batches += 1
            
            # 存储预测和标签
            all_preds.append(predicted.cpu())
            all_labels.append(labels.cpu())
            
            # 更新进度条
            if (batch_idx + 1) % self.log_interval == 0:
                avg_loss = total_loss / num_batches
                avg_acc = total_correct / total_samples
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'acc': f'{avg_acc:.4f}'
                })
        
        avg_loss = total_loss / num_batches
        avg_acc = total_correct / total_samples
        
        metrics = {
            'train_loss': avg_loss,
            'train_acc': avg_acc
        }
        
        # 计算额外指标
        if self.metrics_fn is not None:
            all_preds = torch.cat(all_preds)
            all_labels = torch.cat(all_labels)
            extra_metrics = self.metrics_fn(all_preds, all_labels, prefix='train_')
            metrics.update(extra_metrics)
        
        return metrics
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """验证"""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        num_batches = 0
        
        all_preds = []
        all_labels = []
        
        for batch in self.val_loader:
            # 获取数据
            if isinstance(batch, (tuple, list)):
                inputs, labels = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
            else:
                raise ValueError("Batch must contain (inputs, labels)")
            
            # 前向传播
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
            
            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            
            # 记录损失
            total_loss += loss.item()
            num_batches += 1
            
            # 存储预测和标签
            all_preds.append(predicted.cpu())
            all_labels.append(labels.cpu())
        
        avg_loss = total_loss / num_batches
        avg_acc = total_correct / total_samples
        
        metrics = {
            'val_loss': avg_loss,
            'val_acc': avg_acc
        }
        
        # 计算额外指标
        if self.metrics_fn is not None:
            all_preds = torch.cat(all_preds)
            all_labels = torch.cat(all_labels)
            extra_metrics = self.metrics_fn(all_preds, all_labels, prefix='val_')
            metrics.update(extra_metrics)
        
        return metrics
    
    def train(self):
        """完整训练流程"""
        print(f"Starting supervised training")
        print(f"Device: {self.device}")
        print(f"Max epochs: {self.max_epochs}")
        print(f"Use AMP: {self.use_amp}")
        print(f"Gradient accumulation steps: {self.accumulation_steps}")
        
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
            self.history['train_acc'].append(train_metrics['train_acc'])
            if 'val_loss' in val_metrics:
                self.history['val_loss'].append(val_metrics['val_loss'])
                self.history['val_acc'].append(val_metrics['val_acc'])
            self.history['learning_rate'].append(current_lr)
            
            # 打印日志
            epoch_time = time.time() - epoch_start_time
            log_msg = f"Epoch {epoch+1}/{self.max_epochs} - {epoch_time:.2f}s - "
            log_msg += f"train_loss: {train_metrics['train_loss']:.4f} - "
            log_msg += f"train_acc: {train_metrics['train_acc']:.4f}"
            
            if 'val_loss' in val_metrics:
                log_msg += f" - val_loss: {val_metrics['val_loss']:.4f} - "
                log_msg += f"val_acc: {val_metrics['val_acc']:.4f}"
            
            log_msg += f" - lr: {current_lr:.6f}"
            print(log_msg)
            
            # 保存最佳模型
            if 'val_acc' in val_metrics:
                if val_metrics['val_acc'] > self.best_val_acc:
                    self.best_val_acc = val_metrics['val_acc']
                    self.save_checkpoint('best_model.pth')
            
            # 调用on_epoch_end回调
            metrics = {**train_metrics, **val_metrics, 'lr': current_lr}
            for callback in self.callbacks:
                if hasattr(callback, 'on_epoch_end'):
                    stop_training = callback.on_epoch_end(self, epoch, metrics)
                    if stop_training:
                        print(f"Early stopping triggered at epoch {epoch+1}")
                        self.stop_training = True
                        break
            
            # 检查是否需要停止
            if self.stop_training:
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
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'history': self.history,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        filepath = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, filepath)
        # print(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filename: str):
        """加载检查点"""
        filepath = os.path.join(self.save_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.history = checkpoint['history']
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Checkpoint loaded: {filepath}")
    
    @torch.no_grad()
    def predict(self, data_loader: DataLoader) -> tuple:
        """
        预测
        
        Returns:
            predictions: 预测结果
            labels: 真实标签（如果有）
        """
        self.model.eval()
        predictions_list = []
        labels_list = []
        
        for batch in data_loader:
            if isinstance(batch, (tuple, list)):
                inputs = batch[0].to(self.device)
                if len(batch) > 1:
                    labels = batch[1]
                    labels_list.append(labels)
            else:
                inputs = batch.to(self.device)
            
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            predictions_list.append(predicted.cpu())
        
        predictions = torch.cat(predictions_list, dim=0)
        
        if labels_list:
            labels = torch.cat(labels_list, dim=0)
            return predictions, labels
        else:
            return predictions, None


if __name__ == '__main__':
    # 测试代码
    import torch
    from torch.utils.data import TensorDataset
    
    # 创建虚拟数据
    num_samples = 200
    num_features = 64
    num_classes = 10
    
    X_train = torch.randn(num_samples, num_features)
    y_train = torch.randint(0, num_classes, (num_samples,))
    
    X_val = torch.randn(50, num_features)
    y_val = torch.randint(0, num_classes, (50,))
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 创建简单的分类模型
    model = nn.Sequential(
        nn.Linear(num_features, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, num_classes)
    )
    
    # 创建训练器
    trainer = SupervisedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=nn.CrossEntropyLoss(),
        device='cuda' if torch.cuda.is_available() else 'cpu',
        optimizer_config={'name': 'adam', 'learning_rate': 1e-3},
        scheduler_config={'name': 'step', 'step_size': 10, 'gamma': 0.5},
        max_epochs=5,
        save_dir='./test_checkpoints',
        use_amp=False
    )
    
    print("Testing supervised trainer...")
    history = trainer.train()
    print("Training history:", history)
