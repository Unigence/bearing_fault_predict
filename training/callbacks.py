"""
Training Callbacks
提供训练过程中的回调函数，如早停、保存检查点、学习率监控等
"""

import os
import torch
import numpy as np
from typing import Any, Dict, Optional, List
import json
from datetime import datetime


class Callback:
    """回调函数基类"""
    
    def on_train_begin(self, trainer):
        """训练开始时调用"""
        pass
    
    def on_train_end(self, trainer):
        """训练结束时调用"""
        pass
    
    def on_epoch_begin(self, trainer, epoch: int):
        """每个epoch开始时调用"""
        pass
    
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float]) -> bool:
        """
        每个epoch结束时调用
        
        Returns:
            bool: 是否停止训练
        """
        return False
    
    def on_batch_begin(self, trainer, batch: int):
        """每个batch开始时调用"""
        pass
    
    def on_batch_end(self, trainer, batch: int, metrics: Dict[str, float]):
        """每个batch结束时调用"""
        pass


class EarlyStopping(Callback):
    """早停回调"""
    
    def __init__(
        self,
        monitor: str = 'val_loss',
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min',
        restore_best_weights: bool = True,
        verbose: bool = True
    ):
        """
        Args:
            monitor: 监控的指标名称
            patience: 容忍的epoch数
            min_delta: 最小改进量
            mode: 'min' 或 'max'，指标越小越好还是越大越好
            restore_best_weights: 是否恢复最佳权重
            verbose: 是否打印信息
        """
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        
        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
            self.best = np.inf
        else:
            self.monitor_op = np.greater
            self.min_delta *= 1
            self.best = -np.inf
    
    def on_train_begin(self, trainer):
        """重置状态"""
        self.wait = 0
        self.stopped_epoch = 0
        if self.mode == 'min':
            self.best = np.inf
        else:
            self.best = -np.inf
    
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float]) -> bool:
        """检查是否需要早停"""
        current = metrics.get(self.monitor)
        
        if current is None:
            if self.verbose:
                print(f"Warning: EarlyStopping monitor '{self.monitor}' not found in metrics")
            return False
        
        # 检查是否有改进
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = {
                    k: v.cpu().clone() for k, v in trainer.model.state_dict().items()
                }
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                if self.verbose:
                    print(f"\nEarly stopping triggered!")
                    print(f"Restoring model weights from epoch {epoch - self.patience + 1}")
                
                if self.restore_best_weights and self.best_weights is not None:
                    trainer.model.load_state_dict(self.best_weights)
                
                return True
        
        return False
    
    def on_train_end(self, trainer):
        """训练结束时打印信息"""
        if self.stopped_epoch > 0 and self.verbose:
            print(f"Early stopping at epoch {self.stopped_epoch + 1}")


class ModelCheckpoint(Callback):
    """模型检查点回调"""
    
    def __init__(
        self,
        filepath: str,
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_best_only: bool = True,
        save_freq: int = 1,
        verbose: bool = True
    ):
        """
        Args:
            filepath: 保存路径（可包含格式化字符串，如 'model_{epoch:02d}_{val_loss:.4f}.pth'）
            monitor: 监控的指标
            mode: 'min' 或 'max'
            save_best_only: 是否只保存最佳模型
            save_freq: 保存频率（每隔多少个epoch保存一次）
            verbose: 是否打印信息
        """
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_freq = save_freq
        self.verbose = verbose
        
        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.inf
        else:
            self.monitor_op = np.greater
            self.best = -np.inf
        
        # 创建目录
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float]) -> bool:
        """保存检查点"""
        # 检查是否到了保存频率
        if (epoch + 1) % self.save_freq != 0:
            return False
        
        current = metrics.get(self.monitor)
        
        if current is None:
            if self.verbose:
                print(f"Warning: ModelCheckpoint monitor '{self.monitor}' not found in metrics")
            current = 0.0
        
        # 决定是否保存
        should_save = False
        
        if self.save_best_only:
            if self.monitor_op(current, self.best):
                should_save = True
                self.best = current
        else:
            should_save = True
        
        if should_save:
            # 格式化文件名
            filepath = self.filepath.format(epoch=epoch, **metrics)
            
            # 保存检查点
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': trainer.model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'metrics': metrics,
                'best_metric': self.best
            }
            
            if hasattr(trainer, 'scheduler') and trainer.scheduler is not None:
                checkpoint['scheduler_state_dict'] = trainer.scheduler.state_dict()
            
            torch.save(checkpoint, filepath)
            
            if self.verbose:
                print(f"\nCheckpoint saved: {filepath}")
                print(f"  {self.monitor}: {current:.4f}")
        
        return False


class LearningRateMonitor(Callback):
    """学习率监控回调"""
    
    def __init__(self, verbose: bool = True):
        """
        Args:
            verbose: 是否打印学习率变化
        """
        self.verbose = verbose
        self.lr_history = []
    
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float]) -> bool:
        """记录学习率"""
        lr = trainer.optimizer.param_groups[0]['lr']
        self.lr_history.append(lr)
        
        if self.verbose and epoch % 10 == 0:
            print(f"Learning Rate at epoch {epoch + 1}: {lr:.6e}")
        
        return False


class MetricLogger(Callback):
    """指标记录回调"""
    
    def __init__(
        self,
        log_dir: str = './logs',
        filename: str = 'metrics.json'
    ):
        """
        Args:
            log_dir: 日志目录
            filename: 日志文件名
        """
        self.log_dir = log_dir
        self.filename = filename
        self.metrics_history = []
        
        os.makedirs(log_dir, exist_ok=True)
    
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float]) -> bool:
        """记录指标"""
        metrics_with_epoch = {'epoch': epoch + 1, **metrics}
        self.metrics_history.append(metrics_with_epoch)
        
        # 保存到文件
        filepath = os.path.join(self.log_dir, self.filename)
        with open(filepath, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        return False


class ProgressBar(Callback):
    """进度条回调"""
    
    def __init__(self, show_epoch_progress: bool = True):
        """
        Args:
            show_epoch_progress: 是否显示epoch进度
        """
        self.show_epoch_progress = show_epoch_progress
    
    def on_epoch_begin(self, trainer, epoch: int):
        """显示epoch开始信息"""
        if self.show_epoch_progress:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{trainer.max_epochs}")
            print(f"{'='*60}")


class LearningRateScheduler(Callback):
    """学习率调度器回调（用于ReduceLROnPlateau等需要在epoch结束时更新的调度器）"""
    
    def __init__(
        self,
        scheduler,
        monitor: str = 'val_loss'
    ):
        """
        Args:
            scheduler: 学习率调度器
            monitor: 监控的指标（用于ReduceLROnPlateau）
        """
        self.scheduler = scheduler
        self.monitor = monitor
    
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float]) -> bool:
        """更新学习率"""
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            current = metrics.get(self.monitor)
            if current is not None:
                self.scheduler.step(current)
        else:
            self.scheduler.step()
        
        return False


class GradientClipping(Callback):
    """梯度裁剪回调"""
    
    def __init__(
        self,
        max_norm: float = 1.0,
        norm_type: float = 2.0
    ):
        """
        Args:
            max_norm: 最大梯度范数
            norm_type: 范数类型
        """
        self.max_norm = max_norm
        self.norm_type = norm_type
    
    def on_batch_end(self, trainer, batch: int, metrics: Dict[str, float]):
        """裁剪梯度"""
        torch.nn.utils.clip_grad_norm_(
            trainer.model.parameters(),
            max_norm=self.max_norm,
            norm_type=self.norm_type
        )


class TensorBoardLogger(Callback):
    """TensorBoard日志回调"""
    
    def __init__(self, log_dir: str = './runs'):
        """
        Args:
            log_dir: TensorBoard日志目录
        """
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=log_dir)
            self.enabled = True
        except ImportError:
            print("TensorBoard not available. Install it with: pip install tensorboard")
            self.enabled = False
    
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float]) -> bool:
        """记录指标到TensorBoard"""
        if not self.enabled:
            return False
        
        for metric_name, metric_value in metrics.items():
            if metric_name != 'lr':
                self.writer.add_scalar(metric_name, metric_value, epoch)
        
        # 记录学习率
        if 'lr' in metrics:
            self.writer.add_scalar('learning_rate', metrics['lr'], epoch)
        
        return False
    
    def on_train_end(self, trainer):
        """关闭TensorBoard writer"""
        if self.enabled:
            self.writer.close()


class CallbackList:
    """回调函数列表管理器"""
    
    def __init__(self, callbacks: Optional[List[Callback]] = None):
        """
        Args:
            callbacks: 回调函数列表
        """
        self.callbacks = callbacks or []
    
    def append(self, callback: Callback):
        """添加回调函数"""
        self.callbacks.append(callback)
    
    def on_train_begin(self, trainer):
        """调用所有回调的on_train_begin"""
        for callback in self.callbacks:
            callback.on_train_begin(trainer)
    
    def on_train_end(self, trainer):
        """调用所有回调的on_train_end"""
        for callback in self.callbacks:
            callback.on_train_end(trainer)
    
    def on_epoch_begin(self, trainer, epoch: int):
        """调用所有回调的on_epoch_begin"""
        for callback in self.callbacks:
            callback.on_epoch_begin(trainer, epoch)
    
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float]) -> bool:
        """调用所有回调的on_epoch_end，返回是否停止训练"""
        for callback in self.callbacks:
            if callback.on_epoch_end(trainer, epoch, metrics):
                return True
        return False


if __name__ == '__main__':
    # 测试回调函数
    
    # 创建模拟的trainer
    class MockTrainer:
        def __init__(self):
            self.model = torch.nn.Linear(10, 5)
            self.optimizer = torch.optim.Adam(self.model.parameters())
            self.max_epochs = 20
    
    trainer = MockTrainer()
    
    # 测试早停
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        mode='min',
        verbose=True
    )
    
    print("Testing Early Stopping:")
    early_stopping.on_train_begin(trainer)
    
    # 模拟训练过程
    val_losses = [1.0, 0.9, 0.8, 0.85, 0.84, 0.83, 0.82, 0.81]
    for epoch, val_loss in enumerate(val_losses):
        metrics = {'val_loss': val_loss}
        should_stop = early_stopping.on_epoch_end(trainer, epoch, metrics)
        print(f"Epoch {epoch}, val_loss: {val_loss:.3f}, should_stop: {should_stop}")
        if should_stop:
            break
    
    # 测试模型检查点
    print("\n\nTesting Model Checkpoint:")
    checkpoint_callback = ModelCheckpoint(
        filepath='./test_checkpoints/model_{epoch:02d}_{val_loss:.4f}.pth',
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=True
    )
    
    for epoch, val_loss in enumerate(val_losses[:4]):
        metrics = {'val_loss': val_loss}
        checkpoint_callback.on_epoch_end(trainer, epoch, metrics)
