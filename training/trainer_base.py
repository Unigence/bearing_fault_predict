"""
训练器基类
定义通用的训练逻辑和接口
"""
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
from pathlib import Path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.checkpoint import CheckpointManager
from utils.metrics import MetricsTracker
from utils.visualization import TrainingVisualizer
from training.callbacks import CallbackList, EarlyStopping, ModelCheckpoint


class TrainerBase(ABC):
    """训练器基类"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        device: str = 'cuda',
        experiment_dir: str = 'experiments/runs',
        use_amp: bool = False
    ):
        """
        Args:
            model: 模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            optimizer: 优化器
            scheduler: 学习率调度器
            device: 训练设备
            experiment_dir: 实验保存目录
            use_amp: 是否使用混合精度训练
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.use_amp = use_amp
        
        # 移动模型到设备
        self.model.to(self.device)
        
        # 创建实验目录
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化工具
        self.checkpoint_manager = CheckpointManager(self.experiment_dir / 'checkpoints')
        self.metrics_tracker = MetricsTracker()
        self.visualizer = TrainingVisualizer(self.experiment_dir / 'visualizations')
        
        # 初始化AMP
        self.scaler = torch.cuda.amp.GradScaler() if use_amp else None
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = None
        
        # Callbacks
        self.callbacks = CallbackList()
    
    def add_callback(self, callback):
        """添加callback"""
        self.callbacks.add(callback)
    
    def setup_callbacks(self, early_stopping_config: Dict, checkpoint_config: Dict):
        """
        设置callbacks
        
        Args:
            early_stopping_config: 早停配置
            checkpoint_config: checkpoint配置
        """
        # 早停
        if early_stopping_config.get('enable', True):
            early_stopping = EarlyStopping(
                patience=early_stopping_config.get('patience', 10),
                monitor=early_stopping_config.get('monitor', 'val_loss'),
                mode=early_stopping_config.get('mode', 'min'),
                restore_best_weights=early_stopping_config.get('restore_best_weights', True)
            )
            self.add_callback(early_stopping)
        
        # 模型保存
        if checkpoint_config.get('save_best', True):
            model_checkpoint = ModelCheckpoint(
                save_dir=self.checkpoint_manager.checkpoint_dir,
                monitor=checkpoint_config.get('monitor_metric', 'val_acc'),
                mode=checkpoint_config.get('monitor_mode', 'max'),
                save_frequency=checkpoint_config.get('save_frequency', 5),
                keep_last_n=checkpoint_config.get('keep_last_n', 3)
            )
            self.add_callback(model_checkpoint)
    
    @abstractmethod
    def train_epoch(self) -> Dict[str, float]:
        """
        训练一个epoch
        
        Returns:
            metrics: 训练指标字典
        """
        pass
    
    @abstractmethod
    def validate_epoch(self) -> Dict[str, float]:
        """
        验证一个epoch
        
        Returns:
            metrics: 验证指标字典
        """
        pass
    
    @abstractmethod
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算损失
        
        Args:
            batch: 数据batch
        
        Returns:
            loss: 总损失
            loss_dict: 损失详情字典
        """
        pass
    
    def train(
        self,
        epochs: int,
        log_interval: int = 10,
        save_config: Optional[Dict] = None
    ):
        """
        完整训练流程
        
        Args:
            epochs: 训练轮数
            log_interval: 日志打印间隔
            save_config: 保存配置
        """
        print("=" * 80)
        print(f"开始训练: {epochs} epochs")
        print("=" * 80)
        
        # 保存配置
        if save_config:
            self._save_config(save_config)
        
        # 训练循环
        for epoch in range(epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # 训练一个epoch
            train_metrics = self.train_epoch()
            
            # 验证一个epoch
            val_metrics = self.validate_epoch()
            
            # 更新学习率
            self._update_lr(val_metrics)
            
            # 记录指标
            epoch_metrics = {**train_metrics, **val_metrics}
            self.metrics_tracker.update(epoch_metrics)
            
            # 回调
            callback_metrics = {
                'epoch': epoch,
                'train_loss': train_metrics.get('train_loss', 0),
                'val_loss': val_metrics.get('val_loss', 0),
                'val_acc': val_metrics.get('val_acc', 0)
            }
            self.callbacks.on_epoch_end(epoch, callback_metrics, self.model, self.optimizer)
            
            # 打印日志
            epoch_time = time.time() - epoch_start_time
            self._print_epoch_log(epoch, epochs, train_metrics, val_metrics, epoch_time)
            
            # 检查早停
            if self.callbacks.should_stop():
                print(f"\n早停触发,在第 {epoch+1} 轮停止训练")
                break
        
        # 训练结束
        print("\n" + "=" * 80)
        print("训练完成!")
        print("=" * 80)
        
        # 绘制训练曲线
        self._plot_training_curves()
        
        # 加载最佳模型
        self._load_best_model()
    
    def _update_lr(self, val_metrics: Dict[str, float]):
        """更新学习率"""
        if self.scheduler is not None:
            # ReduceLROnPlateau需要传入metric
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics.get('val_acc', 0))
            # CombinedScheduler也需要传入metric
            elif hasattr(self.scheduler, 'step') and 'metric' in self.scheduler.step.__code__.co_varnames:
                self.scheduler.step(metric=val_metrics.get('val_acc', 0))
            else:
                self.scheduler.step()
    
    def _print_epoch_log(
        self,
        epoch: int,
        total_epochs: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        epoch_time: float
    ):
        """打印epoch日志"""
        lr = self.optimizer.param_groups[0]['lr']
        
        log_str = f"Epoch [{epoch+1:3d}/{total_epochs}] | "
        log_str += f"Time: {epoch_time:.2f}s | "
        log_str += f"LR: {lr:.6f} | "
        
        # 训练指标
        for key, value in train_metrics.items():
            log_str += f"{key}: {value:.4f} | "
        
        # 验证指标
        for key, value in val_metrics.items():
            log_str += f"{key}: {value:.4f} | "
        
        print(log_str.rstrip(" | "))
    
    def _plot_training_curves(self):
        """绘制训练曲线"""
        history = self.metrics_tracker.get_history()
        
        if 'train_loss' in history and 'val_loss' in history:
            self.visualizer.plot_loss_curves(
                history['train_loss'],
                history['val_loss'],
                save_path=self.experiment_dir / 'visualizations' / 'loss_curve.png'
            )
        
        if 'train_acc' in history and 'val_acc' in history:
            self.visualizer.plot_accuracy_curves(
                history['train_acc'],
                history['val_acc'],
                save_path=self.experiment_dir / 'visualizations' / 'acc_curve.png'
            )
    
    def _load_best_model(self):
        """加载最佳模型"""
        best_checkpoint = self.checkpoint_manager.get_best_checkpoint()
        if best_checkpoint is not None and os.path.exists(best_checkpoint):
            print(f"\n加载最佳模型: {best_checkpoint}")
            checkpoint = torch.load(best_checkpoint, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
    
    def _save_config(self, config: Dict):
        """保存配置"""
        import yaml
        config_path = self.experiment_dir / 'config.yaml'
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    def save_checkpoint(self, filepath: str, **extra_state):
        """
        保存checkpoint
        
        Args:
            filepath: 保存路径
            **extra_state: 额外的状态
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            **extra_state
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str, load_optimizer: bool = True):
        """
        加载checkpoint
        
        Args:
            filepath: checkpoint路径
            load_optimizer: 是否加载优化器状态
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.current_epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_metric = checkpoint.get('best_metric')
        
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"Checkpoint加载成功: {filepath}")
        print(f"  Epoch: {self.current_epoch}, Step: {self.global_step}")
