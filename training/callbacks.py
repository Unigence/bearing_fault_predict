"""
训练回调函数
实现早停、模型保存等功能
"""
import torch
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np


class Callback:
    """回调基类"""
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], model: torch.nn.Module, optimizer: torch.optim.Optimizer):
        """Epoch结束时调用"""
        pass
    
    def should_stop(self) -> bool:
        """是否应该停止训练"""
        return False


class EarlyStopping(Callback):
    """早停回调"""
    
    def __init__(
        self,
        patience: int = 10,
        monitor: str = 'val_loss',
        mode: str = 'min',
        min_delta: float = 0.0,
        restore_best_weights: bool = True
    ):
        """
        Args:
            patience: 耐心值,多少个epoch没有改善就停止
            monitor: 监控的指标名称
            mode: 'min' 或 'max'
            min_delta: 最小改善量
            restore_best_weights: 是否恢复最佳权重
        """
        self.patience = patience
        self.monitor = monitor
        self.mode = mode
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.counter = 0
        self.best_value = None
        self.best_weights = None
        self.stop_training = False
        
        # 比较函数
        if mode == 'min':
            self.is_better = lambda current, best: current < best - min_delta
        else:
            self.is_better = lambda current, best: current > best + min_delta
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], model: torch.nn.Module, optimizer: torch.optim.Optimizer):
        """Epoch结束时检查早停条件"""
        current_value = metrics.get(self.monitor)
        
        if current_value is None:
            return
        
        if self.best_value is None:
            # 第一个epoch
            self.best_value = current_value
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        elif self.is_better(current_value, self.best_value):
            # 有改善
            self.best_value = current_value
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            # 没有改善
            self.counter += 1
            if self.counter >= self.patience:
                self.stop_training = True
                print(f"\n早停触发: {self.monitor}在{self.patience}个epoch内没有改善")
                
                # 恢复最佳权重
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                    print(f"已恢复最佳模型(Epoch {epoch - self.patience})")
    
    def should_stop(self) -> bool:
        """是否应该停止训练"""
        return self.stop_training


class ModelCheckpoint(Callback):
    """模型保存回调"""
    
    def __init__(
        self,
        save_dir: str,
        monitor: str = 'val_acc',
        mode: str = 'max',
        save_best_only: bool = False,
        save_frequency: int = 5,
        keep_last_n: int = 3,
        verbose: bool = True
    ):
        """
        Args:
            save_dir: 保存目录
            monitor: 监控的指标
            mode: 'min' 或 'max'
            save_best_only: 是否只保存最佳模型
            save_frequency: 保存频率(每N个epoch)
            keep_last_n: 保留最近N个checkpoint
            verbose: 是否打印信息
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_frequency = save_frequency
        self.keep_last_n = keep_last_n
        self.verbose = verbose
        
        self.best_value = None
        self.checkpoints = []  # 保存的checkpoint列表
        
        # 比较函数
        if mode == 'min':
            self.is_better = lambda current, best: current < best
        else:
            self.is_better = lambda current, best: current > best
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], model: torch.nn.Module, optimizer: torch.optim.Optimizer):
        """Epoch结束时保存模型"""
        current_value = metrics.get(self.monitor)
        
        if current_value is None:
            return
        
        # 判断是否需要保存
        should_save = False
        is_best = False
        
        if self.best_value is None:
            # 第一个epoch,总是保存
            should_save = True
            is_best = True
            self.best_value = current_value
        elif self.is_better(current_value, self.best_value):
            # 发现更好的模型
            should_save = True
            is_best = True
            self.best_value = current_value
        elif not self.save_best_only and (epoch + 1) % self.save_frequency == 0:
            # 定期保存
            should_save = True
        
        if should_save:
            # 构建文件名
            if is_best:
                filename = 'best_model.pth'
            else:
                filename = f'checkpoint_epoch_{epoch+1}.pth'
            
            filepath = self.save_dir / filename
            
            # 保存checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
                self.monitor: current_value
            }
            
            torch.save(checkpoint, filepath)
            
            if self.verbose:
                status = "best" if is_best else "periodic"
                print(f"\n模型已保存 ({status}): {filepath}")
            
            # 记录checkpoint
            if not is_best:
                self.checkpoints.append((epoch, filepath))
                self._cleanup_old_checkpoints()
    
    def _cleanup_old_checkpoints(self):
        """清理旧的checkpoint"""
        if len(self.checkpoints) > self.keep_last_n:
            # 删除最旧的checkpoint
            num_to_delete = len(self.checkpoints) - self.keep_last_n
            for i in range(num_to_delete):
                epoch, filepath = self.checkpoints[i]
                if filepath.exists():
                    filepath.unlink()
                    if self.verbose:
                        print(f"删除旧checkpoint: {filepath}")
            
            # 更新列表
            self.checkpoints = self.checkpoints[num_to_delete:]


class CallbackList:
    """回调列表管理器"""
    
    def __init__(self, callbacks: Optional[List[Callback]] = None):
        """
        Args:
            callbacks: 回调列表
        """
        self.callbacks = callbacks or []
    
    def add(self, callback: Callback):
        """添加回调"""
        self.callbacks.append(callback)
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], model: torch.nn.Module, optimizer: torch.optim.Optimizer):
        """Epoch结束时调用所有回调"""
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, metrics, model, optimizer)
    
    def should_stop(self) -> bool:
        """检查是否应该停止训练"""
        return any(callback.should_stop() for callback in self.callbacks)


if __name__ == '__main__':
    """测试代码"""
    print("=" * 70)
    print("Callbacks测试")
    print("=" * 70)
    
    # 测试早停
    print("\n1. 测试EarlyStopping")
    early_stopping = EarlyStopping(patience=3, monitor='val_loss', mode='min')
    
    # 模拟训练过程
    test_metrics_list = [
        {'val_loss': 0.5},
        {'val_loss': 0.4},  # 改善
        {'val_loss': 0.45}, # 没改善
        {'val_loss': 0.46}, # 没改善
        {'val_loss': 0.47}, # 没改善,应该触发早停
    ]
    
    dummy_model = torch.nn.Linear(10, 5)
    dummy_optimizer = torch.optim.Adam(dummy_model.parameters())
    
    for epoch, metrics in enumerate(test_metrics_list):
        early_stopping.on_epoch_end(epoch, metrics, dummy_model, dummy_optimizer)
        print(f"   Epoch {epoch}: val_loss={metrics['val_loss']:.2f}, "
              f"counter={early_stopping.counter}, should_stop={early_stopping.should_stop()}")
        
        if early_stopping.should_stop():
            print(f"   早停触发!")
            break
    
    print("\n" + "=" * 70)
    print("✓ Callbacks测试完成")
    print("=" * 70)
