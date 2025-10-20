"""
Learning Rate Scheduler Factory
支持多种学习率调度策略的统一创建接口
"""

import torch.optim as optim
from torch.optim.lr_scheduler import (
    StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR,
    ReduceLROnPlateau, CyclicLR, OneCycleLR, CosineAnnealingWarmRestarts,
    LambdaLR, _LRScheduler
)
from typing import Dict, Any, Optional, Callable
import math


class WarmupScheduler(_LRScheduler):
    """
    学习率预热调度器
    在warmup_epochs期间线性增加学习率，之后保持不变或使用其他调度器
    """
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs: int,
        base_scheduler: Optional[_LRScheduler] = None,
        last_epoch: int = -1
    ):
        """
        Args:
            optimizer: 优化器
            warmup_epochs: 预热轮数
            base_scheduler: 预热后使用的调度器
            last_epoch: 上一个epoch
        """
        self.warmup_epochs = warmup_epochs
        self.base_scheduler = base_scheduler
        super(WarmupScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # 线性预热
            alpha = self.last_epoch / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # 使用base_scheduler或保持不变
            if self.base_scheduler is not None:
                return self.base_scheduler.get_last_lr()
            return self.base_lrs
    
    def step(self, epoch=None):
        if self.last_epoch >= self.warmup_epochs and self.base_scheduler is not None:
            self.base_scheduler.step(epoch)
        super(WarmupScheduler, self).step(epoch)


class SchedulerFactory:
    """学习率调度器工厂类"""
    
    SUPPORTED_SCHEDULERS = {
        'step': StepLR,
        'multistep': MultiStepLR,
        'exponential': ExponentialLR,
        'cosine': CosineAnnealingLR,
        'plateau': ReduceLROnPlateau,
        'cyclic': CyclicLR,
        'onecycle': OneCycleLR,
        'cosine_warmup': CosineAnnealingWarmRestarts,
        'warmup': WarmupScheduler,
    }
    
    @staticmethod
    def create_scheduler(
        optimizer: optim.Optimizer,
        scheduler_name: str = 'step',
        # StepLR参数
        step_size: int = 30,
        gamma: float = 0.1,
        # MultiStepLR参数
        milestones: Optional[list] = None,
        # CosineAnnealingLR参数
        T_max: int = 50,
        eta_min: float = 0,
        # ReduceLROnPlateau参数
        mode: str = 'min',
        factor: float = 0.1,
        patience: int = 10,
        threshold: float = 1e-4,
        # CyclicLR参数
        base_lr: float = 1e-4,
        max_lr: float = 1e-2,
        step_size_up: int = 2000,
        # OneCycleLR参数
        max_lr_onecycle: float = 1e-2,
        total_steps: Optional[int] = None,
        epochs: Optional[int] = None,
        steps_per_epoch: Optional[int] = None,
        # CosineAnnealingWarmRestarts参数
        T_0: int = 10,
        T_mult: int = 2,
        # Warmup参数
        warmup_epochs: int = 5,
        base_scheduler_name: Optional[str] = None,
        base_scheduler_params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> _LRScheduler:
        """
        创建学习率调度器
        
        Args:
            optimizer: 优化器
            scheduler_name: 调度器名称
            其他参数为各个调度器的特定参数
            
        Returns:
            scheduler: 调度器实例
        """
        scheduler_name = scheduler_name.lower()
        
        if scheduler_name not in SchedulerFactory.SUPPORTED_SCHEDULERS:
            raise ValueError(
                f"Unsupported scheduler: {scheduler_name}. "
                f"Supported schedulers: {list(SchedulerFactory.SUPPORTED_SCHEDULERS.keys())}"
            )
        
        scheduler_class = SchedulerFactory.SUPPORTED_SCHEDULERS[scheduler_name]
        
        # 根据调度器类型创建实例
        if scheduler_name == 'step':
            scheduler = scheduler_class(
                optimizer,
                step_size=step_size,
                gamma=gamma,
                **kwargs
            )
        
        elif scheduler_name == 'multistep':
            if milestones is None:
                milestones = [30, 60, 90]
            scheduler = scheduler_class(
                optimizer,
                milestones=milestones,
                gamma=gamma,
                **kwargs
            )
        
        elif scheduler_name == 'exponential':
            scheduler = scheduler_class(
                optimizer,
                gamma=gamma,
                **kwargs
            )
        
        elif scheduler_name == 'cosine':
            scheduler = scheduler_class(
                optimizer,
                T_max=T_max,
                eta_min=eta_min,
                **kwargs
            )
        
        elif scheduler_name == 'plateau':
            scheduler = scheduler_class(
                optimizer,
                mode=mode,
                factor=factor,
                patience=patience,
                threshold=threshold,
                **kwargs
            )
        
        elif scheduler_name == 'cyclic':
            scheduler = scheduler_class(
                optimizer,
                base_lr=base_lr,
                max_lr=max_lr,
                step_size_up=step_size_up,
                **kwargs
            )
        
        elif scheduler_name == 'onecycle':
            if total_steps is None and epochs is not None and steps_per_epoch is not None:
                total_steps = epochs * steps_per_epoch
            
            if total_steps is None:
                raise ValueError("OneCycleLR requires total_steps or (epochs and steps_per_epoch)")
            
            scheduler = scheduler_class(
                optimizer,
                max_lr=max_lr_onecycle,
                total_steps=total_steps,
                **kwargs
            )
        
        elif scheduler_name == 'cosine_warmup':
            scheduler = scheduler_class(
                optimizer,
                T_0=T_0,
                T_mult=T_mult,
                eta_min=eta_min,
                **kwargs
            )
        
        elif scheduler_name == 'warmup':
            # 创建预热调度器
            base_scheduler = None
            if base_scheduler_name is not None:
                if base_scheduler_params is None:
                    base_scheduler_params = {}
                base_scheduler = SchedulerFactory.create_scheduler(
                    optimizer,
                    base_scheduler_name,
                    **base_scheduler_params
                )
            
            scheduler = WarmupScheduler(
                optimizer,
                warmup_epochs=warmup_epochs,
                base_scheduler=base_scheduler,
                **kwargs
            )
        
        else:
            scheduler = scheduler_class(optimizer, **kwargs)
        
        return scheduler
    
    @staticmethod
    def create_cosine_with_warmup(
        optimizer: optim.Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        eta_min: float = 0,
        **kwargs
    ) -> WarmupScheduler:
        """
        创建带预热的余弦退火调度器
        
        Args:
            optimizer: 优化器
            warmup_epochs: 预热轮数
            max_epochs: 最大轮数
            eta_min: 最小学习率
            
        Returns:
            scheduler: 调度器实例
        """
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max_epochs - warmup_epochs,
            eta_min=eta_min,
            **kwargs
        )
        
        warmup_scheduler = WarmupScheduler(
            optimizer,
            warmup_epochs=warmup_epochs,
            base_scheduler=cosine_scheduler
        )
        
        return warmup_scheduler
    
    @staticmethod
    def create_polynomial_decay(
        optimizer: optim.Optimizer,
        max_epochs: int,
        power: float = 1.0,
        min_lr: float = 0,
        **kwargs
    ) -> LambdaLR:
        """
        创建多项式衰减调度器
        
        Args:
            optimizer: 优化器
            max_epochs: 最大轮数
            power: 多项式的幂次
            min_lr: 最小学习率
            
        Returns:
            scheduler: 调度器实例
        """
        def polynomial_decay(epoch):
            if epoch >= max_epochs:
                return min_lr
            return (1 - epoch / max_epochs) ** power
        
        scheduler = LambdaLR(optimizer, lr_lambda=polynomial_decay, **kwargs)
        return scheduler
    
    @staticmethod
    def get_scheduler_info(scheduler: _LRScheduler) -> Dict[str, Any]:
        """
        获取调度器信息
        
        Args:
            scheduler: 调度器实例
            
        Returns:
            info: 调度器信息字典
        """
        info = {
            'scheduler_type': type(scheduler).__name__,
            'last_epoch': scheduler.last_epoch,
        }
        
        # 获取当前学习率
        try:
            current_lr = scheduler.get_last_lr()
            info['current_lr'] = current_lr
        except:
            info['current_lr'] = None
        
        return info


def create_scheduler_from_config(
    optimizer: optim.Optimizer,
    config: Dict[str, Any],
    total_epochs: Optional[int] = None
) -> _LRScheduler:
    """
    便捷函数：从配置字典创建调度器

    Args:
        optimizer: 优化器
        config: 配置字典
        total_epochs: 总训练轮数（某些调度器需要，如cosine）

    Returns:
        scheduler: 调度器实例

    Example:
        config = {
            'name': 'cosine',
            'T_max': 100,
            'eta_min': 1e-6
        }
    """
    scheduler_name = config.get('name', 'step')
    scheduler_config = {k: v for k, v in config.items() if k != 'name'}

    # 如果提供了total_epochs且配置中需要T_max，自动设置
    if total_epochs is not None:
        if scheduler_name == 'cosine' and 'T_max' not in scheduler_config:
            scheduler_config['T_max'] = total_epochs
        elif scheduler_name == 'onecycle' and 'epochs' not in scheduler_config:
            scheduler_config['epochs'] = total_epochs

    return SchedulerFactory.create_scheduler(
        optimizer=optimizer,
        scheduler_name=scheduler_name,
        **scheduler_config
    )


if __name__ == '__main__':
    # 测试代码
    import torch
    import torch.nn as nn
    from optimizer_factory import OptimizerFactory

    # 创建简单模型和优化器
    model = nn.Linear(10, 5)
    optimizer = OptimizerFactory.create_optimizer(model, 'adam', learning_rate=1e-3)

    # 测试1: StepLR
    scheduler = SchedulerFactory.create_scheduler(
        optimizer,
        scheduler_name='step',
        step_size=10,
        gamma=0.1
    )
    print("StepLR Scheduler:", scheduler)
    print("Info:", SchedulerFactory.get_scheduler_info(scheduler))

    # 测试2: CosineAnnealingLR
    optimizer2 = OptimizerFactory.create_optimizer(model, 'adam', learning_rate=1e-3)
    scheduler_cosine = SchedulerFactory.create_scheduler(
        optimizer2,
        scheduler_name='cosine',
        T_max=50,
        eta_min=1e-6
    )
    print("\nCosineAnnealingLR:", scheduler_cosine)

    # 测试3: 从配置字典创建
    optimizer3 = OptimizerFactory.create_optimizer(model, 'adam', learning_rate=1e-3)
    config = {
        'name': 'cosine',
        'eta_min': 1e-6
    }
    scheduler_from_config = create_scheduler_from_config(
        optimizer3,
        config,
        total_epochs=100
    )
    print("\nScheduler from config:", scheduler_from_config)
    print("Info:", SchedulerFactory.get_scheduler_info(scheduler_from_config))

    # 测试4: 带预热的余弦退火
    optimizer4 = OptimizerFactory.create_optimizer(model, 'adam', learning_rate=1e-3)
    scheduler_warmup = SchedulerFactory.create_cosine_with_warmup(
        optimizer4,
        warmup_epochs=5,
        max_epochs=100,
        eta_min=1e-6
    )
    print("\nCosine with Warmup:", scheduler_warmup)

    # 模拟训练过程
    print("\n模拟训练过程（前10个epoch）：")
    for epoch in range(10):
        scheduler_warmup.step()
        lr = optimizer4.param_groups[0]['lr']
        print(f"Epoch {epoch}: lr = {lr:.6f}")

    # 测试5: ReduceLROnPlateau
    optimizer5 = OptimizerFactory.create_optimizer(model, 'adam', learning_rate=1e-3)
    scheduler_plateau = SchedulerFactory.create_scheduler(
        optimizer5,
        scheduler_name='plateau',
        mode='min',
        factor=0.5,
        patience=5
    )
    print("\nReduceLROnPlateau:", scheduler_plateau)

    print("\n✓ All tests passed!")
