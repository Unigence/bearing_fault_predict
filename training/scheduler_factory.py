"""
Learning Rate Scheduler Factory
支持多种学习率调度策略的统一创建接口
包括: warmup, combined, 以及所有标准PyTorch调度器
"""

import torch.optim as optim
from torch.optim.lr_scheduler import (
    StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR,
    ReduceLROnPlateau, CyclicLR, OneCycleLR, CosineAnnealingWarmRestarts,
    LambdaLR, _LRScheduler
)
from typing import Dict, Any, Optional, Callable, Union
import math


class WarmupScheduler(_LRScheduler):
    """
    学习率预热调度器
    在warmup_epochs期间线性增加学习率，之后切换到base_scheduler
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs: int,
        warmup_start_lr: float = 0.0,
        base_scheduler: Optional[_LRScheduler] = None,
        last_epoch: int = -1
    ):
        """
        Args:
            optimizer: 优化器
            warmup_epochs: 预热轮数
            warmup_start_lr: 预热起始学习率
            base_scheduler: 预热后使用的调度器
            last_epoch: 上一个epoch
        """
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.base_scheduler = base_scheduler
        super(WarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # 线性预热: 从warmup_start_lr到base_lr
            alpha = self.last_epoch / self.warmup_epochs
            return [
                self.warmup_start_lr + (base_lr - self.warmup_start_lr) * alpha
                for base_lr in self.base_lrs
            ]
        else:
            # 使用base_scheduler或保持不变
            if self.base_scheduler is not None:
                return self.base_scheduler.get_last_lr()
            return self.base_lrs

    def step(self, epoch=None, metrics=None):
        """
        更新学习率

        Args:
            epoch: 当前epoch
            metrics: 用于ReduceLROnPlateau的metric值
        """
        if self.last_epoch >= self.warmup_epochs and self.base_scheduler is not None:
            # Warmup结束后,使用base_scheduler
            if isinstance(self.base_scheduler, ReduceLROnPlateau):
                if metrics is not None:
                    self.base_scheduler.step(metrics)
            else:
                self.base_scheduler.step(epoch)

        super(WarmupScheduler, self).step(epoch)


class CombinedScheduler(_LRScheduler):
    """
    组合调度器
    在指定的epoch切换不同的调度策略
    例如: 前10个epoch使用warmup，之后切换到cosine
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        schedulers: list,
        switch_epochs: list,
        last_epoch: int = -1
    ):
        """
        Args:
            optimizer: 优化器
            schedulers: 调度器列表
            switch_epochs: 切换epoch列表 (必须递增)
            last_epoch: 上一个epoch

        Example:
            scheduler = CombinedScheduler(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                switch_epochs=[5]  # 在epoch 5切换
            )
        """
        if len(schedulers) != len(switch_epochs) + 1:
            raise ValueError("schedulers数量必须比switch_epochs多1")

        self.schedulers = schedulers
        self.switch_epochs = switch_epochs
        self.current_scheduler_idx = 0
        super(CombinedScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return self.schedulers[self.current_scheduler_idx].get_last_lr()

    def step(self, epoch=None, metrics=None):
        """更新调度器"""
        # 检查是否需要切换调度器
        if self.last_epoch + 1 in self.switch_epochs:
            self.current_scheduler_idx = self.switch_epochs.index(self.last_epoch + 1) + 1
            print(f"[CombinedScheduler] 切换到调度器 {self.current_scheduler_idx}: "
                  f"{type(self.schedulers[self.current_scheduler_idx]).__name__}")

        # 更新当前调度器
        current_scheduler = self.schedulers[self.current_scheduler_idx]
        if isinstance(current_scheduler, ReduceLROnPlateau):
            if metrics is not None:
                current_scheduler.step(metrics)
        else:
            current_scheduler.step(epoch)

        super(CombinedScheduler, self).step(epoch)


class SchedulerFactory:
    """学习率调度器工厂类"""

    SUPPORTED_SCHEDULERS = {
        'step': StepLR,
        'multistep': MultiStepLR,
        'exponential': ExponentialLR,
        'cosine': CosineAnnealingLR,
        'plateau': ReduceLROnPlateau,
        'reduce_on_plateau': ReduceLROnPlateau,  # 别名
        'cyclic': CyclicLR,
        'onecycle': OneCycleLR,
        'cosine_warm_restarts': CosineAnnealingWarmRestarts,
        'warmup': WarmupScheduler,
        'combined': CombinedScheduler,
    }

    @staticmethod
    def create_scheduler(
        optimizer: optim.Optimizer,
        scheduler_type: str = 'step',
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
        T_mult: int = 1,
        # Warmup参数
        warmup_epochs: int = 0,
        warmup_start_lr: float = 0.0,
        base_scheduler_type: Optional[str] = None,
        base_scheduler_params: Optional[Dict[str, Any]] = None,
        # Combined参数
        schedulers_config: Optional[list] = None,
        switch_epochs: Optional[list] = None,
        **kwargs
    ) -> _LRScheduler:
        """
        创建学习率调度器

        Args:
            optimizer: 优化器
            scheduler_type: 调度器类型名称
            其他参数为各个调度器的特定参数

        Returns:
            scheduler: 调度器实例
        """
        scheduler_type = scheduler_type.lower()

        if scheduler_type not in SchedulerFactory.SUPPORTED_SCHEDULERS:
            raise ValueError(
                f"Unsupported scheduler: {scheduler_type}. "
                f"Supported schedulers: {list(SchedulerFactory.SUPPORTED_SCHEDULERS.keys())}"
            )

        scheduler_class = SchedulerFactory.SUPPORTED_SCHEDULERS[scheduler_type]

        # ==================== 基础调度器 ====================
        if scheduler_type == 'step':
            scheduler = scheduler_class(
                optimizer,
                step_size=step_size,
                gamma=gamma,
                **kwargs
            )

        elif scheduler_type == 'multistep':
            if milestones is None:
                milestones = [30, 60, 90]
            scheduler = scheduler_class(
                optimizer,
                milestones=milestones,
                gamma=gamma,
                **kwargs
            )

        elif scheduler_type == 'exponential':
            scheduler = scheduler_class(
                optimizer,
                gamma=gamma,
                **kwargs
            )

        elif scheduler_type == 'cosine':
            scheduler = scheduler_class(
                optimizer,
                T_max=T_max,
                eta_min=eta_min,
                **kwargs
            )

        elif scheduler_type in ['plateau', 'reduce_on_plateau']:
            scheduler = scheduler_class(
                optimizer,
                mode=mode,
                factor=factor,
                patience=patience,
                threshold=threshold,
                **kwargs
            )

        elif scheduler_type == 'cyclic':
            scheduler = scheduler_class(
                optimizer,
                base_lr=base_lr,
                max_lr=max_lr,
                step_size_up=step_size_up,
                **kwargs
            )

        elif scheduler_type == 'onecycle':
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

        elif scheduler_type == 'cosine_warm_restarts':
            scheduler = scheduler_class(
                optimizer,
                T_0=T_0,
                T_mult=T_mult,
                eta_min=eta_min,
                **kwargs
            )

        # ==================== Warmup调度器 ====================
        elif scheduler_type == 'warmup':
            # 创建预热调度器
            base_scheduler = None
            if base_scheduler_type is not None:
                if base_scheduler_params is None:
                    base_scheduler_params = {}
                base_scheduler = SchedulerFactory.create_scheduler(
                    optimizer,
                    base_scheduler_type,
                    **base_scheduler_params
                )

            scheduler = WarmupScheduler(
                optimizer,
                warmup_epochs=warmup_epochs,
                warmup_start_lr=warmup_start_lr,
                base_scheduler=base_scheduler,
                **kwargs
            )

        # ==================== Combined调度器 ====================
        elif scheduler_type == 'combined':
            if schedulers_config is None or switch_epochs is None:
                raise ValueError("Combined scheduler requires schedulers_config and switch_epochs")

            # 创建所有子调度器
            schedulers = []
            for sch_config in schedulers_config:
                sch_type = sch_config.pop('type', sch_config.pop('name', 'step'))
                sub_scheduler = SchedulerFactory.create_scheduler(
                    optimizer,
                    sch_type,
                    **sch_config
                )
                schedulers.append(sub_scheduler)

            scheduler = CombinedScheduler(
                optimizer,
                schedulers=schedulers,
                switch_epochs=switch_epochs,
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
        warmup_start_lr: float = 0.0,
        eta_min: float = 0,
        **kwargs
    ) -> WarmupScheduler:
        """
        便捷方法: 创建带预热的余弦退火调度器

        Args:
            optimizer: 优化器
            warmup_epochs: 预热轮数
            max_epochs: 最大轮数
            warmup_start_lr: 预热起始学习率
            eta_min: 最小学习率

        Returns:
            scheduler: WarmupScheduler实例
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
            warmup_start_lr=warmup_start_lr,
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
    total_epochs: Optional[int] = None,
    steps_per_epoch: Optional[int] = None
) -> Union[_LRScheduler, tuple]:
    """
    便捷函数：从配置字典创建调度器

    统一使用 'type' 字段指定调度器类型
    支持warmup+base_scheduler的组合配置

    Args:
        optimizer: 优化器
        config: 配置字典
        total_epochs: 总训练轮数
        steps_per_epoch: 每个epoch的step数 (用于某些调度器)

    Returns:
        scheduler: 调度器实例
        或 (scheduler, needs_metric): 如果是ReduceLROnPlateau,返回元组标记需要metric

    Example 1 - 基础调度器:
        config = {
            'type': 'cosine',
            'T_max': 100,
            'eta_min': 1e-6
        }

    Example 2 - Warmup + Cosine:
        config = {
            'type': 'cosine_warm_restarts',
            'T_0': 10,
            'T_mult': 1,
            'eta_min': 1e-6,
            'warmup_epochs': 3,
            'warmup_start_lr': 0.0001
        }

    Example 3 - Combined:
        config = {
            'type': 'combined',
            'schedulers_config': [
                {'type': 'warmup', 'warmup_epochs': 5, 'warmup_start_lr': 0.0001},
                {'type': 'cosine', 'T_max': 95, 'eta_min': 1e-6}
            ],
            'switch_epochs': [5]
        }
    """
    # 获取调度器类型 (统一使用'type'字段)
    scheduler_type = config.get('type', config.get('name', 'step'))

    # 复制配置以避免修改原始配置
    scheduler_config = {k: v for k, v in config.items() if k not in ['type', 'name']}

    # 检查是否需要warmup包装 (针对配置中有warmup_epochs但type不是warmup的情况)
    has_warmup = scheduler_config.get('warmup_epochs', 0) > 0

    if has_warmup and scheduler_type != 'warmup' and scheduler_type != 'combined':
        # 需要创建warmup包装的调度器
        warmup_epochs = scheduler_config.pop('warmup_epochs')
        warmup_start_lr = scheduler_config.pop('warmup_start_lr', 0.0)

        # 创建base scheduler
        base_scheduler = SchedulerFactory.create_scheduler(
            optimizer,
            scheduler_type,
            **scheduler_config
        )

        # 用warmup包装
        scheduler = WarmupScheduler(
            optimizer,
            warmup_epochs=warmup_epochs,
            warmup_start_lr=warmup_start_lr,
            base_scheduler=base_scheduler
        )

        print(f"✓ 创建Warmup调度器: warmup_epochs={warmup_epochs}, "
              f"base_scheduler={scheduler_type}")
    else:
        # 直接创建调度器
        # 如果提供了total_epochs，自动设置某些参数
        if total_epochs is not None:
            if scheduler_type == 'cosine' and 'T_max' not in scheduler_config:
                scheduler_config['T_max'] = total_epochs
            elif scheduler_type == 'onecycle' and 'epochs' not in scheduler_config:
                scheduler_config['epochs'] = total_epochs
                if steps_per_epoch is not None:
                    scheduler_config['steps_per_epoch'] = steps_per_epoch

        scheduler = SchedulerFactory.create_scheduler(
            optimizer=optimizer,
            scheduler_type=scheduler_type,
            **scheduler_config
        )

    # 如果是ReduceLROnPlateau，返回标记
    needs_metric = isinstance(scheduler, ReduceLROnPlateau) or \
                   (isinstance(scheduler, WarmupScheduler) and
                    isinstance(scheduler.base_scheduler, ReduceLROnPlateau))

    if needs_metric:
        return scheduler, True
    else:
        return scheduler, False


if __name__ == '__main__':
    # 测试代码
    import torch
    import torch.nn as nn
    from optimizer_factory import OptimizerFactory

    print("=" * 80)
    print("测试学习率调度器工厂")
    print("=" * 80)

    # 创建简单模型和优化器
    model = nn.Linear(10, 5)
    optimizer = OptimizerFactory.create_optimizer(model, 'adam', learning_rate=1e-3)

    # 测试1: 基础调度器
    print("\n[测试1] 基础StepLR调度器")
    scheduler1 = SchedulerFactory.create_scheduler(
        optimizer,
        scheduler_type='step',
        step_size=10,
        gamma=0.1
    )
    print(f"✓ {type(scheduler1).__name__}")

    # 测试2: Warmup + Cosine
    print("\n[测试2] Warmup + Cosine调度器")
    optimizer2 = OptimizerFactory.create_optimizer(model, 'adam', learning_rate=1e-3)
    config2 = {
        'type': 'cosine',
        'T_max': 95,
        'eta_min': 1e-6,
        'warmup_epochs': 5,
        'warmup_start_lr': 1e-5
    }
    scheduler2, needs_metric = create_scheduler_from_config(optimizer2, config2, total_epochs=100)
    print(f"✓ {type(scheduler2).__name__}, needs_metric={needs_metric}")

    # 模拟训练
    print("\n模拟前10个epoch:")
    for epoch in range(10):
        scheduler2.step()
        lr = optimizer2.param_groups[0]['lr']
        print(f"  Epoch {epoch}: lr = {lr:.6f}")

    # 测试3: CosineAnnealingWarmRestarts + Warmup
    print("\n[测试3] CosineAnnealingWarmRestarts + Warmup")
    optimizer3 = OptimizerFactory.create_optimizer(model, 'adam', learning_rate=1e-3)
    config3 = {
        'type': 'cosine_warm_restarts',
        'T_0': 10,
        'T_mult': 1,
        'eta_min': 1e-6,
        'warmup_epochs': 3,
        'warmup_start_lr': 1e-5
    }
    scheduler3, _ = create_scheduler_from_config(optimizer3, config3)
    print(f"✓ {type(scheduler3).__name__}")

    # 测试4: ReduceLROnPlateau
    print("\n[测试4] ReduceLROnPlateau (需要metric)")
    optimizer4 = OptimizerFactory.create_optimizer(model, 'adam', learning_rate=1e-3)
    config4 = {
        'type': 'reduce_on_plateau',
        'mode': 'min',
        'factor': 0.5,
        'patience': 5
    }
    scheduler4, needs_metric = create_scheduler_from_config(optimizer4, config4)
    print(f"✓ {type(scheduler4).__name__}, needs_metric={needs_metric}")

    # 模拟训练
    print("模拟训练 (传入loss作为metric):")
    for epoch in range(5):
        fake_loss = 1.0 - epoch * 0.1  # 模拟loss下降
        scheduler4.step(fake_loss)
        lr = optimizer4.param_groups[0]['lr']
        print(f"  Epoch {epoch}: loss={fake_loss:.2f}, lr={lr:.6f}")

    # 测试5: Combined调度器
    print("\n[测试5] Combined调度器")
    optimizer5 = OptimizerFactory.create_optimizer(model, 'adam', learning_rate=1e-3)
    config5 = {
        'type': 'combined',
        'schedulers_config': [
            {'type': 'warmup', 'warmup_epochs': 5, 'warmup_start_lr': 1e-5},
            {'type': 'cosine', 'T_max': 45, 'eta_min': 1e-6}
        ],
        'switch_epochs': [5]
    }
    scheduler5, _ = create_scheduler_from_config(optimizer5, config5, total_epochs=50)
    print(f"✓ {type(scheduler5).__name__}")

    print("\n模拟训练 (切换点在epoch 5):")
    for epoch in range(10):
        scheduler5.step()
        lr = optimizer5.param_groups[0]['lr']
        print(f"  Epoch {epoch}: lr={lr:.6f}")

    print("\n" + "=" * 80)
    print("✓ 所有测试通过!")
    print("=" * 80)
