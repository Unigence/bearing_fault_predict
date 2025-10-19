"""
Optimizer Factory
支持多种优化器的统一创建接口
"""

import torch.optim as optim
from torch.optim import Optimizer
from typing import Dict, Any, List
import torch.nn as nn


class OptimizerFactory:
    """优化器工厂类"""

    SUPPORTED_OPTIMIZERS = {
        'sgd': optim.SGD,
        'adam': optim.Adam,
        'adamw': optim.AdamW,
        'rmsprop': optim.RMSprop,
        'adagrad': optim.Adagrad,
        'adadelta': optim.Adadelta,
        'adamax': optim.Adamax,
    }

    @staticmethod
    def create_optimizer(
            model: nn.Module,
            optimizer_name: str = 'adam',
            learning_rate: float = 1e-3,
            weight_decay: float = 0.0,
            momentum: float = 0.9,
            betas: tuple = (0.9, 0.999),
            eps: float = 1e-8,
            nesterov: bool = False,
            amsgrad: bool = False,
            **kwargs
    ) -> Optimizer:
        """
        创建优化器

        Args:
            model: 模型
            optimizer_name: 优化器名称
            learning_rate: 学习率
            weight_decay: 权重衰减
            momentum: 动量系数（SGD、RMSprop）
            betas: Adam系列优化器的beta参数
            eps: 数值稳定性参数
            nesterov: 是否使用Nesterov动量（SGD）
            amsgrad: 是否使用AMSGrad（Adam）
            **kwargs: 其他优化器特定参数

        Returns:
            optimizer: 优化器实例
        """
        optimizer_name = optimizer_name.lower()

        if optimizer_name not in OptimizerFactory.SUPPORTED_OPTIMIZERS:
            raise ValueError(
                f"Unsupported optimizer: {optimizer_name}. "
                f"Supported optimizers: {list(OptimizerFactory.SUPPORTED_OPTIMIZERS.keys())}"
            )

        optimizer_class = OptimizerFactory.SUPPORTED_OPTIMIZERS[optimizer_name]

        # 获取模型参数
        params = model.parameters()

        # 根据优化器类型设置参数
        if optimizer_name == 'sgd':
            optimizer = optimizer_class(
                params,
                lr=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay,
                nesterov=nesterov,
                **kwargs
            )

        elif optimizer_name in ['adam', 'adamw']:
            optimizer = optimizer_class(
                params,
                lr=learning_rate,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
                amsgrad=amsgrad,
                **kwargs
            )

        elif optimizer_name == 'adamax':
            optimizer = optimizer_class(
                params,
                lr=learning_rate,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
                **kwargs
            )

        elif optimizer_name == 'rmsprop':
            optimizer = optimizer_class(
                params,
                lr=learning_rate,
                momentum=momentum,
                eps=eps,
                weight_decay=weight_decay,
                **kwargs
            )

        elif optimizer_name in ['adagrad', 'adadelta']:
            optimizer = optimizer_class(
                params,
                lr=learning_rate,
                eps=eps,
                weight_decay=weight_decay,
                **kwargs
            )

        else:
            # 默认配置
            optimizer = optimizer_class(
                params,
                lr=learning_rate,
                weight_decay=weight_decay,
                **kwargs
            )

        return optimizer

    @staticmethod
    def create_optimizer_with_param_groups(
            param_groups: List[Dict[str, Any]],
            optimizer_name: str = 'adam',
            **kwargs
    ) -> Optimizer:
        """
        使用参数组创建优化器（支持不同层使用不同学习率）

        Args:
            param_groups: 参数组列表，每个元素为字典，包含'params'和其他配置
            optimizer_name: 优化器名称
            **kwargs: 其他优化器参数

        Returns:
            optimizer: 优化器实例

        Example:
            param_groups = [
                {'params': model.encoder.parameters(), 'lr': 1e-4},
                {'params': model.classifier.parameters(), 'lr': 1e-3}
            ]
        """
        optimizer_name = optimizer_name.lower()

        if optimizer_name not in OptimizerFactory.SUPPORTED_OPTIMIZERS:
            raise ValueError(
                f"Unsupported optimizer: {optimizer_name}. "
                f"Supported optimizers: {list(OptimizerFactory.SUPPORTED_OPTIMIZERS.keys())}"
            )

        optimizer_class = OptimizerFactory.SUPPORTED_OPTIMIZERS[optimizer_name]
        optimizer = optimizer_class(param_groups, **kwargs)

        return optimizer

    @staticmethod
    def get_optimizer_info(optimizer: Optimizer) -> Dict[str, Any]:
        """
        获取优化器信息

        Args:
            optimizer: 优化器实例

        Returns:
            info: 优化器信息字典
        """
        info = {
            'optimizer_type': type(optimizer).__name__,
            'param_groups': len(optimizer.param_groups),
            'default_lr': optimizer.defaults.get('lr', None),
            'weight_decay': optimizer.defaults.get('weight_decay', None),
        }

        # 获取每个参数组的学习率
        learning_rates = [group['lr'] for group in optimizer.param_groups]
        info['learning_rates'] = learning_rates

        return info


def create_optimizer_from_config(model: nn.Module, config: Dict[str, Any]) -> Optimizer:
    """
    便捷函数：从配置字典创建优化器

    ⚠️ 注意：这是修复后的函数名，与launcher中的导入一致

    Args:
        model: 模型
        config: 配置字典

    Returns:
        optimizer: 优化器实例

    Example:
        config = {
            'name': 'adam',
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'betas': (0.9, 0.999)
        }
    """
    optimizer_name = config.get('name', 'adam')
    optimizer_config = {k: v for k, v in config.items() if k != 'name'}

    return OptimizerFactory.create_optimizer(
        model=model,
        optimizer_name=optimizer_name,
        **optimizer_config
    )


# 保留旧的函数名以保持向后兼容
def create_optimizer(model: nn.Module, config: Dict[str, Any]) -> Optimizer:
    """
    便捷函数：从配置字典创建优化器（向后兼容）

    建议使用 create_optimizer_from_config
    """
    return create_optimizer_from_config(model, config)


if __name__ == '__main__':
    # 测试代码
    import torch
    import torch.nn as nn

    # 创建简单模型
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )

    # 测试1: 创建Adam优化器
    optimizer = OptimizerFactory.create_optimizer(
        model=model,
        optimizer_name='adam',
        learning_rate=1e-3,
        weight_decay=1e-4
    )
    print("Adam Optimizer:", optimizer)
    print("Optimizer Info:", OptimizerFactory.get_optimizer_info(optimizer))

    # 测试2: 创建SGD优化器
    optimizer_sgd = OptimizerFactory.create_optimizer(
        model=model,
        optimizer_name='sgd',
        learning_rate=0.01,
        momentum=0.9,
        nesterov=True
    )
    print("\nSGD Optimizer:", optimizer_sgd)

    # 测试3: 使用配置字典创建
    config = {
        'name': 'adamw',
        'learning_rate': 5e-4,
        'weight_decay': 1e-4
    }
    optimizer_from_config = create_optimizer_from_config(model, config)
    print("\nOptimizer from config:", optimizer_from_config)

    # 测试4: 使用参数组
    param_groups = [
        {'params': model[0].parameters(), 'lr': 1e-4},
        {'params': model[2].parameters(), 'lr': 1e-3}
    ]
    optimizer_groups = OptimizerFactory.create_optimizer_with_param_groups(
        param_groups=param_groups,
        optimizer_name='adam'
    )
    print("\nOptimizer with param groups:", optimizer_groups)
    print("Info:", OptimizerFactory.get_optimizer_info(optimizer_groups))

    print("\n✓ All tests passed!")
