"""
微调启动脚本
负责初始化和启动有监督微调
"""
import torch
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import create_model
from datasets import BearingDataset
from torch.utils.data import DataLoader
from augmentation import get_augmentation_pipeline
from training.finetune.supervised_trainer import SupervisedTrainer
from training.optimizer_factory import create_optimizer_from_config
from training.scheduler_factory import create_scheduler_from_config
from utils.config_parser import ModelConfigParser, TrainConfigParser, AugmentationConfigParser
from utils.seed import set_seed


def create_supervised_dataloaders(
    data_config: dict,
    aug_config: AugmentationConfigParser,
    batch_size: int,
    max_epochs: int
):
    """
    创建有监督数据加载器
    
    Args:
        data_config: 数据配置
        aug_config: 增强配置解析器
        batch_size: batch大小
        max_epochs: 最大训练轮数(用于渐进式增强)
    
    Returns:
        train_loader, val_loader
    """
    # 创建训练增强
    # 使用渐进式增强或恒定增强
    if aug_config.use_progressive() and not aug_config.use_constant():
        # 渐进式增强(epoch=0时使用weak)
        train_augmentation = get_augmentation_pipeline(
            stage='supervised',
            epoch=0,
            max_epochs=max_epochs,
            mode='train'
        )
    else:
        # 恒定增强
        intensity = aug_config.get_default_intensity()
        train_augmentation = get_augmentation_pipeline(
            stage='supervised',
            epoch=int(max_epochs * 0.5),  # 使用中等epoch对应的强度
            max_epochs=max_epochs,
            mode='train'
        )
    
    # 创建数据集
    train_dataset = BearingDataset(
        data_dir=data_config.get('train_dir', 'raw_datasets/train'),
        mode='train',
        fold=data_config.get('current_fold', 0),
        n_folds=data_config.get('n_folds', 5),
        window_size=data_config.get('window_size', 512),
        window_step=data_config.get('window_step', 256),
        timefreq_method=data_config.get('timefreq_method', 'stft'),
        augmentation=train_augmentation,
        cache_data=data_config.get('cache_data', True)
    )

    val_dataset = BearingDataset(
        data_dir=data_config.get('train_dir', 'raw_datasets/train'),
        mode='val',
        fold=data_config.get('current_fold', 0),
        n_folds=data_config.get('n_folds', 5),
        window_size=data_config.get('window_size', 512),
        window_step=data_config.get('window_step', 256),
        timefreq_method=data_config.get('timefreq_method', 'stft'),
        augmentation=None,  # 验证集不增强
        cache_data=data_config.get('cache_data', True)
    )

    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=data_config.get('num_workers', 4),
        pin_memory=data_config.get('pin_memory', True)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_config.get('num_workers', 4),
        pin_memory=data_config.get('pin_memory', True)
    )

    return train_loader, val_loader


def launch_finetune(
    model_config: ModelConfigParser,
    train_config: TrainConfigParser,
    aug_config: AugmentationConfigParser,
    pretrained_weights_path: Optional[str] = None,
    experiment_name: str = None
):
    """
    启动有监督微调

    Args:
        model_config: 模型配置解析器
        train_config: 训练配置解析器
        aug_config: 增强配置解析器
        pretrained_weights_path: 预训练权重路径(可选)
        experiment_name: 实验名称

    Returns:
        model: 训练好的模型
        experiment_dir: 实验目录
        final_model_path: 最终模型路径
    """
    # 设置随机种子
    seed = train_config.get_seed()
    set_seed(seed)

    # 获取设备
    device = train_config.get_device()
    if not torch.cuda.is_available() and device == 'cuda':
        print("⚠️  CUDA不可用,使用CPU训练")
        device = 'cpu'

    print("=" * 80)
    print("有监督微调")
    print("=" * 80)
    print(f"设备: {device}")
    print(f"随机种子: {seed}")

    # 创建实验目录
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"finetune_{timestamp}"

    experiment_base = train_config.get('experiment.save_dir', 'experiments/runs')
    experiment_dir = Path(experiment_base) / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    print(f"实验目录: {experiment_dir}")

    # 创建模型
    print("\n创建模型...")
    model_params = model_config.get_model_params()
    model = create_model(**model_params, enable_contrastive=False)  # 🔧 微调阶段不需要投影头

    # 加载预训练权重(如果提供)
    if pretrained_weights_path is not None and os.path.exists(pretrained_weights_path):
        print(f"\n加载预训练权重: {pretrained_weights_path}")
        model.load_pretrained_backbone(pretrained_weights_path)
        print("✓ 预训练权重加载成功")

    # 打印模型信息
    param_dict = model.count_parameters()
    print(f"✓ 模型创建成功")
    print(f"  - 配置: {model_params['config']}")
    print(f"  - 总参数: {param_dict['total']:,}")
    print(f"  - 可训练参数: {param_dict['trainable']:,}")

    # 获取微调配置
    finetune_params = train_config.get_finetune_params()
    data_params = train_config.get_data_params()

    # 创建数据加载器
    print("\n创建数据加载器...")
    train_loader, val_loader = create_supervised_dataloaders(
        data_config=data_params,
        aug_config=aug_config,
        batch_size=finetune_params['batch_size'],
        max_epochs=finetune_params['epochs']
    )
    print(f"✓ 数据加载器创建成功")
    print(f"  - 训练集: {len(train_loader.dataset)} 样本")
    print(f"  - 验证集: {len(val_loader.dataset)} 样本")
    print(f"  - Batch size: {finetune_params['batch_size']}")

    # 创建优化器
    print("\n创建优化器...")
    optimizer = create_optimizer_from_config(
        model,
        finetune_params['optimizer']
    )
    print(f"✓ 优化器创建成功: {type(optimizer).__name__}")

    # 创建学习率调度器
    print("\n创建学习率调度器...")
    scheduler, needs_metric = create_scheduler_from_config(
        optimizer,
        finetune_params['scheduler'],
        total_epochs=finetune_params['epochs'],
        steps_per_epoch=len(train_loader)
    )
    print(f"✓ 调度器创建成功: {type(scheduler).__name__}")
    if needs_metric:
        print(f"  ⚠️  此调度器需要metric,trainer将自动传入验证指标")

    # 检查是否使用Mixup
    mixup_config = None
    if train_config.get('training_mode.use_mixup', False):
        mixup_params = train_config.get('mixup', {})
        mixup_config = {
            'alpha': mixup_params.get('alpha', 0.2),
            'prob': mixup_params.get('prob', 0.5),
            'time_domain': mixup_params.get('time_domain', {}),
            'frequency_domain': mixup_params.get('frequency_domain', {}),
            'feature_level': mixup_params.get('feature_level', {})
        }

        print("✓ Mixup配置:")
        print(f"  - Alpha: {mixup_config['alpha']}")
        print(f"  - 应用概率: {mixup_config['prob']}")
    else:
        print("✓ Mixup未启用")

    # 创建训练器
    print("\n创建训练器...")

    # 准备损失配置
    loss_config = finetune_params['loss'].copy()
    loss_config['focal']['num_classes'] = model_params['num_classes']

    trainer = SupervisedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_config=loss_config,
        device=device,
        experiment_dir=str(experiment_dir),
        use_amp=train_config.use_amp(),
        mixup_config=mixup_config,  # 传入mixup配置
        gradient_clip_max_norm=finetune_params['gradient_clip'].get('max_norm', 1.0)
    )

    # 设置callbacks
    trainer.setup_callbacks(
        early_stopping_config=finetune_params.get('early_stopping', {}),
        checkpoint_config=model_config.get_checkpoint_params()
    )

    # 开始训练
    print("\n" + "=" * 80)
    print("开始微调")
    print("=" * 80)

    trainer.train(
        epochs=finetune_params['epochs'],
        log_interval=train_config.get('experiment.logging.log_interval', 10),
        save_config={
            'model': model_config.to_dict(),
            'train': train_config.to_dict(),
            'augmentation': aug_config.to_dict()
        }
    )

    # 保存最终模型
    final_model_path = experiment_dir / 'final_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model_params
    }, final_model_path)

    print("\n" + "=" * 80)
    print("✓ 微调完成!")
    print("=" * 80)
    print(f"最终模型: {final_model_path}")
    print(f"实验目录: {experiment_dir}")

    return model, experiment_dir, final_model_path


if __name__ == '__main__':
    """独立运行微调"""
    # 加载配置
    model_config = ModelConfigParser('configs/model_config.yaml')
    train_config = TrainConfigParser('configs/train_config.yaml')
    aug_config = AugmentationConfigParser('configs/augmentation_config.yaml')

    # 启动微调(不使用预训练)
    model, exp_dir, model_path = launch_finetune(
        model_config,
        train_config,
        aug_config,
        pretrained_weights_path=None
    )
