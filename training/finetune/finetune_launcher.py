"""
微调启动脚本 (修复版)
修复内容:
1. 移除ProgressiveAugmentationTrainer类 (功能已整合到SupervisedTrainer)
2. 修复数据增强配置逻辑,支持禁用/恒定/渐进式三种模式
3. 修复预训练权重加载逻辑
4. 简化代码结构
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

    修复内容:
    1. 支持禁用数据增强
    2. 支持恒定增强强度选择
    3. 支持渐进式增强

    Args:
        data_config: 数据配置
        aug_config: 增强配置解析器
        batch_size: batch大小
        max_epochs: 最大训练轮数(用于渐进式增强)

    Returns:
        train_loader, val_loader, use_progressive
    """
    # 🔧 修复: 支持三种数据增强模式
    enable_progressive = aug_config.use_progressive()
    enable_constant = aug_config.use_constant()

    # 判断增强模式
    if not enable_progressive and not enable_constant:
        # 模式1: 禁用数据增强
        train_augmentation = None
        use_progressive = False
        print(f"  数据增强: 禁用")
    elif enable_constant:
        # 模式2: 恒定增强
        intensity = aug_config.get_default_intensity()  # weak/medium/strong
        train_augmentation = get_augmentation_pipeline(
            stage='supervised',
            intensity=intensity,  # 直接指定强度
            mode='train'
        )
        use_progressive = False
        print(f"  数据增强: 恒定增强 (强度={intensity})")
    else:
        # 模式3: 渐进式增强
        train_augmentation = get_augmentation_pipeline(
            stage='supervised',
            epoch=0,  # 初始使用weak
            max_epochs=max_epochs,
            mode='train'
        )
        use_progressive = True
        print(f"  数据增强: 渐进式增强")

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

    # 验证集不使用增强
    val_dataset = BearingDataset(
        data_dir=data_config.get('train_dir', 'raw_datasets/train'),
        mode='val',
        fold=data_config.get('current_fold', 0),
        n_folds=data_config.get('n_folds', 5),
        window_size=data_config.get('window_size', 512),
        window_step=data_config.get('window_step', 512),
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
        pin_memory=data_config.get('pin_memory', True),
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_config.get('num_workers', 4),
        pin_memory=data_config.get('pin_memory', True)
    )

    return train_loader, val_loader, use_progressive


def launch_finetune(
    model_config: ModelConfigParser,
    train_config: TrainConfigParser,
    aug_config: AugmentationConfigParser,
    pretrained_weights: Optional[str] = None,
    experiment_name: str = None
):
    """
    启动微调

    修复内容:
    1. 修复预训练权重加载逻辑 - 支持从yaml配置加载
    2. 简化代码结构

    Args:
        model_config: 模型配置解析器
        train_config: 训练配置解析器
        aug_config: 增强配置解析器
        pretrained_weights: 预训练权重路径 (优先级高于yaml配置)
        experiment_name: 实验名称

    Returns:
        finetuned_model: 微调后的模型
        experiment_dir: 实验目录
        best_weights_path: 最佳权重路径
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
    model = create_model(**model_params, enable_contrastive=False)

    # 🔧 修复: 预训练权重加载逻辑
    # 优先级: 命令行参数 > yaml配置
    if pretrained_weights is None:
        # 尝试从yaml配置加载
        pretrain_config = model_config.get_pretrain_params()
        if pretrain_config.get('use_pretrain', False):
            pretrained_weights = pretrain_config.get('checkpoint_path', None)
            if pretrained_weights:
                print(f"\n✓ 从配置文件读取预训练权重路径: {pretrained_weights}")

    # 加载预训练权重
    if pretrained_weights and os.path.exists(pretrained_weights):
        print(f"\n加载预训练权重: {pretrained_weights}")
        try:
            model.load_pretrained_backbone(pretrained_weights)
            print(f"✓ 预训练权重加载成功")
        except Exception as e:
            print(f"⚠️  预训练权重加载失败: {e}")
            print(f"   继续使用随机初始化权重...")
    elif pretrained_weights:
        print(f"⚠️  预训练权重文件不存在: {pretrained_weights}")
        print(f"   继续使用随机初始化权重...")
    else:
        print(f"\n使用随机初始化权重 (未指定预训练权重)")

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
    train_loader, val_loader, use_progressive = create_supervised_dataloaders(
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
        total_epochs=finetune_params['epochs']
    )
    print(f"✓ 调度器创建成功: {type(scheduler).__name__}")

    # 🔧 修复: 使用修复后的SupervisedTrainer
    print("\n创建训练器...")
    trainer = SupervisedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_config=finetune_params['loss'],
        device=device,
        experiment_dir=str(experiment_dir),
        use_amp=train_config.use_amp(),
        mixup_config=train_config.get('mixup', None) if train_config.use_mixup() else None,
        gradient_clip_max_norm=finetune_params['gradient_clip'].get('max_norm', 1.0),
        use_progressive_aug=use_progressive,
        max_epochs=finetune_params['epochs']
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

    # 保存最佳权重
    best_weights_path = experiment_dir / 'best_model.pth'

    print("\n" + "=" * 80)
    print("✓ 微调完成!")
    print("=" * 80)
    print(f"最佳权重: {best_weights_path}")
    print(f"实验目录: {experiment_dir}")

    return model, experiment_dir, best_weights_path

