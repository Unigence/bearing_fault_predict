"""
预训练启动脚本
"""
import torch
import os
import sys
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import create_model
from datasets import ContrastiveDataset
from torch.utils.data import DataLoader
from augmentation import ContrastiveAugmentation
from training.pretrain.contrastive_trainer import ContrastiveTrainer
from training.optimizer_factory import create_optimizer_from_config
from training.scheduler_factory import create_scheduler_from_config
from utils.config_parser import ModelConfigParser, TrainConfigParser, AugmentationConfigParser
from utils.seed import set_seed


def create_contrastive_dataloaders(
    data_config: dict,
    aug_config: dict,
    batch_size: int
):
    """
    创建对比学习数据加载器

    🔧 修复: 训练集使用强增强,验证集使用弱增强

    Args:
        data_config: 数据配置
        aug_config: 增强配置
        batch_size: batch大小

    Returns:
        train_loader, val_loader
    """
    # 🔧 训练集: 创建强对比学习增强
    train_aug = ContrastiveAugmentation(
        strong_aug_prob=aug_config.get('strong_aug_prob', 0.5)
    )

    # 🔧 验证集: 创建弱对比学习增强（用于更稳定的评估）
    val_aug = ContrastiveAugmentation(
        strong_aug_prob=0.0  # 验证集只使用基础增强,不使用强增强
    )

    # 创建训练数据集
    train_dataset = ContrastiveDataset(
        data_dir=data_config.get('train_dir', 'raw_datasets/train'),
        window_size=data_config.get('window_size', 512),
        window_step=data_config.get('window_step', 256),
        mode='train',
        fold=data_config.get('current_fold', 0),
        n_folds=data_config.get('n_folds', 5),
        timefreq_method=data_config.get('timefreq_method', 'stft'),
        augmentation=train_aug,  # 使用强增强
        cache_data=data_config.get('cache_data', True)
    )

    # 🔧 创建验证数据集（使用弱增强）
    val_dataset = ContrastiveDataset(
        data_dir=data_config.get('train_dir', 'raw_datasets/train'),
        window_size=data_config.get('window_size', 512),
        window_step=data_config.get('window_step', 256),
        mode='val',
        fold=data_config.get('current_fold', 0),
        n_folds=data_config.get('n_folds', 5),
        timefreq_method=data_config.get('timefreq_method', 'stft'),
        augmentation=val_aug,  # 🔧 使用弱增强而非强增强
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

    print(f"  ✅ 训练集使用强增强(prob={aug_config.get('strong_aug_prob', 0.5)})")
    print(f"  ✅ 验证集使用弱增强(prob=0.0, 仅基础变换)")

    return train_loader, val_loader


def launch_pretrain(
    model_config: ModelConfigParser,
    train_config: TrainConfigParser,
    aug_config: AugmentationConfigParser,
    experiment_name: str = None
):
    """
    启动预训练 (支持 k-fold 交叉验证)

    Args:
        model_config: 模型配置解析器
        train_config: 训练配置解析器
        aug_config: 增强配置解析器
        experiment_name: 实验名称

    Returns:
        pretrained_model: 预训练好的模型（最后一个 fold 的模型）
        experiment_dir: 实验目录
        pretrained_weights_path: 预训练权重路径（最佳 fold 的权重）
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
    print("对比学习预训练 (K-Fold 交叉验证)")
    print("=" * 80)
    print(f"设备: {device}")
    print(f"随机种子: {seed}")

    # 创建实验目录
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"pretrain_{timestamp}"

    experiment_base = train_config.get('experiment.save_dir', 'experiments/runs')
    experiment_dir = Path(experiment_base) / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    print(f"实验目录: {experiment_dir}")

    # 获取配置
    pretrain_params = train_config.get_pretrain_params()
    data_params = train_config.get_data_params()
    aug_params = aug_config.get_contrastive_aug_params()

    # 获取 k-fold 配置
    n_folds = data_params.get('n_folds', 5)
    print(f"\n使用 {n_folds}-Fold 交叉验证")

    # 存储每个 fold 的结果
    fold_results = []
    best_fold = None
    best_val_loss = float('inf')

    # 循环遍历所有 folds
    for fold in range(n_folds):
        print("\n" + "=" * 80)
        print(f"Fold {fold + 1}/{n_folds}")
        print("=" * 80)

        # 为每个 fold 设置随机种子（保证可复现性）
        set_seed(seed + fold)

        # 创建 fold 特定的实验目录
        fold_dir = experiment_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        # 创建模型
        print(f"\n创建模型 (Fold {fold + 1})...")
        model_params = model_config.get_model_params()
        model = create_model(**model_params, enable_contrastive=True)

        # 打印模型信息（仅第一个 fold）
        if fold == 0:
            param_dict = model.count_parameters()
            print(f"✓ 模型创建成功")
            print(f"  - 配置: {model_params['config']}")
            print(f"  - 总参数: {param_dict['total']:,}")
            print(f"  - 可训练参数: {param_dict['trainable']:,}")
            print(f"  - 投影头参数: {param_dict.get('projection_head', 0):,}")

        # 更新当前 fold
        data_params['current_fold'] = fold

        # 创建数据加载器
        print(f"\n创建数据加载器 (Fold {fold + 1})...")
        train_loader, val_loader = create_contrastive_dataloaders(
            data_config=data_params,
            aug_config=aug_params,
            batch_size=pretrain_params['batch_size']
        )
        print(f"✓ 数据加载器创建成功")
        print(f"  - 训练集: {len(train_loader.dataset)} 样本")
        print(f"  - 验证集: {len(val_loader.dataset)} 样本")
        print(f"  - Batch size: {pretrain_params['batch_size']}")

        # 创建优化器
        optimizer = create_optimizer_from_config(
            model,
            pretrain_params['optimizer']
        )

        # 创建学习率调度器
        scheduler, needs_metric = create_scheduler_from_config(
            optimizer,
            pretrain_params['scheduler'],
            total_epochs=pretrain_params['epochs']
        )

        # 创建训练器
        print(f"\n创建训练器 (Fold {fold + 1})...")
        trainer = ContrastiveTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_type=pretrain_params['loss'].get('type', 'ntxent'),
            temperature=pretrain_params['loss'].get('temperature', 0.07),
            device=device,
            experiment_dir=str(fold_dir),
            use_amp=train_config.use_amp(),
            gradient_clip_max_norm=pretrain_params['gradient_clip'].get('max_norm', 1.0)
        )

        # 设置callbacks
        trainer.setup_callbacks(
            early_stopping_config=pretrain_params.get('early_stopping', {}),
            checkpoint_config=model_config.get_checkpoint_params()
        )

        # 开始训练
        print(f"\n开始训练 Fold {fold + 1}...")
        trainer.train(
            epochs=pretrain_params['epochs'],
            log_interval=train_config.get('experiment.logging.log_interval', 10),
            save_config={
                'model': model_config.to_dict(),
                'train': train_config.to_dict(),
                'augmentation': aug_config.to_dict(),
                'fold': fold
            }
        )

        # 保存当前 fold 的权重
        fold_weights_path = fold_dir / 'pretrained_weights.pth'
        trainer.save_pretrained_weights(str(fold_weights_path))

        # 获取最佳验证损失
        best_metric_value, best_epoch = trainer.metrics_tracker.get_best_metric('val_loss', mode='min')

        # 记录 fold 结果
        fold_results.append({
            'fold': fold,
            'best_val_loss': best_metric_value,
            'best_epoch': best_epoch,
            'weights_path': fold_weights_path
        })

        print(f"\n✓ Fold {fold + 1} 完成")
        print(f"  - 最佳验证损失: {best_metric_value:.4f} (Epoch {best_epoch})")
        print(f"  - 权重路径: {fold_weights_path}")

        # 更新最佳 fold
        if best_metric_value < best_val_loss:
            best_val_loss = best_metric_value
            best_fold = fold
            best_model = model

    # 打印所有 folds 的总结
    print("\n" + "=" * 80)
    print("K-Fold 交叉验证结果总结")
    print("=" * 80)
    for result in fold_results:
        mark = "⭐ " if result['fold'] == best_fold else "  "
        print(f"{mark}Fold {result['fold'] + 1}: val_loss = {result['best_val_loss']:.4f} (epoch {result['best_epoch']})")

    # 计算平均和标准差
    val_losses = [r['best_val_loss'] for r in fold_results]
    import numpy as np
    mean_val_loss = np.mean(val_losses)
    std_val_loss = np.std(val_losses)

    print(f"\n验证损失统计:")
    print(f"  - 平均值: {mean_val_loss:.4f}")
    print(f"  - 标准差: {std_val_loss:.4f}")
    print(f"  - 最佳 Fold: {best_fold + 1} (val_loss = {best_val_loss:.4f})")

    # 复制最佳 fold 的权重到主目录
    pretrained_weights_path = experiment_dir / 'pretrained_weights.pth'
    import shutil
    shutil.copy(fold_results[best_fold]['weights_path'], pretrained_weights_path)

    print("\n" + "=" * 80)
    print("✓ 预训练完成!")
    print("=" * 80)
    print(f"预训练权重 (最佳 Fold): {pretrained_weights_path}")
    print(f"实验目录: {experiment_dir}")
    print(f"各 Fold 结果保存在: {experiment_dir}/fold_*")

    return best_model, experiment_dir, pretrained_weights_path


if __name__ == '__main__':
    """独立运行预训练"""
    print("=" * 70)
    print("预训练启动脚本测试（已修复版本）")
    print("=" * 70)

    print("\n✅ 修复说明:")
    print("  4. 验证集使用弱增强")
    print("     - 训练集: strong_aug_prob=0.5 (强增强)")
    print("     - 验证集: strong_aug_prob=0.0 (弱增强/基础变换)")
    print("     - 原因: 验证集需要更稳定的评估,不应使用强增强")
    print("     - 注意: 对比学习仍需要两个视图,但增强强度降低")

    print("\n✓ 预训练启动脚本模块加载成功")
    print("=" * 70)
