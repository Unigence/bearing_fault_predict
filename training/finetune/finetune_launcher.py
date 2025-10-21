"""
微调启动脚本
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
        use_progressive = True
    else:
        # 恒定增强
        intensity = aug_config.get_default_intensity()
        train_augmentation = get_augmentation_pipeline(
            stage='supervised',
            epoch=int(max_epochs * 0.5),  # 使用中等epoch对应的强度
            max_epochs=max_epochs,
            mode='train'
        )
        use_progressive = False

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


class ProgressiveAugmentationTrainer(SupervisedTrainer):
    """
    继承自SupervisedTrainer,添加每个epoch更新augmentation的功能
    """

    def __init__(
        self,
        *args,
        use_progressive_aug: bool = False,
        max_epochs: int = 100,
        **kwargs
    ):
        """
        Args:
            use_progressive_aug: 是否使用渐进式增强
            max_epochs: 最大epoch数
        """
        super().__init__(*args, **kwargs)
        self.use_progressive_aug = use_progressive_aug
        self.max_epochs = max_epochs

        if self.use_progressive_aug:
            print(f"  ✅ 启用渐进式增强,将在每个epoch更新augmentation强度")

    def update_augmentation(self, epoch: int):
        """
        更新训练集的augmentation

        Args:
            epoch: 当前epoch
        """
        if not self.use_progressive_aug:
            return

        # 创建新的augmentation pipeline
        new_augmentation = get_augmentation_pipeline(
            stage='supervised',
            epoch=epoch,
            max_epochs=self.max_epochs,
            mode='train'
        )

        # 更新dataset的augmentation
        self.train_loader.dataset.augmentation = new_augmentation

        # 打印当前增强强度
        progress = epoch / self.max_epochs
        if progress < 0.3:
            intensity = "弱"
        elif progress < 0.7:
            intensity = "中"
        else:
            intensity = "强"

        print(f"  📊 Epoch {epoch+1}: 更新增强强度 -> {intensity} (progress={progress:.2f})")

    def train(
        self,
        epochs: int,
        log_interval: int = 10,
        save_config: Optional[dict] = None
    ):
        """
        重写train方法,在每个epoch开始前更新augmentation

        Args:
            epochs: 训练轮数
            log_interval: 日志打印间隔
            save_config: 保存配置
        """
        print("=" * 80)
        print(f"开始训练: {epochs} epochs")
        if self.use_progressive_aug:
            print("  ✅ 渐进式增强已启用")
        print("=" * 80)

        # 保存配置
        if save_config:
            self._save_config(save_config)

        # 训练循环
        for epoch in range(epochs):
            self.current_epoch = epoch

            # 每个epoch开始前更新augmentation
            self.update_augmentation(epoch)

            # 训练一个epoch
            train_metrics = self.train_epoch()

            # 验证一个epoch
            val_metrics = self.validate_epoch()

            # 更新学习率
            self._update_lr(val_metrics)

            # 更新渐进式损失的epoch
            # self.update_epoch(epoch, epochs)

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
            import time
            epoch_time = 0  # 可以在实际实现中计时
            self._print_epoch_log(epoch, epochs, train_metrics, val_metrics, epoch_time)

            # 检查早停
            if self.callbacks.should_stop():
                print(f"\n早停触发,在第 {epoch+1} 轮停止训练")
                break

        # 绘制训练曲线
        self._plot_curves()

        # 训练结束
        print("\n" + "=" * 80)
        print("训练完成!")
        print("=" * 80)


def launch_finetune(
    model_config: ModelConfigParser,
    train_config: TrainConfigParser,
    aug_config: AugmentationConfigParser,
    pretrained_weights: Optional[str] = None,
    experiment_name: str = None
):
    """
    启动微调

    Args:
        model_config: 模型配置解析器
        train_config: 训练配置解析器
        aug_config: 增强配置解析器
        pretrained_weights: 预训练权重路径
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

    # 加载预训练权重
    if pretrained_weights and os.path.exists(pretrained_weights):
        print(f"\n加载预训练权重: {pretrained_weights}")
        model.load_pretrained_backbone(pretrained_weights)

    # 打印模型信息
    param_dict = model.count_parameters()
    print(f"✓ 模型创建成功")
    print(f"  - 配置: {model_params['config']}")
    print(f"  - 总参数: {param_dict['total']:,}")
    print(f"  - 可训练参数: {param_dict['trainable']:,}")

    # 获取微调配置
    finetune_params = train_config.get_finetune_params()
    data_params = train_config.get_data_params()

    # 创建数据加载器时获取是否使用渐进式增强
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
    if use_progressive:
        print(f"  ✅ 使用渐进式增强")

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

    # 使用新的ProgressiveAugmentationTrainer
    print("\n创建训练器...")
    trainer = ProgressiveAugmentationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_config=finetune_params['loss'],
        device=device,
        experiment_dir=str(experiment_dir),
        use_amp=train_config.use_amp(),
        mixup_config=train_config.get('mixup', None),
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


if __name__ == '__main__':
    """独立运行微调"""
    print("=" * 70)
    print("微调启动脚本测试（已修复版本）")
    print("=" * 70)
