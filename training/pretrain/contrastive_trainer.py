"""
对比学习训练器
用于预训练阶段的自监督学习
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Dict, Any, Tuple
from tqdm import tqdm
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.trainer_base import TrainerBase
from losses import NTXentLoss, SupConLoss
from training.scheduler_factory import WarmupScheduler


class ContrastiveTrainer(TrainerBase):
    """对比学习训练器"""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        loss_type: str = 'ntxent',
        temperature: float = 0.07,
        device: str = 'cuda',
        experiment_dir: str = 'experiments/runs',
        use_amp: bool = False,
        gradient_clip_max_norm: float = 1.0
    ):
        """
        Args:
            model: 模型
            train_loader: 训练数据加载器(对比学习数据集)
            val_loader: 验证数据加载器
            optimizer: 优化器
            scheduler: 学习率调度器
            loss_type: 损失类型 'ntxent' | 'supcon'
            temperature: 温度参数
            device: 训练设备
            experiment_dir: 实验保存目录
            use_amp: 是否使用混合精度训练
            gradient_clip_max_norm: 梯度裁剪的最大范数
        """
        super().__init__(
            model, train_loader, val_loader, optimizer, scheduler,
            device, experiment_dir, use_amp
        )

        self.loss_type = loss_type
        self.temperature = temperature
        self.gradient_clip_max_norm = gradient_clip_max_norm

        # 创建对比学习损失
        if loss_type == 'ntxent':
            self.criterion = NTXentLoss(temperature=temperature)
        elif loss_type == 'supcon':
            self.criterion = SupConLoss(temperature=temperature)
        else:
            raise ValueError(f"不支持的损失类型: {loss_type}")

        # 检查scheduler是否需要metric
        self.scheduler_needs_metric = self._check_scheduler_needs_metric()

        print(f"ContrastiveTrainer初始化完成")
        print(f"  - 损失类型: {loss_type}")
        print(f"  - 温度参数: {temperature}")
        print(f"  - 梯度裁剪: {gradient_clip_max_norm}")

    def setup_callbacks(self, early_stopping_config: Dict, checkpoint_config: Dict):
        """
        🔧 修复: 重写callbacks设置,确保预训练监控正确的指标

        对于预训练:
        - EarlyStopping 监控 'val_loss' (验证集对比损失)
        - ModelCheckpoint 监控 'val_loss' (而非 val_acc,因为预训练没有准确率)
        """
        from training.callbacks import EarlyStopping, ModelCheckpoint

        # 早停
        if early_stopping_config.get('enable', True):
            # 🔧 预训练阶段应该监控 val_loss
            monitor = early_stopping_config.get('monitor', 'val_loss')
            if monitor == 'loss':
                monitor = 'val_loss'  # 统一使用验证集指标

            early_stopping = EarlyStopping(
                patience=early_stopping_config.get('patience', 10),
                monitor=monitor,
                mode=early_stopping_config.get('mode', 'min'),
                restore_best_weights=early_stopping_config.get('restore_best_weights', True)
            )
            self.add_callback(early_stopping)
            print(f"  ✅ EarlyStopping: monitor={monitor}, mode=min, patience={early_stopping_config.get('patience', 10)}")

        # 模型保存
        if checkpoint_config.get('save_best', True):
            # 🔧 预训练阶段强制监控 val_loss (忽略配置文件中的 val_acc)
            model_checkpoint = ModelCheckpoint(
                save_dir=self.checkpoint_manager.checkpoint_dir,
                monitor='val_loss',  # 🔧 预训练固定监控 val_loss
                mode='min',  # 🔧 预训练固定为 min 模式
                save_frequency=checkpoint_config.get('save_frequency', 5),
                keep_last_n=checkpoint_config.get('keep_last_n', 3)
            )
            self.add_callback(model_checkpoint)
            print(f"  ✅ ModelCheckpoint: monitor=val_loss, mode=min")

    def _check_scheduler_needs_metric(self) -> bool:
        """检查调度器是否需要metric(如ReduceLROnPlateau)"""
        return isinstance(self.scheduler, ReduceLROnPlateau)

    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f'Train Epoch {self.current_epoch}')

        for batch in pbar:
            # 计算损失
            loss, _ = self.compute_loss(batch)

            # 反向传播
            if self.use_amp:
                self.scaler.scale(loss).backward()
                if self.gradient_clip_max_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip_max_norm
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.gradient_clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip_max_norm
                    )
                self.optimizer.step()

            self.optimizer.zero_grad()

            # 累计统计
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # 更新进度条
            pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / num_batches

        return {
            'loss': avg_loss,
            'train_loss': avg_loss
        }

    def validate_epoch(self) -> Dict[str, float]:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                loss, _ = self.compute_loss(batch)
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches

        return {
            'loss': avg_loss,
            'val_loss': avg_loss
        }

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算对比学习损失

        Args:
            batch: 数据批次,包含两个增强视图
                   格式: {'view1': {...}, 'view2': {...}, 'label': ...}

        Returns:
            loss: 损失值
            metrics: 指标字典
        """
        # 🔧 修复: 移动数据到设备
        view1 = {
            'temporal': batch['view1']['temporal'].to(self.device),
            'frequency': batch['view1']['frequency'].to(self.device),
            'timefreq': batch['view1']['timefreq'].to(self.device)
        }

        view2 = {
            'temporal': batch['view2']['temporal'].to(self.device),
            'frequency': batch['view2']['frequency'].to(self.device),
            'timefreq': batch['view2']['timefreq'].to(self.device)
        }

        # 🔧 修复: 构建模型需要的batch格式
        # 模型的forward期望: batch = {'view1': {...}, 'view2': {...}}
        model_batch = {'view1': view1, 'view2': view2}

        # 如果使用SupCon,还需要传递标签
        if 'label' in batch:
            model_batch['label'] = batch['label'].to(self.device)

        # 前向传播 (对比学习模式)
        if self.use_amp:
            with torch.amp.autocast(device_type='cuda' if 'cuda' in self.device else 'cpu'):
                z1, z2 = self.model(model_batch, mode='contrastive')
                loss = self.criterion(z1, z2)
        else:
            z1, z2 = self.model(model_batch, mode='contrastive')
            loss = self.criterion(z1, z2)

        metrics = {
            'loss': loss.item()
        }

        return loss, metrics

    def train(self, epochs: int, log_interval: int = 10, save_config: Dict = None):
        """
        训练主循环

        Args:
            epochs: 训练轮数
            log_interval: 日志打印间隔
            save_config: 保存的配置信息
        """
        print(f"\n开始训练 {epochs} 个epochs...")
        print(f"  - 训练集: {len(self.train_loader.dataset)} 样本")
        print(f"  - 验证集: {len(self.val_loader.dataset)} 样本")
        print(f"  - Batch size: {self.train_loader.batch_size}")

        # 保存配置
        if save_config is not None:
            self._save_config(save_config)

        history = {
            'train_loss': [],
            'val_loss': []
        }

        for epoch in range(epochs):
            self.current_epoch = epoch

            # 训练
            train_metrics = self.train_epoch()

            # 验证
            val_metrics = self.validate_epoch()

            # 合并指标
            epoch_metrics = {**train_metrics, **val_metrics}

            # 记录历史
            history['train_loss'].append(train_metrics['loss'])
            history['val_loss'].append(val_metrics['loss'])

            # 打印日志
            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")

            # 调用callbacks
            self.callbacks.on_epoch_end(epoch, epoch_metrics, self.model, self.optimizer)

            # 更新学习率
            if self.scheduler is not None:
                if self.scheduler_needs_metric:
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()

                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"  Learning Rate: {current_lr:.6f}")

            # 检查早停
            if self.callbacks.should_stop():
                print(f"\n早停触发,停止训练")
                break

        # 训练结束后加载最佳模型
        self._load_best_model()

        # 绘制训练曲线
        self._plot_curves(history)

        print("\n训练完成!")

    def save_pretrained_weights(self, save_path: str):
        """
        🔧 新增方法: 保存预训练权重

        只保存模型的state_dict,用于后续微调阶段加载
        注意: 会排除projection_head的权重(因为微调不需要)

        Args:
            save_path: 保存路径
        """
        print(f"\n保存预训练权重到: {save_path}")

        # 获取完整的state_dict
        full_state_dict = self.model.state_dict()

        # 🔧 只保留backbone权重,排除projection_head和classifier
        pretrained_dict = {}
        excluded_keys = []

        for k, v in full_state_dict.items():
            # 排除投影头和分类器
            if 'projection_head' in k or 'classifier' in k:
                excluded_keys.append(k)
                continue
            pretrained_dict[k] = v

        # 保存
        torch.save({
            'model_state_dict': pretrained_dict,
            'epoch': self.current_epoch,
            'temperature': self.temperature,
            'loss_type': self.loss_type
        }, save_path)

        print(f"✓ 预训练权重保存成功")
        print(f"  - 保存参数: {len(pretrained_dict)} 个")
        print(f"  - 排除参数: {len(excluded_keys)} 个")
        if excluded_keys:
            print(f"  - 排除的层: {', '.join(set([k.split('.')[0] for k in excluded_keys]))}")


if __name__ == '__main__':
    """测试代码"""
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from models import create_model
    from datasets import ContrastiveDataset
    from torch.utils.data import DataLoader
    from training.optimizer_factory import create_optimizer_from_config
    from training.scheduler_factory import create_scheduler_from_config

    print("=" * 80)
    print("测试ContrastiveTrainer (修复版)")
    print("=" * 80)

    # 创建模型
    model = create_model(config='small', enable_contrastive=True)
    print(f"✓ 模型创建成功")

    # 创建数据集(使用少量数据测试)
    print("\n创建数据集...")
    train_dataset = ContrastiveDataset(
        data_dir='raw_datasets/train',
        mode='train',
        fold=0,
        n_folds=5,
        use_labels=False,
        cache_data=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0
    )

    print(f"✓ 数据加载器创建成功")

    # 创建优化器和调度器
    optimizer_config = {'type': 'adam', 'lr': 0.001}
    optimizer = create_optimizer_from_config(model, optimizer_config)

    scheduler_config = {'type': 'cosine', 'T_max': 10, 'eta_min': 1e-6}
    scheduler, needs_metric = create_scheduler_from_config(
        optimizer,
        scheduler_config,
        total_epochs=10
    )

    print(f"✓ 优化器和调度器创建成功")
    print(f"  - Scheduler需要metric: {needs_metric}")

    # 创建训练器
    trainer = ContrastiveTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_type='ntxent',
        temperature=0.07,
        device='cpu',
        experiment_dir='/tmp/test_contrastive_fixed',
        use_amp=False
    )

    print(f"✓ Trainer创建成功")

    # 设置callbacks
    print("\n设置callbacks...")
    trainer.setup_callbacks(
        early_stopping_config={'enable': True, 'patience': 3, 'monitor': 'val_loss'},
        checkpoint_config={'save_best': True, 'monitor_metric': 'val_acc'}  # 这里会被覆盖
    )

    # 测试保存预训练权重方法
    print("\n测试save_pretrained_weights方法...")
    test_path = '/tmp/test_pretrained_weights.pth'
    trainer.save_pretrained_weights(test_path)

    # 验证保存的文件
    if os.path.exists(test_path):
        checkpoint = torch.load(test_path, map_location='cpu')
        print(f"\n✓ 权重文件验证成功")
        print(f"  - Keys: {list(checkpoint.keys())}")
        print(f"  - Epoch: {checkpoint['epoch']}")
        print(f"  - 参数数量: {len(checkpoint['model_state_dict'])}")

    print("\n" + "=" * 80)
    print("✓ 所有测试通过!")
    print("=" * 80)
