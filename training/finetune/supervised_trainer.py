"""
有监督训练器
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple, Optional
from tqdm import tqdm
from pathlib import Path
import time
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.trainer_base import TrainerBase
from losses import CombinedLoss, ProgressiveCombinedLoss, compute_class_weights
from augmentation.mixup import MultiModalMixup, ManifoldMixup
from utils.visualization import TrainingVisualizer
from training.callbacks import EarlyStopping, ModelCheckpoint

class SupervisedTrainer(TrainerBase):
    """有监督训练器"""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        loss_config: Dict[str, Any],
        device: str = 'cuda',
        experiment_dir: str = 'experiments/runs',
        use_amp: bool = False,
        mixup_config: Optional[Dict[str, Any]] = None,
        gradient_clip_max_norm: float = 1.0
    ):
        """
        Args:
            model: 模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            optimizer: 优化器
            scheduler: 学习率调度器
            loss_config: 损失函数配置
            device: 训练设备
            experiment_dir: 实验保存目录
            use_amp: 是否使用混合精度训练
            mixup_config: Mixup配置字典
            gradient_clip_max_norm: 梯度裁剪的最大范数
        """
        super().__init__(
            model, train_loader, val_loader, optimizer, scheduler,
            device, experiment_dir, use_amp
        )

        self.gradient_clip_max_norm = gradient_clip_max_norm

        # 创建损失函数
        self.criterion = self._create_criterion(loss_config)

        # 创建Mixup管理器
        self.mixup_manager = None
        if mixup_config:
            self.mixup_manager = MultiModalMixup(
                time_domain_config=mixup_config.get('time_domain'),
                frequency_domain_config=mixup_config.get('frequency_domain'),
                feature_level_config=mixup_config.get('feature_level')
            )

        print(f"SupervisedTrainer初始化完成")
        if self.mixup_manager:
            print(f"  - 时域Mixup: {self.mixup_manager.time_domain_enabled}")
            print(f"  - 频域Mixup: {self.mixup_manager.frequency_domain_enabled}")
            print(f"  - 特征层Mixup: {self.mixup_manager.feature_level_enabled}")
            print(f"  ⚠️  注意: 使用输入层Mixup时将禁用ArcFace损失(仅使用Softmax)")
        else:
            print(f"  - Mixup: 未启用")
        print(f"  - 梯度裁剪: {gradient_clip_max_norm}")

    def _create_criterion(self, loss_config: Dict[str, Any]):
        """创建损失函数"""
        if loss_config.get('use_progressive', False):
            # 渐进式组合损失
            criterion = ProgressiveCombinedLoss(
                focal_alpha=loss_config['focal'].get('alpha', None),
                focal_gamma_init=loss_config['focal'].get('gamma_init', None),
                focal_gamma_min=loss_config['focal'].get('gamma_min', None),
                arcface_weight_init=loss_config['arcface'].get('weight_init', None),
                arcface_weight_max=loss_config['arcface'].get('weight_max', None),
                label_smoothing=loss_config.get('label_smoothing', 0.0)
            )
        else:
            # 固定权重组合损失
            criterion = CombinedLoss(
                focal_alpha=loss_config['focal'].get('alpha', None),
                focal_gamma=loss_config['focal'].get('gamma_init', None),
                focal_weight=loss_config['focal'].get('weight', None),
                arcface_weight=loss_config['arcface'].get('weight_init', 0.5),
                label_smoothing=loss_config.get('label_smoothing', 0.0)
            )

        return criterion

    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """
        将batch数据移到设备上

        Args:
            batch: 批次数据字典

        Returns:
            移到设备后的batch
        """
        device_batch = {}

        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            elif isinstance(value, dict):
                # 递归处理嵌套字典（如对比学习的view1/view2）
                device_batch[key] = {}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, torch.Tensor):
                        device_batch[key][sub_key] = sub_value.to(self.device)
                    else:
                        device_batch[key][sub_key] = sub_value
            else:
                device_batch[key] = value

        return device_batch

    def train_epoch(self) -> Dict[str, float]:
        """
        训练一个epoch

        Returns:
            metrics: 训练指标字典 {'train_loss': ..., 'train_acc': ...}
        """
        self.model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        pbar = tqdm(self.train_loader, desc=f"Training Epoch {self.current_epoch+1}")

        for batch_idx, batch in enumerate(pbar):
            # 【关键修复】：先将整个batch移到设备
            batch = self._move_batch_to_device(batch)

            # 应用输入层mixup（如果启用）
            if self.mixup_manager:
                # 注意：现在batch已经在设备上，所以不需要在apply_input_mixup中再次移动
                mixed_batch, labels_a, labels_b, lam = self.mixup_manager.apply_input_mixup(
                    batch, self.device
                )
                use_mixup = labels_b is not None
            else:
                mixed_batch = batch
                labels_a = batch['label']  # 已经在设备上
                labels_b = None
                lam = 1.0
                use_mixup = False

            # 前向传播 - 根据是否使用Mixup决定计算方式
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    softmax_logits, arcface_logits, features = self.model(
                        mixed_batch, mode='supervised'
                    )

                    # Mixup时只使用Softmax损失,不使用ArcFace
                    if use_mixup:
                        loss_a = F.cross_entropy(softmax_logits, labels_a, reduction='mean')
                        loss_b = F.cross_entropy(softmax_logits, labels_b, reduction='mean')
                        loss = lam * loss_a + (1 - lam) * loss_b
                    else:
                        loss, loss_dict = self.criterion(softmax_logits, arcface_logits, labels_a)
            else:
                softmax_logits, arcface_logits, features = self.model(
                    mixed_batch, mode='supervised'
                )

                if use_mixup:
                    loss_a = F.cross_entropy(softmax_logits, labels_a, reduction='mean')
                    loss_b = F.cross_entropy(softmax_logits, labels_b, reduction='mean')
                    loss = lam * loss_a + (1 - lam) * loss_b
                else:
                    loss, loss_dict = self.criterion(softmax_logits, arcface_logits, labels_a)

            # 反向传播和优化
            self.optimizer.zero_grad()

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

            # 统计准确率 - 使用softmax_logits
            predictions = torch.argmax(softmax_logits, dim=1)
            if use_mixup:
                # Mixup情况下，使用主要标签计算准确率
                correct = (predictions == labels_a).sum().item()
            else:
                correct = (predictions == labels_a).sum().item()

            total_correct += correct
            total_samples += labels_a.size(0)
            total_loss += loss.item()

            # 更新进度条
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': total_correct / total_samples
            })

        # 计算平均指标
        avg_loss = total_loss / len(self.train_loader)
        avg_acc = total_correct / total_samples

        return {
            'train_loss': avg_loss,
            'train_acc': avg_acc
        }

    def validate_epoch(self) -> Dict[str, float]:
        """
        验证一个epoch

        Returns:
            metrics: 验证指标字典 {'val_loss': ..., 'val_acc': ...}
        """
        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        pbar = tqdm(self.val_loader, desc=f"Validation Epoch {self.current_epoch+1}")

        with torch.no_grad():
            for batch in pbar:
                # 【关键修复】：将batch移到设备
                batch = self._move_batch_to_device(batch)
                labels = batch['label']  # 已经在设备上

                # 验证时也使用双头输出和完整损失
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        softmax_logits, arcface_logits, features = self.model(
                            batch, mode='supervised'
                        )
                        loss, loss_dict = self.criterion(softmax_logits, arcface_logits, labels)
                else:
                    softmax_logits, arcface_logits, features = self.model(
                        batch, mode='supervised'
                    )
                    loss, loss_dict = self.criterion(softmax_logits, arcface_logits, labels)

                # 统计准确率 - 使用softmax_logits
                predictions = torch.argmax(softmax_logits, dim=1)
                correct = (predictions == labels).sum().item()

                total_correct += correct
                total_samples += labels.size(0)
                total_loss += loss.item()

                # 更新进度条
                pbar.set_postfix({
                    'loss': loss.item(),
                    'acc': total_correct / total_samples
                })

        # 计算平均指标
        avg_loss = total_loss / len(self.val_loader)
        avg_acc = total_correct / total_samples

        return {
            'val_loss': avg_loss,
            'val_acc': avg_acc
        }

    def compute_loss(self, batch, labels=None):
        """
        计算损失（用于基类的通用训练流程）

        Args:
            batch: 输入批次
            labels: 标签（如果None，从batch中获取）

        Returns:
            loss: 损失值
            loss_dict: 损失组成字典
        """
        if labels is None:
            labels = batch['label']

        # 前向传播
        if self.use_amp:
            with torch.cuda.amp.autocast():
                softmax_logits, arcface_logits, features = self.model(
                    batch, mode='supervised'
                )
                loss, loss_dict = self.criterion(softmax_logits, arcface_logits, labels)
        else:
            softmax_logits, arcface_logits, features = self.model(
                batch, mode='supervised'
            )
            loss, loss_dict = self.criterion(softmax_logits, arcface_logits, labels)

        return loss, loss_dict

    def setup_callbacks(
            self,
            early_stopping_config: Dict[str, Any],
            checkpoint_config: Dict[str, Any]
    ):
        """
        设置callbacks (覆盖父类方法以适配监督学习)

        对于监督学习:
        - EarlyStopping 监控 配置指定的指标 (通常是'val_acc', mode='max')
        - ModelCheckpoint 监控 配置指定的指标 (通常是'val_acc', mode='max')

        Args:
            early_stopping_config: 早停配置
            checkpoint_config: checkpoint配置
        """
        print("\n配置Callbacks:")

        # 早停配置
        if early_stopping_config.get('enable', True):
            monitor = early_stopping_config.get('monitor', 'val_acc')
            mode = early_stopping_config.get('mode', 'max')

            early_stopping = EarlyStopping(
                patience=early_stopping_config.get('patience', 15),
                monitor=monitor,
                mode=mode,
                restore_best_weights=early_stopping_config.get('restore_best_weights', True)
            )
            self.add_callback(early_stopping)
            print(f"  ✅ EarlyStopping:")
            print(f"     - monitor: {monitor}")
            print(f"     - mode: {mode}")
            print(f"     - patience: {early_stopping.patience}")

        # Checkpoint配置
        if checkpoint_config.get('save_best', True):
            monitor = checkpoint_config.get('monitor_metric', 'val_acc')
            mode = checkpoint_config.get('monitor_mode', 'max')

            model_checkpoint = ModelCheckpoint(
                save_dir=self.checkpoint_manager.checkpoint_dir,
                monitor=monitor,
                mode=mode,
                save_frequency=checkpoint_config.get('save_frequency', 5),
                keep_last_n=checkpoint_config.get('keep_last_n', 3)
            )
            self.add_callback(model_checkpoint)
            print(f"  ✅ ModelCheckpoint:")
            print(f"     - monitor: {monitor}")
            print(f"     - mode: {mode}")
            print(f"     - save_frequency: {checkpoint_config.get('save_frequency', 5)}")
            print(f"     - keep_last_n: {checkpoint_config.get('keep_last_n', 3)}")

    def train(
            self,
            epochs: int,
            log_interval: int = 10,
            save_config: Optional[Dict] = None
    ):
        """
        完整训练流程 (带进度条和完善的callback机制)

        Args:
            epochs: 训练轮数
            log_interval: 日志打印间隔
            save_config: 保存配置
        """
        print("=" * 80)
        print(f"开始有监督训练: {epochs} epochs")
        print(f"设备: {self.device}")
        print(f"训练样本数: {len(self.train_loader.dataset)}")
        print(f"验证样本数: {len(self.val_loader.dataset)}")
        print(f"Batch size: {self.train_loader.batch_size}")
        print("=" * 80)

        # 保存配置
        if save_config:
            self._save_config(save_config)

        # 训练循环
        for epoch in range(epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            # 🔧 如果使用渐进式损失，更新损失权重
            if isinstance(self.criterion, ProgressiveCombinedLoss):
                self.criterion.update_weights(epoch, epochs)

            # 训练一个epoch
            train_metrics = self.train_epoch(epoch=epoch, log_interval=log_interval)

            # 验证一个epoch
            val_metrics = self.validate_epoch(epoch=epoch)

            # 更新学习率
            self._update_lr(val_metrics)

            # 记录学习率
            lr = self.optimizer.param_groups[0]['lr']

            # 记录指标
            epoch_metrics = {
                **train_metrics,
                **val_metrics,
                'learning_rate': lr
            }
            self.metrics_tracker.update(epoch_metrics)

            # 回调
            callback_metrics = {
                'epoch': epoch,
                'train_loss': train_metrics.get('train_loss', 0),
                'val_loss': val_metrics.get('val_loss', 0),
                'val_acc': val_metrics.get('val_acc', 0)
            }
            self.callbacks.on_epoch_end(epoch, callback_metrics, self.model, self.optimizer)

            # 打印epoch总结
            epoch_time = time.time() - epoch_start_time
            self._print_epoch_summary(epoch, epochs, train_metrics, val_metrics, epoch_time)

            # 检查早停
            if self.callbacks.should_stop():
                print(f"\n{'=' * 80}")
                print(f"早停触发，在第 {epoch + 1} 轮停止训练")
                print(f"{'=' * 80}")
                break

        # 训练结束
        print("\n" + "=" * 80)
        print("有监督训练完成!")
        print("=" * 80)

        # 绘制训练曲线
        self._plot_curves()

        # 加载最佳模型
        self._load_best_model()

    def _print_epoch_summary(
            self,
            epoch: int,
            total_epochs: int,
            train_metrics: Dict[str, float],
            val_metrics: Dict[str, float],
            epoch_time: float
    ):
        """打印epoch总结信息"""
        lr = self.optimizer.param_groups[0]['lr']

        print(f"\n{'=' * 80}")
        print(f"Epoch [{epoch + 1:3d}/{total_epochs}] 总结:")
        print(f"  - 训练损失: {train_metrics['train_loss']:.4f}")
        print(f"  - 训练准确率: {train_metrics['train_acc']:.4f}")
        print(f"  - 验证损失: {val_metrics['val_loss']:.4f}")
        print(f"  - 验证准确率: {val_metrics['val_acc']:.4f}")
        print(f"  - 学习率: {lr:.6f}")
        print(f"  - 用时: {epoch_time:.2f}s")
        print(f"{'=' * 80}")

    def _plot_curves(self):
        """
        绘制训练曲线

        对于监督学习，主要绘制:
        - 损失曲线 (train_loss vs val_loss)
        - 准确率曲线 (train_acc vs val_acc)
        - 学习率曲线
        """
        print("\n绘制训练曲线...")

        # 确保可视化目录存在
        vis_dir = Path(self.experiment_dir) / 'visualizations'
        vis_dir.mkdir(parents=True, exist_ok=True)

        # 获取训练历史
        history = self.metrics_tracker.get_history()

        # 创建可视化器
        visualizer = TrainingVisualizer(save_dir=str(vis_dir))

        # 绘制损失曲线
        if 'train_loss' in history and 'val_loss' in history:
            visualizer.plot_loss_curves(
                train_loss=history['train_loss'],
                val_loss=history['val_loss'],
                title='Training and Validation Loss',
                save_name='loss_curves.png',
                show=False
            )
            print(f"  ✓ 损失曲线已保存: {vis_dir / 'loss_curves.png'}")

        # 绘制准确率曲线
        if 'train_acc' in history and 'val_acc' in history:
            visualizer.plot_accuracy_curves(
                train_acc=history['train_acc'],
                val_acc=history['val_acc'],
                title='Training and Validation Accuracy',
                save_name='accuracy_curves.png',
                show=False
            )
            print(f"  ✓ 准确率曲线已保存: {vis_dir / 'accuracy_curves.png'}")

        # 绘制学习率曲线
        if 'learning_rate' in history and len(history['learning_rate']) > 0:
            visualizer.plot_learning_rate(
                learning_rates=history['learning_rate'],
                title='Learning Rate Schedule',
                save_name='learning_rate.png',
                show=False
            )
            print(f"  ✓ 学习率曲线已保存: {vis_dir / 'learning_rate.png'}")

        print("\n  评估验证集并生成可视化...")
        self._plot_evaluation_visualizations(vis_dir)
        print("✓ 训练曲线绘制完成")

    def _plot_evaluation_visualizations(self, vis_dir: Path):
        """在验证集上评估并生成混淆矩阵和t-SNE可视化"""

        # 设置模型为评估模式但保持双头输出
        self.model.train()
        for module in self.model.modules():
            if isinstance(module, (nn.Dropout, nn.BatchNorm1d, nn.BatchNorm2d)):
                module.eval()

        all_labels = []
        all_preds = []
        all_features = []

        print("    收集验证集预测和特征...")
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="    提取特征", leave=False, ncols=80):
                batch = self._move_batch_to_device(batch)
                labels = batch['label']

                # 前向传播获取特征
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        softmax_logits, arcface_logits, features = self.model(
                            batch, mode='supervised'
                        )
                else:
                    softmax_logits, arcface_logits, features = self.model(
                        batch, mode='supervised'
                    )

                # 收集预测和特征
                predictions = torch.argmax(softmax_logits, dim=1)
                all_labels.append(labels.cpu().numpy())
                all_preds.append(predictions.cpu().numpy())
                all_features.append(features.cpu().numpy())

        # 合并所有batch
        all_labels = np.concatenate(all_labels)
        all_preds = np.concatenate(all_preds)
        all_features = np.concatenate(all_features)

        # 1. 绘制混淆矩阵
        print("    绘制混淆矩阵...")
        cm = confusion_matrix(all_labels, all_preds)

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=range(cm.shape[0]),
            yticklabels=range(cm.shape[1]),
            cbar_kws={'label': 'Count'}
        )
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()

        cm_path = vis_dir / 'confusion_matrix.png'
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    ✓ 混淆矩阵已保存: {cm_path}")

        # 2. 绘制归一化混淆矩阵
        print("    绘制归一化混淆矩阵...")
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=range(cm.shape[0]),
            yticklabels=range(cm.shape[1]),
            cbar_kws={'label': 'Proportion'},
            vmin=0,
            vmax=1
        )
        plt.title('Normalized Confusion Matrix', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()

        cm_norm_path = vis_dir / 'confusion_matrix_normalized.png'
        plt.savefig(cm_norm_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    ✓ 归一化混淆矩阵已保存: {cm_norm_path}")

        # 3. 绘制t-SNE可视化
        print("    绘制t-SNE可视化...")

        # 降采样以加速（如果样本太多）
        max_samples = 2000
        if len(all_features) > max_samples:
            indices = np.random.choice(len(all_features), max_samples, replace=False)
            features_subset = all_features[indices]
            labels_subset = all_labels[indices]
        else:
            features_subset = all_features
            labels_subset = all_labels

        # 执行t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        features_2d = tsne.fit_transform(features_subset)

        # 绘制t-SNE图
        plt.figure(figsize=(12, 10))
        num_classes = len(np.unique(labels_subset))
        colors = plt.cm.tab10(np.linspace(0, 1, num_classes))

        for i in range(num_classes):
            mask = labels_subset == i
            plt.scatter(
                features_2d[mask, 0],
                features_2d[mask, 1],
                c=[colors[i]],
                label=f'Class {i}',
                alpha=0.6,
                s=50,
                edgecolors='black',
                linewidths=0.5
            )

        plt.xlabel('t-SNE Component 1', fontsize=12)
        plt.ylabel('t-SNE Component 2', fontsize=12)
        plt.title('t-SNE Visualization of Features', fontsize=14, fontweight='bold', pad=15)
        plt.legend(fontsize=10, loc='best', framealpha=0.9)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()

        tsne_path = vis_dir / 'tsne_visualization.png'
        plt.savefig(tsne_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    ✓ t-SNE可视化已保存: {tsne_path}")

        # 4. 打印分类报告
        print("\n    分类报告:")
        print(classification_report(
            all_labels,
            all_preds,
            target_names=[f'Class {i}' for i in range(num_classes)],
            digits=4
        ))

# ============================================================================
# 预训练权重加载工具函数
# ============================================================================

def load_pretrained_weights(
        model: nn.Module,
        pretrained_path: str,
        device: str = 'cuda',
        freeze_backbone: bool = False,
        freeze_ratio: float = 0.5,
        strict: bool = False
) -> nn.Module:
    """
    加载预训练权重 (仅加载backbone，不加载projection_head和classifier)

    Args:
        model: 目标模型
        pretrained_path: 预训练权重路径
        device: 设备
        freeze_backbone: 是否冻结backbone
        freeze_ratio: 冻结比例 (0.0-1.0)
        strict: 是否严格匹配权重

    Returns:
        model: 加载权重后的模型
    """
    print(f"\n{'=' * 80}")
    print(f"加载预训练权重: {pretrained_path}")
    print(f"{'=' * 80}")

    # 检查文件是否存在
    if not Path(pretrained_path).exists():
        raise FileNotFoundError(f"预训练权重文件不存在: {pretrained_path}")

    # 加载预训练权重
    checkpoint = torch.load(pretrained_path, map_location=device)

    # 提取模型权重
    if 'model_state_dict' in checkpoint:
        pretrained_dict = checkpoint['model_state_dict']
        print(f"✓ 从checkpoint中提取预训练权重")
        if 'epoch' in checkpoint:
            print(f"  - 预训练epoch: {checkpoint['epoch']}")
        if 'best_val_loss' in checkpoint:
            print(f"  - 最佳验证损失: {checkpoint['best_val_loss']:.4f}")
    else:
        pretrained_dict = checkpoint
        print(f"✓ 直接加载预训练权重")

    # 获取模型当前的state_dict
    model_dict = model.state_dict()

    # 过滤出匹配的权重（排除projection_head和classifier）
    matched_dict = {}
    unmatched_keys = []

    for k, v in pretrained_dict.items():
        # 检查key是否在模型中存在
        if k in model_dict:
            # 检查shape是否匹配
            if model_dict[k].shape == v.shape:
                matched_dict[k] = v
            else:
                unmatched_keys.append(f"{k} (shape mismatch: {model_dict[k].shape} vs {v.shape})")
        else:
            unmatched_keys.append(f"{k} (not in model)")

    # 更新模型权重
    model_dict.update(matched_dict)
    model.load_state_dict(model_dict, strict=False)

    print(f"\n✓ 预训练权重加载完成")
    print(f"  - 匹配参数数量: {len(matched_dict)}")
    print(f"  - 未匹配参数数量: {len(unmatched_keys)}")

    if unmatched_keys:
        print(f"\n未匹配的参数:")
        for key in unmatched_keys[:5]:  # 只显示前5个
            print(f"  - {key}")
        if len(unmatched_keys) > 5:
            print(f"  ... 还有 {len(unmatched_keys) - 5} 个未匹配参数")

    # 冻结backbone（如果需要）
    if freeze_backbone:
        print(f"\n冻结backbone参数 (freeze_ratio={freeze_ratio}):")

        # 获取所有需要冻结的层
        backbone_params = [
            name for name, param in model.named_parameters()
            if 'classifier' not in name and 'projection_head' not in name
        ]

        num_to_freeze = int(len(backbone_params) * freeze_ratio)
        frozen_params = backbone_params[:num_to_freeze]

        for name, param in model.named_parameters():
            if name in frozen_params:
                param.requires_grad = False

        print(f"  - 总backbone参数: {len(backbone_params)}")
        print(f"  - 冻结参数数量: {num_to_freeze}")
        print(f"  - 可训练参数数量: {len(backbone_params) - num_to_freeze}")

    print(f"{'=' * 80}\n")

    return model
