"""
对比学习训练器
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Any
import time
import sys
import os
from tqdm import tqdm

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.trainer_base import TrainerBase
from utils.visualization import TrainingVisualizer


class ContrastiveTrainer(TrainerBase):
    """
    对比学习训练器 (完整增强版)

    功能特性：
    - 支持NT-Xent和SupCon两种对比学习损失
    - 集成tqdm进度条显示
    - 完善的callback机制（EarlyStopping、ModelCheckpoint）
    - 自动保存最佳模型和定期checkpoint
    - 训练曲线可视化
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_type: str = 'ntxent',
        temperature: float = 0.07,
        device: str = 'cuda',
        experiment_dir: str = './experiments',
        use_amp: bool = False,
        gradient_clip_max_norm: float = 1.0,
        use_tqdm: bool = True
    ):
        """
        初始化对比学习训练器

        Args:
            model: 模型 (需要支持 mode='contrastive')
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            optimizer: 优化器
            scheduler: 学习率调度器
            loss_type: 损失类型 ('ntxent' 或 'supcon')
            temperature: 温度参数
            device: 训练设备
            experiment_dir: 实验目录
            use_amp: 是否使用混合精度训练
            gradient_clip_max_norm: 梯度裁剪最大范数
            use_tqdm: 是否使用tqdm进度条
        """
        # 调用父类初始化
        super().__init__(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            experiment_dir=experiment_dir,
            use_amp=use_amp
        )

        # 保存额外的参数
        self.temperature = temperature
        self.loss_type = loss_type
        self.gradient_clip_max_norm = gradient_clip_max_norm
        self.use_tqdm = use_tqdm

        # 创建对比学习损失函数
        self._create_criterion()

        print(f"ContrastiveTrainer 初始化完成")
        print(f"  - 损失类型: {loss_type}")
        print(f"  - 温度参数: {temperature}")
        print(f"  - 梯度裁剪: {gradient_clip_max_norm}")
        print(f"  - 使用tqdm: {use_tqdm}")

    def _create_criterion(self):
        """创建对比学习损失函数"""
        from losses import NTXentLoss, SupConLoss

        if self.loss_type == 'ntxent':
            self.criterion = NTXentLoss(temperature=self.temperature)
        elif self.loss_type == 'supcon':
            self.criterion = SupConLoss(temperature=self.temperature)
        else:
            raise ValueError(f"不支持的损失类型: {self.loss_type}")

    def compute_loss(self, batch: Dict, **kwargs) -> tuple:
        """
        计算对比学习损失 (实现父类抽象方法)

        Args:
            batch: 批次数据字典,包含:
                {
                    'view1': {'temporal', 'frequency', 'timefreq'},
                    'view2': {'temporal', 'frequency', 'timefreq'},
                    'label': ... (可选,仅SupCon需要)
                }
            **kwargs: 其他参数

        Returns:
            loss: 损失值 (torch.Tensor)
            metrics: 指标字典 (Dict)
        """
        # 使用 mode='contrastive' 调用模型
        z1, z2 = self.model(batch, mode='contrastive')

        # 计算对比损失
        if self.loss_type == 'supcon' and 'label' in batch:
            # SupCon模式需要标签
            labels = batch['label'].to(self.device)
            loss = self.criterion(z1, z2, labels)
        else:
            # NT-Xent模式不需要标签
            loss = self.criterion(z1, z2)

        # 对比学习通常不需要额外指标
        metrics = {}

        return loss, metrics

    def train_epoch(self, epoch: int = 0, log_interval: int = 10) -> Dict[str, float]:
        """
        训练一个epoch (带tqdm进度条)

        Args:
            epoch: 当前epoch编号
            log_interval: 日志打印间隔

        Returns:
            metrics: 训练指标字典 {'train_loss': ...}
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # 创建tqdm进度条
        if self.use_tqdm:
            pbar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch+1} [Train]",
                ncols=100,
                leave=True
            )
        else:
            pbar = self.train_loader

        for batch_idx, batch in enumerate(pbar):
            # 将batch移到设备
            batch = self._move_batch_to_device(batch)

            # 前向传播
            self.optimizer.zero_grad()

            if self.use_amp:
                # 混合精度训练
                with torch.cuda.amp.autocast():
                    loss, _ = self.compute_loss(batch)

                self.scaler.scale(loss).backward()

                # 梯度裁剪
                if self.gradient_clip_max_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip_max_norm
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # 正常训练
                loss, _ = self.compute_loss(batch)
                loss.backward()

                # 梯度裁剪
                if self.gradient_clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip_max_norm
                    )

                self.optimizer.step()

            # 累积损失
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # 更新tqdm进度条
            if self.use_tqdm:
                avg_loss = total_loss / num_batches
                lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'lr': f'{lr:.6f}'
                })

        avg_loss = total_loss / num_batches
        return {'train_loss': avg_loss}

    def validate_epoch(self, epoch: int = 0) -> Dict[str, float]:
        """
        验证一个epoch (带tqdm进度条)

        Args:
            epoch: 当前epoch编号

        Returns:
            metrics: 验证指标字典 {'val_loss': ...}
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        # 创建tqdm进度条
        if self.use_tqdm:
            pbar = tqdm(
                self.val_loader,
                desc=f"Epoch {epoch+1} [Val]",
                ncols=100,
                leave=True
            )
        else:
            pbar = self.val_loader

        with torch.no_grad():
            for batch in pbar:
                # 将batch移到设备
                batch = self._move_batch_to_device(batch)

                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        loss, _ = self.compute_loss(batch)
                else:
                    loss, _ = self.compute_loss(batch)

                total_loss += loss.item()
                num_batches += 1

                # 更新tqdm进度条
                if self.use_tqdm:
                    avg_loss = total_loss / num_batches
                    pbar.set_postfix({'loss': f'{avg_loss:.4f}'})

        avg_loss = total_loss / num_batches
        return {'val_loss': avg_loss}

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
            if key in ['view1', 'view2']:
                # view1和view2是包含多个模态的字典
                device_batch[key] = {}
                for modal_key, modal_tensor in value.items():
                    device_batch[key][modal_key] = modal_tensor.to(self.device)
            elif isinstance(value, torch.Tensor):
                # 标签等其他tensor
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value

        return device_batch

    def setup_callbacks(
        self,
        early_stopping_config: Dict,
        checkpoint_config: Dict
    ):
        """
        设置callbacks (覆盖父类方法以适配对比学习)

        对于预训练:
        - EarlyStopping 监控 'val_loss' (mode='min')
        - ModelCheckpoint 监控 'val_loss' (mode='min')

        Args:
            early_stopping_config: 早停配置
            checkpoint_config: checkpoint配置
        """
        from training.callbacks import EarlyStopping, ModelCheckpoint

        print("\n配置Callbacks:")

        # 早停配置
        if early_stopping_config.get('enable', True):
            # 对比学习固定监控 val_loss
            monitor = 'val_loss'

            early_stopping = EarlyStopping(
                patience=early_stopping_config.get('patience', 10),
                monitor=monitor,
                mode='min',  # 对比学习固定为 min 模式
                restore_best_weights=early_stopping_config.get('restore_best_weights', True)
            )
            self.add_callback(early_stopping)
            print(f"  ✅ EarlyStopping:")
            print(f"     - monitor: {monitor}")
            print(f"     - mode: min")
            print(f"     - patience: {early_stopping.patience}")

        # Checkpoint配置
        if checkpoint_config.get('save_best', True):
            model_checkpoint = ModelCheckpoint(
                save_dir=self.checkpoint_manager.checkpoint_dir,
                monitor='val_loss',  # 对比学习固定监控 val_loss
                mode='min',  # 对比学习固定为 min 模式
                save_frequency=checkpoint_config.get('save_frequency', 5),
                keep_last_n=checkpoint_config.get('keep_last_n', 3)
            )
            self.add_callback(model_checkpoint)
            print(f"  ✅ ModelCheckpoint:")
            print(f"     - monitor: val_loss")
            print(f"     - mode: min")
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
        print(f"开始对比学习预训练: {epochs} epochs")
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
                'val_acc': 0  # 对比学习没有accuracy,传入0
            }
            self.callbacks.on_epoch_end(epoch, callback_metrics, self.model, self.optimizer)

            # 打印epoch总结
            epoch_time = time.time() - epoch_start_time
            self._print_epoch_summary(epoch, epochs, train_metrics, val_metrics, epoch_time)

            # 检查早停
            if self.callbacks.should_stop():
                print(f"\n{'='*80}")
                print(f"早停触发,在第 {epoch+1} 轮停止训练")
                print(f"{'='*80}")
                break

        # 训练结束
        print("\n" + "=" * 80)
        print("对比学习预训练完成!")
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

        print(f"\n{'='*80}")
        print(f"Epoch [{epoch+1:3d}/{total_epochs}] 总结:")
        print(f"  - 训练损失: {train_metrics['train_loss']:.4f}")
        print(f"  - 验证损失: {val_metrics['val_loss']:.4f}")
        print(f"  - 学习率: {lr:.6f}")
        print(f"  - 用时: {epoch_time:.2f}s")
        print(f"{'='*80}")

    def _extract_features(self, max_samples: int = 2000):
        """
        从验证集提取特征用于可视化

        Args:
            max_samples: 最大采样数量（避免内存溢出）

        Returns:
            features: 特征数组 (N, feature_dim)
            labels: 标签数组 (N,)
        """
        self.model.eval()
        all_features = []
        all_labels = []

        print(f"\n从验证集提取特征 (最多 {max_samples} 个样本)...")

        with torch.no_grad():
            for batch in self.val_loader:
                # 将batch移到设备
                batch = self._move_batch_to_device(batch)

                # 提取view1的特征（对比学习中两个view都可以，这里选view1）
                if 'view1' in batch:
                    # 对比学习模式：提取view1的融合特征
                    feat_t = self.model.temporal_branch(batch['view1']['temporal'])
                    feat_f = self.model.frequency_branch(batch['view1']['frequency'])
                    feat_tf = self.model.timefreq_branch(batch['view1']['timefreq'])
                    fused_feat = self.model.fusion(feat_t, feat_f, feat_tf)
                else:
                    # 有监督模式：直接提取融合特征
                    feat_t = self.model.temporal_branch(batch['temporal'])
                    feat_f = self.model.frequency_branch(batch['frequency'])
                    feat_tf = self.model.timefreq_branch(batch['timefreq'])
                    fused_feat = self.model.fusion(feat_t, feat_f, feat_tf)

                # 获取标签
                if 'label' in batch:
                    labels = batch['label']
                elif 'labels' in batch:
                    labels = batch['labels']
                else:
                    print("  ⚠ 警告: batch中没有标签，无法绘制t-SNE")
                    return None, None

                all_features.append(fused_feat.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

                # 检查是否达到最大样本数
                total_samples = sum(len(f) for f in all_features)
                if total_samples >= max_samples:
                    break

        # 合并所有batch
        features = np.concatenate(all_features, axis=0)[:max_samples]
        labels = np.concatenate(all_labels, axis=0)[:max_samples]

        print(f"  ✓ 提取完成: {len(features)} 个样本，特征维度 {features.shape[1]}")
        print(f"  ✓ 类别分布: {np.bincount(labels)}")

        return features, labels

    def _plot_curves(self):
        """
        绘制训练曲线

        对于对比学习,主要绘制:
        - 损失曲线 (train_loss vs val_loss)
        - 学习率曲线
        - t-SNE特征可视化（新增）
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
                title='Contrastive Learning Loss',
                save_name='contrastive_loss_curves.png',
                show=False
            )
            print(f"  ✓ 损失曲线已保存: {vis_dir / 'contrastive_loss_curves.png'}")

        # 绘制学习率曲线
        if 'learning_rate' in history and len(history['learning_rate']) > 0:
            visualizer.plot_learning_rate(
                learning_rates=history['learning_rate'],
                title='Learning Rate Schedule',
                save_name='learning_rate.png',
                show=False
            )
            print(f"  ✓ 学习率曲线已保存: {vis_dir / 'learning_rate.png'}")

        # 绘制t-SNE特征可视化
        print("\n绘制t-SNE特征可视化...")
        features, labels = self._extract_features(max_samples=2000)

        if features is not None and labels is not None:
            from utils.visualization import plot_embeddings_2d

            # 绘制t-SNE
            tsne_path = vis_dir / 'tsne_visualization.png'
            plot_embeddings_2d(
                embeddings=features,
                labels=labels,
                method='tsne',
                title='t-SNE Visualization of Learned Features',
                figsize=(10, 8),
                save_path=str(tsne_path),
                show=False
            )
            print(f"  ✓ t-SNE可视化已保存: {tsne_path}")

            # 同时绘制PCA作为对比
            pca_path = vis_dir / 'pca_visualization.png'
            plot_embeddings_2d(
                embeddings=features,
                labels=labels,
                method='pca',
                title='PCA Visualization of Learned Features',
                figsize=(10, 8),
                save_path=str(pca_path),
                show=False
            )
            print(f"  ✓ PCA可视化已保存: {pca_path}")

        print("✓ 训练曲线绘制完成")

    def save_pretrained_weights(self, save_path: str):
        """
        保存预训练权重

        只保存模型的backbone权重,排除projection_head和classifier
        这些权重可以用于后续的微调阶段

        Args:
            save_path: 保存路径
        """
        print(f"\n{'='*80}")
        print(f"保存预训练权重到: {save_path}")
        print(f"{'='*80}")

        # 获取完整的state_dict
        full_state_dict = self.model.state_dict()

        # 只保留backbone权重,排除projection_head和classifier
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
            'loss_type': self.loss_type,
            'best_val_loss': self.metrics_tracker.get_best_metric('val_loss', mode='min')[0]
        }, save_path)

        print(f"✓ 预训练权重保存成功")
        print(f"  - 保存参数数量: {len(pretrained_dict)}")
        print(f"  - 排除参数数量: {len(excluded_keys)}")

        if excluded_keys:
            unique_layers = set([k.split('.')[0] for k in excluded_keys])
            print(f"  - 排除的层: {', '.join(unique_layers)}")

        print(f"{'='*80}")


if __name__ == '__main__':
    """测试代码"""
    print("=" * 80)
    print("ContrastiveTrainer 模块测试")
    print("=" * 80)
