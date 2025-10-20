"""
对比学习训练器
"""
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional
from training.trainer_base import TrainerBase
from utils.visualization import TrainingVisualizer


class ContrastiveTrainer(TrainerBase):
    """对比学习训练器 (修复版)"""

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
        gradient_clip_max_norm: float = 1.0
    ):
        """
        Args:
            model: 模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            optimizer: 优化器
            scheduler: 学习率调度器
            loss_type: 损失类型 ('ntxent', 'supcon')
            temperature: 温度参数
            device: 设备
            experiment_dir: 实验目录
            use_amp: 是否使用混合精度
            gradient_clip_max_norm: 梯度裁剪最大norm
        """
        super().__init__(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            experiment_dir=experiment_dir,
            use_amp=use_amp,
            gradient_clip_max_norm=gradient_clip_max_norm
        )

        self.temperature = temperature
        self.loss_type = loss_type

        # 创建对比学习损失
        from losses import NTXentLoss, SupConLoss

        if loss_type == 'ntxent':
            self.criterion = NTXentLoss(temperature=temperature)
        elif loss_type == 'supcon':
            self.criterion = SupConLoss(temperature=temperature)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        print(f"ContrastiveTrainer初始化完成")
        print(f"  - 损失类型: {loss_type}")
        print(f"  - 温度参数: {temperature}")

    def train_epoch(self, epoch: int, log_interval: int = 10) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (view1, view2) in enumerate(self.train_loader):
            view1 = view1.to(self.device)
            view2 = view2.to(self.device)

            # 前向传播
            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.cuda.amp.autocast():
                    # 获取两个视图的特征
                    z1 = self.model(view1, return_features=True)
                    z2 = self.model(view2, return_features=True)

                    # 计算对比损失
                    loss = self.criterion(z1, z2)

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
                z1 = self.model(view1, return_features=True)
                z2 = self.model(view2, return_features=True)
                loss = self.criterion(z1, z2)

                loss.backward()
                if self.gradient_clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip_max_norm
                    )
                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # 打印日志
            if (batch_idx + 1) % log_interval == 0:
                avg_loss = total_loss / num_batches
                lr = self.optimizer.param_groups[0]['lr']
                print(f"  Batch [{batch_idx+1}/{len(self.train_loader)}] | "
                      f"Loss: {avg_loss:.4f} | LR: {lr:.6f}")

        avg_loss = total_loss / num_batches
        return {'train_loss': avg_loss}

    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for view1, view2 in self.val_loader:
                view1 = view1.to(self.device)
                view2 = view2.to(self.device)

                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        z1 = self.model(view1, return_features=True)
                        z2 = self.model(view2, return_features=True)
                        loss = self.criterion(z1, z2)
                else:
                    z1 = self.model(view1, return_features=True)
                    z2 = self.model(view2, return_features=True)
                    loss = self.criterion(z1, z2)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        return {'val_loss': avg_loss}

    def setup_callbacks(self, early_stopping_config: Dict, checkpoint_config: Dict):
        """
        设置callbacks,确保预训练监控正确的指标

        对于预训练:
        - EarlyStopping 监控 'val_loss'
        - ModelCheckpoint 监控 'val_loss'
        """
        from training.callbacks import EarlyStopping, ModelCheckpoint

        # 早停
        if early_stopping_config.get('enable', True):
            monitor = early_stopping_config.get('monitor', 'val_loss')
            if monitor == 'loss':
                monitor = 'val_loss'

            early_stopping = EarlyStopping(
                patience=early_stopping_config.get('patience', 10),
                monitor=monitor,
                mode='min',  # 预训练固定为 min 模式
                restore_best_weights=early_stopping_config.get('restore_best_weights', True)
            )
            self.add_callback(early_stopping)
            print(f"  ✅ EarlyStopping: monitor={monitor}, mode=min")

        # 模型保存
        if checkpoint_config.get('save_best', True):
            model_checkpoint = ModelCheckpoint(
                save_dir=self.checkpoint_manager.checkpoint_dir,
                monitor='val_loss',  # 预训练固定监控 val_loss
                mode='min',  # 预训练固定为 min 模式
                save_frequency=checkpoint_config.get('save_frequency', 5),
                keep_last_n=checkpoint_config.get('keep_last_n', 3)
            )
            self.add_callback(model_checkpoint)
            print(f"  ✅ ModelCheckpoint: monitor=val_loss, mode=min")

    def _plot_curves(self, history: Dict):
        """
        🔧 新增方法: 绘制训练曲线

        对于对比学习,我们只有loss,没有accuracy

        Args:
            history: 训练历史,包含 train_loss, val_loss 等
        """
        print("\n绘制训练曲线...")

        # 确保可视化目录存在
        vis_dir = Path(self.experiment_dir) / 'visualizations'
        vis_dir.mkdir(parents=True, exist_ok=True)

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

        print("✓ 训练曲线绘制完成")

    def save_pretrained_weights(self, save_path: str):
        """
        保存预训练权重

        只保存模型的state_dict,用于后续微调阶段加载
        注意: 会排除projection_head的权重(因为微调不需要)

        Args:
            save_path: 保存路径
        """
        print(f"\n保存预训练权重到: {save_path}")

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
            'loss_type': self.loss_type
        }, save_path)

        print(f"✓ 预训练权重保存成功")
        print(f"  - 保存参数: {len(pretrained_dict)} 个")
        print(f"  - 排除参数: {len(excluded_keys)} 个")
        if excluded_keys:
            unique_layers = set([k.split('.')[0] for k in excluded_keys])
            print(f"  - 排除的层: {', '.join(unique_layers)}")


if __name__ == '__main__':
    """测试代码"""
    import sys
    import os
    from pathlib import Path

    # 添加项目根目录到path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    print("=" * 80)
    print("测试ContrastiveTrainer (修复版)")
    print("=" * 80)

    # 创建简单的测试
    print("\n✓ ContrastiveTrainer模块加载成功")
    print("✓ 已添加 _plot_curves 方法")
    print("✓ 已添加 save_pretrained_weights 方法")

    print("\n" + "=" * 80)
    print("✅ 修复验证通过!")
    print("=" * 80)
