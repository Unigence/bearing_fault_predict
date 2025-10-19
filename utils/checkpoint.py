"""
Checkpoint Management
模型检查点的保存、加载和管理
"""

import os
import torch
import shutil
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
from datetime import datetime
import glob


class CheckpointManager:
    """检查点管理器"""
    
    def __init__(
        self,
        checkpoint_dir: str = './checkpoints',
        max_to_keep: Optional[int] = None,
        keep_best: bool = True
    ):
        """
        Args:
            checkpoint_dir: 检查点保存目录
            max_to_keep: 最多保留的检查点数量（None表示不限制）
            keep_best: 是否始终保留最佳检查点
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_to_keep = max_to_keep
        self.keep_best = keep_best
        
        # 检查点元信息文件
        self.meta_file = self.checkpoint_dir / 'checkpoints_meta.json'
        self.checkpoints_meta = self._load_meta()
    
    def _load_meta(self) -> Dict:
        """加载检查点元信息"""
        if self.meta_file.exists():
            with open(self.meta_file, 'r') as f:
                return json.load(f)
        return {'checkpoints': [], 'best_checkpoint': None}
    
    def _save_meta(self):
        """保存检查点元信息"""
        with open(self.meta_file, 'w') as f:
            json.dump(self.checkpoints_meta, f, indent=2)
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        extra_info: Optional[Dict[str, Any]] = None,
        checkpoint_name: Optional[str] = None,
        is_best: bool = False
    ) -> str:
        """
        保存检查点
        
        Args:
            model: 模型
            optimizer: 优化器
            scheduler: 学习率调度器
            epoch: 当前epoch
            metrics: 指标字典
            extra_info: 额外信息
            checkpoint_name: 检查点名称（如果为None，自动生成）
            is_best: 是否是最佳检查点
            
        Returns:
            checkpoint_path: 检查点路径
        """
        if checkpoint_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            checkpoint_name = f'checkpoint_epoch_{epoch:04d}_{timestamp}.pth'
        
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # 构建检查点
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'timestamp': datetime.now().isoformat(),
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        if metrics is not None:
            checkpoint['metrics'] = metrics
        
        if extra_info is not None:
            checkpoint['extra_info'] = extra_info
        
        # 保存检查点
        torch.save(checkpoint, checkpoint_path)
        
        # 更新元信息
        checkpoint_info = {
            'filename': checkpoint_name,
            'epoch': epoch,
            'timestamp': checkpoint['timestamp'],
            'metrics': metrics or {},
            'is_best': is_best
        }
        
        self.checkpoints_meta['checkpoints'].append(checkpoint_info)
        
        if is_best:
            self.checkpoints_meta['best_checkpoint'] = checkpoint_name
            # 同时保存一份best model
            best_path = self.checkpoint_dir / 'best_model.pth'
            shutil.copy2(checkpoint_path, best_path)
        
        self._save_meta()
        
        # 清理旧检查点
        self._cleanup_checkpoints()
        
        print(f"Checkpoint saved: {checkpoint_path}")
        if is_best:
            print(f"  → Best model updated!")
        
        return str(checkpoint_path)
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = 'cpu',
        strict: bool = True
    ) -> Dict[str, Any]:
        """
        加载检查点
        
        Args:
            checkpoint_path: 检查点路径（可以是相对路径或绝对路径）
            model: 模型
            optimizer: 优化器
            scheduler: 学习率调度器
            device: 设备
            strict: 是否严格匹配模型参数
            
        Returns:
            checkpoint: 检查点字典
        """
        # 如果是相对路径，添加checkpoint_dir
        if not os.path.isabs(checkpoint_path):
            checkpoint_path = self.checkpoint_dir / checkpoint_path
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 加载模型
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        
        # 加载优化器
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 加载调度器
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Checkpoint loaded: {checkpoint_path}")
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        if 'metrics' in checkpoint:
            print(f"  Metrics: {checkpoint['metrics']}")
        
        return checkpoint
    
    def load_best_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = 'cpu'
    ) -> Dict[str, Any]:
        """
        加载最佳检查点
        
        Args:
            model: 模型
            optimizer: 优化器
            scheduler: 学习率调度器
            device: 设备
            
        Returns:
            checkpoint: 检查点字典
        """
        best_checkpoint = self.checkpoints_meta.get('best_checkpoint')
        
        if best_checkpoint is None:
            # 尝试加载best_model.pth
            best_path = self.checkpoint_dir / 'best_model.pth'
            if best_path.exists():
                return self.load_checkpoint(
                    str(best_path), model, optimizer, scheduler, device
                )
            raise ValueError("No best checkpoint found")
        
        return self.load_checkpoint(
            best_checkpoint, model, optimizer, scheduler, device
        )
    
    def load_latest_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = 'cpu'
    ) -> Dict[str, Any]:
        """
        加载最新的检查点
        
        Args:
            model: 模型
            optimizer: 优化器
            scheduler: 学习率调度器
            device: 设备
            
        Returns:
            checkpoint: 检查点字典
        """
        checkpoints = self.checkpoints_meta.get('checkpoints', [])
        
        if not checkpoints:
            raise ValueError("No checkpoints found")
        
        # 按epoch排序，获取最新的
        latest = max(checkpoints, key=lambda x: x['epoch'])
        
        return self.load_checkpoint(
            latest['filename'], model, optimizer, scheduler, device
        )
    
    def get_best_checkpoint(self) -> Optional[str]:
        """
        获取最佳检查点的路径

        ⚠️ 新增方法：修复trainer_base.py中的调用

        Returns:
            best_checkpoint_path: 最佳检查点的完整路径，如果不存在返回None
        """
        best_checkpoint_name = self.checkpoints_meta.get('best_checkpoint')

        if best_checkpoint_name is None:
            # 尝试查找best_model.pth
            best_path = self.checkpoint_dir / 'best_model.pth'
            if best_path.exists():
                return str(best_path)
            return None

        best_path = self.checkpoint_dir / best_checkpoint_name
        return str(best_path) if best_path.exists() else None

    def _cleanup_checkpoints(self):
        """清理旧检查点，保留指定数量的最新检查点"""
        if self.max_to_keep is None:
            return

        checkpoints = self.checkpoints_meta.get('checkpoints', [])

        if len(checkpoints) <= self.max_to_keep:
            return

        # 按epoch排序
        checkpoints_sorted = sorted(checkpoints, key=lambda x: x['epoch'], reverse=True)

        # 确定要删除的检查点
        to_remove = checkpoints_sorted[self.max_to_keep:]

        for ckpt in to_remove:
            # 如果是最佳检查点且keep_best=True，跳过
            if self.keep_best and ckpt.get('is_best', False):
                continue

            # 删除文件
            ckpt_path = self.checkpoint_dir / ckpt['filename']
            if ckpt_path.exists():
                ckpt_path.unlink()
                print(f"Removed old checkpoint: {ckpt['filename']}")

            # 从元信息中移除
            checkpoints.remove(ckpt)

        self.checkpoints_meta['checkpoints'] = checkpoints
        self._save_meta()

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        列出所有检查点

        Returns:
            checkpoints: 检查点信息列表
        """
        return self.checkpoints_meta.get('checkpoints', [])

    def get_best_checkpoint_info(self) -> Optional[Dict[str, Any]]:
        """
        获取最佳检查点信息

        Returns:
            checkpoint_info: 检查点信息
        """
        best_name = self.checkpoints_meta.get('best_checkpoint')

        if best_name is None:
            return None

        for ckpt in self.checkpoints_meta.get('checkpoints', []):
            if ckpt['filename'] == best_name:
                return ckpt

        return None

    def remove_checkpoint(self, checkpoint_name: str):
        """
        删除指定检查点

        Args:
            checkpoint_name: 检查点名称
        """
        ckpt_path = self.checkpoint_dir / checkpoint_name

        if ckpt_path.exists():
            ckpt_path.unlink()
            print(f"Removed checkpoint: {checkpoint_name}")

        # 从元信息中移除
        checkpoints = self.checkpoints_meta.get('checkpoints', [])
        self.checkpoints_meta['checkpoints'] = [
            ckpt for ckpt in checkpoints if ckpt['filename'] != checkpoint_name
        ]
        self._save_meta()


if __name__ == '__main__':
    """测试代码"""
    import torch.nn as nn

    print("=" * 70)
    print("CheckpointManager测试")
    print("=" * 70)

    # 创建测试模型
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )

    # 创建checkpoint管理器
    manager = CheckpointManager(
        checkpoint_dir='./test_checkpoints',
        max_to_keep=3,
        keep_best=True
    )

    # 测试1: 保存检查点
    print("\n1. 测试保存检查点")
    manager.save_checkpoint(
        model=model,
        epoch=0,
        metrics={'loss': 1.5, 'acc': 0.6},
        is_best=False
    )
    manager.save_checkpoint(
        model=model,
        epoch=1,
        metrics={'loss': 1.2, 'acc': 0.7},
        is_best=True
    )
    manager.save_checkpoint(
        model=model,
        epoch=2,
        metrics={'loss': 1.0, 'acc': 0.75},
        is_best=False
    )

    # 测试2: 列出检查点
    print("\n2. 列出所有检查点")
    checkpoints = manager.list_checkpoints()
    for ckpt in checkpoints:
        print(f"  - Epoch {ckpt['epoch']}: {ckpt['filename']} (best={ckpt['is_best']})")

    # 测试3: 获取最佳检查点路径 (新增方法)
    print("\n3. 获取最佳检查点路径")
    best_path = manager.get_best_checkpoint()
    print(f"  Best checkpoint: {best_path}")

    # 测试4: 获取最佳检查点信息
    print("\n4. 获取最佳检查点信息")
    best_info = manager.get_best_checkpoint_info()
    if best_info:
        print(f"  Epoch: {best_info['epoch']}")
        print(f"  Metrics: {best_info['metrics']}")

    # 测试5: 加载最佳检查点
    print("\n5. 加载最佳检查点")
    try:
        checkpoint = manager.load_best_checkpoint(model, device='cpu')
        print(f"  ✓ 加载成功: Epoch {checkpoint['epoch']}")
    except Exception as e:
        print(f"  ✗ 加载失败: {e}")

    print("\n" + "=" * 70)
    print("✓ CheckpointManager测试完成")
    print("=" * 70)

    # 清理测试文件
    import shutil
    if Path('./test_checkpoints').exists():
        shutil.rmtree('./test_checkpoints')
        print("\n✓ 测试文件已清理")
