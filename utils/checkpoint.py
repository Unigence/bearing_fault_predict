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
            c for c in checkpoints if c['filename'] != checkpoint_name
        ]
        
        # 如果删除的是最佳检查点，清除best_checkpoint标记
        if self.checkpoints_meta.get('best_checkpoint') == checkpoint_name:
            self.checkpoints_meta['best_checkpoint'] = None
        
        self._save_meta()
    
    def clear_all_checkpoints(self, keep_best: bool = True):
        """
        清除所有检查点
        
        Args:
            keep_best: 是否保留最佳检查点
        """
        checkpoints = self.checkpoints_meta.get('checkpoints', [])
        best_name = self.checkpoints_meta.get('best_checkpoint')
        
        for ckpt in checkpoints:
            if keep_best and ckpt['filename'] == best_name:
                continue
            
            ckpt_path = self.checkpoint_dir / ckpt['filename']
            if ckpt_path.exists():
                ckpt_path.unlink()
        
        if keep_best and best_name:
            self.checkpoints_meta['checkpoints'] = [
                c for c in checkpoints if c['filename'] == best_name
            ]
        else:
            self.checkpoints_meta['checkpoints'] = []
            self.checkpoints_meta['best_checkpoint'] = None
        
        self._save_meta()
        print("All checkpoints cleared" + (" (except best)" if keep_best else ""))


def save_model(
    model: torch.nn.Module,
    filepath: str,
    save_full_model: bool = False
):
    """
    保存模型
    
    Args:
        model: 模型
        filepath: 保存路径
        save_full_model: 是否保存完整模型（包括结构），否则只保存参数
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if save_full_model:
        torch.save(model, filepath)
    else:
        torch.save(model.state_dict(), filepath)
    
    print(f"Model saved: {filepath}")


def load_model(
    filepath: str,
    model: Optional[torch.nn.Module] = None,
    device: str = 'cpu',
    strict: bool = True
) -> torch.nn.Module:
    """
    加载模型
    
    Args:
        filepath: 模型路径
        model: 模型实例（如果为None，则加载完整模型）
        device: 设备
        strict: 是否严格匹配参数
        
    Returns:
        model: 加载后的模型
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    if model is None:
        # 加载完整模型
        model = torch.load(filepath, map_location=device)
    else:
        # 只加载参数
        state_dict = torch.load(filepath, map_location=device)
        model.load_state_dict(state_dict, strict=strict)
    
    print(f"Model loaded: {filepath}")
    return model


if __name__ == '__main__':
    # 测试检查点管理器
    import torch.nn as nn
    
    # 创建简单模型
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 创建检查点管理器
    manager = CheckpointManager(
        checkpoint_dir='./test_checkpoints',
        max_to_keep=3,
        keep_best=True
    )
    
    # 保存几个检查点
    print("Saving checkpoints:")
    for epoch in range(5):
        metrics = {'loss': 1.0 / (epoch + 1), 'acc': 0.5 + epoch * 0.1}
        is_best = (epoch == 2)  # 假设第3个epoch是最佳的
        
        manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=metrics,
            is_best=is_best
        )
    
    # 列出所有检查点
    print("\n\nAll checkpoints:")
    for ckpt in manager.list_checkpoints():
        print(f"  {ckpt}")
    
    # 获取最佳检查点信息
    print("\n\nBest checkpoint:")
    best_info = manager.get_best_checkpoint_info()
    print(f"  {best_info}")
    
    # 加载最佳检查点
    print("\n\nLoading best checkpoint:")
    checkpoint = manager.load_best_checkpoint(model, optimizer)
    print(f"Loaded epoch: {checkpoint['epoch']}")
    print(f"Metrics: {checkpoint['metrics']}")
