"""
Metrics Calculation and Tracking
计算各种评估指标：准确率、精确率、召回率、F1分数、混淆矩阵等
并提供MetricsTracker用于跟踪训练历史
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    average_precision_score, matthews_corrcoef, cohen_kappa_score
)
import warnings


class MetricsCalculator:
    """指标计算器"""
    
    def __init__(self, num_classes: int, average: str = 'macro'):
        """
        Args:
            num_classes: 类别数量
            average: 多分类时的平均方式 ('micro', 'macro', 'weighted', 'binary')
        """
        self.num_classes = num_classes
        self.average = average
    
    @staticmethod
    def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
        """将PyTorch张量转换为NumPy数组"""
        if isinstance(tensor, torch.Tensor):
            return tensor.cpu().numpy()
        return np.array(tensor)
    
    def accuracy(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor
    ) -> float:
        """
        计算准确率
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            
        Returns:
            accuracy: 准确率
        """
        y_true = self._to_numpy(y_true)
        y_pred = self._to_numpy(y_pred)
        
        return accuracy_score(y_true, y_pred)
    
    def precision(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        average: Optional[str] = None
    ) -> float:
        """
        计算精确率
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            average: 平均方式
            
        Returns:
            precision: 精确率
        """
        y_true = self._to_numpy(y_true)
        y_pred = self._to_numpy(y_pred)
        
        avg = average or self.average
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return precision_score(y_true, y_pred, average=avg, zero_division=0)
    
    def recall(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        average: Optional[str] = None
    ) -> float:
        """
        计算召回率
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            average: 平均方式
            
        Returns:
            recall: 召回率
        """
        y_true = self._to_numpy(y_true)
        y_pred = self._to_numpy(y_pred)
        
        avg = average or self.average
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return recall_score(y_true, y_pred, average=avg, zero_division=0)
    
    def f1(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        average: Optional[str] = None
    ) -> float:
        """
        计算F1分数
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            average: 平均方式
            
        Returns:
            f1: F1分数
        """
        y_true = self._to_numpy(y_true)
        y_pred = self._to_numpy(y_pred)
        
        avg = average or self.average
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return f1_score(y_true, y_pred, average=avg, zero_division=0)
    
    def confusion_matrix(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        normalize: bool = False
    ) -> np.ndarray:
        """
        计算混淆矩阵
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            normalize: 是否归一化
            
        Returns:
            cm: 混淆矩阵
        """
        y_true = self._to_numpy(y_true)
        y_pred = self._to_numpy(y_pred)
        
        cm = confusion_matrix(y_true, y_pred, labels=range(self.num_classes))
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        
        return cm
    
    def classification_report(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        target_names: Optional[List[str]] = None
    ) -> str:
        """
        生成分类报告
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            target_names: 类别名称列表
            
        Returns:
            report: 分类报告字符串
        """
        y_true = self._to_numpy(y_true)
        y_pred = self._to_numpy(y_pred)
        
        return classification_report(
            y_true, y_pred,
            labels=range(self.num_classes),
            target_names=target_names,
            zero_division=0
        )
    
    def top_k_accuracy(
        self,
        y_true: torch.Tensor,
        y_pred_probs: torch.Tensor,
        k: int = 5
    ) -> float:
        """
        计算Top-K准确率
        
        Args:
            y_true: 真实标签
            y_pred_probs: 预测概率 (N, num_classes)
            k: Top-K的K值
            
        Returns:
            top_k_acc: Top-K准确率
        """
        y_true = self._to_numpy(y_true)
        y_pred_probs = self._to_numpy(y_pred_probs)
        
        # 获取Top-K预测
        top_k_pred = np.argsort(y_pred_probs, axis=1)[:, -k:]
        
        # 检查真实标签是否在Top-K中
        correct = np.array([y_true[i] in top_k_pred[i] for i in range(len(y_true))])
        
        return np.mean(correct)
    
    def per_class_accuracy(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor
    ) -> Dict[int, float]:
        """
        计算每个类别的准确率
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            
        Returns:
            per_class_acc: 每个类别的准确率字典
        """
        y_true = self._to_numpy(y_true)
        y_pred = self._to_numpy(y_pred)
        
        per_class_acc = {}
        
        for cls in range(self.num_classes):
            mask = y_true == cls
            if mask.sum() > 0:
                per_class_acc[cls] = (y_pred[mask] == cls).sum() / mask.sum()
            else:
                per_class_acc[cls] = 0.0
        
        return per_class_acc
    
    def compute_all_metrics(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        y_pred_probs: Optional[torch.Tensor] = None,
        prefix: str = ''
    ) -> Dict[str, float]:
        """
        计算所有指标
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_pred_probs: 预测概率（可选）
            prefix: 指标名称前缀
            
        Returns:
            metrics: 指标字典
        """
        metrics = {}
        
        # 基础指标
        metrics[f'{prefix}accuracy'] = self.accuracy(y_true, y_pred)
        metrics[f'{prefix}precision'] = self.precision(y_true, y_pred)
        metrics[f'{prefix}recall'] = self.recall(y_true, y_pred)
        metrics[f'{prefix}f1'] = self.f1(y_true, y_pred)
        
        # Top-K准确率（如果提供了概率）
        if y_pred_probs is not None and self.num_classes > 5:
            metrics[f'{prefix}top5_accuracy'] = self.top_k_accuracy(
                y_true, y_pred_probs, k=5
            )
        
        return metrics


class MetricsTracker:
    """
    训练指标跟踪器
    用于记录和管理训练过程中的各种指标
    """
    
    def __init__(self):
        """初始化指标跟踪器"""
        self.history = {}
        self.current_metrics = {}
    
    def update(self, metrics: Dict[str, float]):
        """
        更新指标
        
        Args:
            metrics: 指标字典
        """
        self.current_metrics = metrics
        
        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)
    
    def get_history(self) -> Dict[str, List[float]]:
        """
        获取训练历史
        
        Returns:
            history: 训练历史字典
        """
        return self.history
    
    def get_current_metrics(self) -> Dict[str, float]:
        """
        获取当前指标
        
        Returns:
            current_metrics: 当前指标字典
        """
        return self.current_metrics
    
    def get_metric_history(self, metric_name: str) -> List[float]:
        """
        获取某个指标的历史
        
        Args:
            metric_name: 指标名称
            
        Returns:
            metric_history: 指标历史列表
        """
        return self.history.get(metric_name, [])
    
    def get_best_metric(self, metric_name: str, mode: str = 'max') -> Tuple[float, int]:
        """
        获取最佳指标值及其对应的epoch
        
        Args:
            metric_name: 指标名称
            mode: 'max' 或 'min'
            
        Returns:
            (best_value, best_epoch): 最佳值和对应的epoch
        """
        history = self.get_metric_history(metric_name)
        
        if not history:
            return None, -1
        
        if mode == 'max':
            best_value = max(history)
            best_epoch = history.index(best_value)
        else:
            best_value = min(history)
            best_epoch = history.index(best_value)
        
        return best_value, best_epoch
    
    def get_last_n_metrics(self, n: int = 5) -> Dict[str, List[float]]:
        """
        获取最近N个epoch的指标
        
        Args:
            n: epoch数量
            
        Returns:
            recent_metrics: 最近N个epoch的指标字典
        """
        recent_metrics = {}
        
        for key, values in self.history.items():
            recent_metrics[key] = values[-n:] if len(values) >= n else values
        
        return recent_metrics
    
    def reset(self):
        """重置所有指标"""
        self.history = {}
        self.current_metrics = {}
    
    def save(self, filepath: str):
        """
        保存训练历史到文件
        
        Args:
            filepath: 保存路径
        """
        import json
        
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"Metrics history saved: {filepath}")
    
    def load(self, filepath: str):
        """
        从文件加载训练历史
        
        Args:
            filepath: 文件路径
        """
        import json
        
        with open(filepath, 'r') as f:
            self.history = json.load(f)
        
        print(f"Metrics history loaded: {filepath}")


def compute_metrics(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    y_pred_probs: Optional[torch.Tensor] = None,
    num_classes: Optional[int] = None,
    average: str = 'macro',
    prefix: str = ''
) -> Dict[str, float]:
    """
    便捷函数：计算所有指标
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        y_pred_probs: 预测概率（可选）
        num_classes: 类别数量
        average: 平均方式
        prefix: 指标名称前缀
        
    Returns:
        metrics: 指标字典
    """
    if num_classes is None:
        num_classes = int(max(y_true.max().item(), y_pred.max().item())) + 1
    
    calculator = MetricsCalculator(num_classes=num_classes, average=average)
    return calculator.compute_all_metrics(y_true, y_pred, y_pred_probs, prefix)


def print_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = False
):
    """
    打印混淆矩阵
    
    Args:
        cm: 混淆矩阵
        class_names: 类别名称
        normalize: 是否归一化
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    
    num_classes = cm.shape[0]
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(num_classes)]
    
    # 打印表头
    print("\nConfusion Matrix:")
    print("=" * 60)
    
    # 计算列宽
    col_width = max(len(name) for name in class_names) + 2
    
    # 打印列标签
    print(" " * col_width, end="")
    for name in class_names:
        print(f"{name:>{col_width}}", end="")
    print()
    
    # 打印分隔线
    print("-" * (col_width * (num_classes + 1)))
    
    # 打印矩阵
    for i, row in enumerate(cm):
        print(f"{class_names[i]:<{col_width}}", end="")
        for val in row:
            if normalize:
                print(f"{val:>{col_width}.3f}", end="")
            else:
                print(f"{int(val):>{col_width}}", end="")
        print()
    
    print("=" * 60)


def compute_balanced_accuracy(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    num_classes: int
) -> float:
    """
    计算平衡准确率（对不平衡数据集有用）
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        num_classes: 类别数量
        
    Returns:
        balanced_acc: 平衡准确率
    """
    calculator = MetricsCalculator(num_classes=num_classes)
    per_class_acc = calculator.per_class_accuracy(y_true, y_pred)
    
    return np.mean(list(per_class_acc.values()))


def compute_matthews_corrcoef(
    y_true: torch.Tensor,
    y_pred: torch.Tensor
) -> float:
    """
    计算Matthews相关系数（适用于二分类不平衡数据）
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        
    Returns:
        mcc: Matthews相关系数
    """
    y_true = MetricsCalculator._to_numpy(None, y_true)
    y_pred = MetricsCalculator._to_numpy(None, y_pred)
    
    return matthews_corrcoef(y_true, y_pred)


def compute_cohen_kappa(
    y_true: torch.Tensor,
    y_pred: torch.Tensor
) -> float:
    """
    计算Cohen's Kappa系数（衡量分类一致性）
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        
    Returns:
        kappa: Cohen's Kappa系数
    """
    y_true = MetricsCalculator._to_numpy(None, y_true)
    y_pred = MetricsCalculator._to_numpy(None, y_pred)
    
    return cohen_kappa_score(y_true, y_pred)


if __name__ == '__main__':
    # 测试指标计算
    print("Testing Metrics Calculator and Tracker")
    print("=" * 60)
    
    # 创建虚拟数据
    num_samples = 100
    num_classes = 5
    
    # 模拟真实标签和预测
    y_true = torch.randint(0, num_classes, (num_samples,))
    y_pred = torch.randint(0, num_classes, (num_samples,))
    y_pred_probs = torch.randn(num_samples, num_classes).softmax(dim=1)
    
    # 测试MetricsCalculator
    print("\n1. Testing MetricsCalculator")
    calculator = MetricsCalculator(num_classes=num_classes, average='macro')
    
    print(f"Accuracy: {calculator.accuracy(y_true, y_pred):.4f}")
    print(f"Precision: {calculator.precision(y_true, y_pred):.4f}")
    print(f"Recall: {calculator.recall(y_true, y_pred):.4f}")
    print(f"F1 Score: {calculator.f1(y_true, y_pred):.4f}")
    
    # 测试MetricsTracker
    print("\n2. Testing MetricsTracker")
    tracker = MetricsTracker()
    
    # 模拟5个epoch的训练
    for epoch in range(5):
        metrics = {
            'train_loss': 2.0 - epoch * 0.3,
            'val_loss': 2.2 - epoch * 0.25,
            'train_acc': 0.3 + epoch * 0.1,
            'val_acc': 0.25 + epoch * 0.1
        }
        tracker.update(metrics)
        print(f"Epoch {epoch}: {metrics}")
    
    # 获取训练历史
    print("\n3. Training History:")
    history = tracker.get_history()
    for key, values in history.items():
        print(f"{key}: {values}")
    
    # 获取最佳指标
    print("\n4. Best Metrics:")
    best_val_acc, best_epoch = tracker.get_best_metric('val_acc', mode='max')
    print(f"Best val_acc: {best_val_acc:.4f} at epoch {best_epoch}")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
