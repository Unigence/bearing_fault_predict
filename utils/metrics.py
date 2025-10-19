"""
Metrics Calculation
计算各种评估指标：准确率、精确率、召回率、F1分数、混淆矩阵等
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
        y_pred: torch.Tensor
    ) -> np.ndarray:
        """
        计算混淆矩阵
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            
        Returns:
            cm: 混淆矩阵 (num_classes, num_classes)
        """
        y_true = self._to_numpy(y_true)
        y_pred = self._to_numpy(y_pred)
        
        return confusion_matrix(y_true, y_pred, labels=range(self.num_classes))
    
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
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return classification_report(
                y_true, y_pred,
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
            y_true: 真实标签 (N,)
            y_pred_probs: 预测概率 (N, num_classes)
            k: K值
            
        Returns:
            top_k_acc: Top-K准确率
        """
        if isinstance(y_pred_probs, torch.Tensor):
            y_pred_probs = y_pred_probs.cpu()
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu()
        
        # 获取Top-K预测
        _, top_k_preds = torch.topk(y_pred_probs, k, dim=1)
        
        # 检查真实标签是否在Top-K中
        correct = torch.sum(top_k_preds == y_true.unsqueeze(1)).item()
        total = y_true.size(0)
        
        return correct / total
    
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
            mask = (y_true == cls)
            if mask.sum() > 0:
                acc = (y_pred[mask] == cls).sum() / mask.sum()
                per_class_acc[cls] = float(acc)
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
        计算所有常用指标
        
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
    print("Testing Metrics Calculator")
    print("=" * 60)
    
    # 创建虚拟数据
    num_samples = 100
    num_classes = 5
    
    # 模拟真实标签和预测
    y_true = torch.randint(0, num_classes, (num_samples,))
    y_pred = torch.randint(0, num_classes, (num_samples,))
    y_pred_probs = torch.randn(num_samples, num_classes).softmax(dim=1)
    
    # 创建计算器
    calculator = MetricsCalculator(num_classes=num_classes, average='macro')
    
    # 计算各种指标
    print("\n1. Basic Metrics:")
    print(f"Accuracy: {calculator.accuracy(y_true, y_pred):.4f}")
    print(f"Precision: {calculator.precision(y_true, y_pred):.4f}")
    print(f"Recall: {calculator.recall(y_true, y_pred):.4f}")
    print(f"F1 Score: {calculator.f1(y_true, y_pred):.4f}")
    
    # Top-K准确率
    print("\n2. Top-K Accuracy:")
    for k in [1, 3, 5]:
        top_k_acc = calculator.top_k_accuracy(y_true, y_pred_probs, k=k)
        print(f"Top-{k} Accuracy: {top_k_acc:.4f}")
    
    # 每个类别的准确率
    print("\n3. Per-Class Accuracy:")
    per_class_acc = calculator.per_class_accuracy(y_true, y_pred)
    for cls, acc in per_class_acc.items():
        print(f"Class {cls}: {acc:.4f}")
    
    # 混淆矩阵
    print("\n4. Confusion Matrix:")
    cm = calculator.confusion_matrix(y_true, y_pred)
    class_names = [f'C{i}' for i in range(num_classes)]
    print_confusion_matrix(cm, class_names=class_names)
    
    # 归一化混淆矩阵
    print("\n5. Normalized Confusion Matrix:")
    print_confusion_matrix(cm, class_names=class_names, normalize=True)
    
    # 分类报告
    print("\n6. Classification Report:")
    report = calculator.classification_report(y_true, y_pred, target_names=class_names)
    print(report)
    
    # 计算所有指标
    print("\n7. All Metrics:")
    all_metrics = calculator.compute_all_metrics(
        y_true, y_pred, y_pred_probs, prefix='test_'
    )
    for name, value in all_metrics.items():
        print(f"{name}: {value:.4f}")
    
    # 平衡准确率
    print("\n8. Balanced Accuracy:")
    balanced_acc = compute_balanced_accuracy(y_true, y_pred, num_classes)
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
