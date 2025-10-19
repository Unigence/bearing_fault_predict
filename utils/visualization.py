"""
Visualization Utilities
可视化训练过程、指标、混淆矩阵等
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import os
from pathlib import Path


# 设置matplotlib风格
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class TrainingVisualizer:
    """训练过程可视化器"""
    
    def __init__(self, save_dir: str = './plots', dpi: int = 100):
        """
        Args:
            save_dir: 图片保存目录
            dpi: 图片分辨率
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
    
    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        metrics: Optional[List[str]] = None,
        title: str = 'Training History',
        figsize: Tuple[int, int] = (12, 8),
        save_name: Optional[str] = None,
        show: bool = True
    ):
        """
        绘制训练历史曲线
        
        Args:
            history: 训练历史字典，包含各个指标的列表
            metrics: 要绘制的指标名称列表（如果为None，绘制所有指标）
            title: 图表标题
            figsize: 图表大小
            save_name: 保存文件名
            show: 是否显示图表
        """
        if metrics is None:
            metrics = list(history.keys())
        
        # 过滤掉不存在的指标
        metrics = [m for m in metrics if m in history and len(history[m]) > 0]
        
        if not metrics:
            print("No metrics to plot")
            return
        
        # 计算子图布局
        n_metrics = len(metrics)
        n_cols = min(2, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        
        if n_metrics == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_metrics > 1 else [axes]
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            values = history[metric]
            epochs = range(1, len(values) + 1)
            
            ax.plot(epochs, values, marker='o', linewidth=2, markersize=4)
            ax.set_xlabel('Epoch', fontsize=10)
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=10)
            ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for idx in range(len(metrics), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # 保存图表
        if save_name:
            filepath = self.save_dir / save_name
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"Plot saved: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_loss_curves(
        self,
        train_loss: List[float],
        val_loss: List[float],
        title: str = 'Training and Validation Loss',
        figsize: Tuple[int, int] = (10, 6),
        save_name: Optional[str] = None,
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """
        绘制训练和验证损失曲线
        
        Args:
            train_loss: 训练损失列表
            val_loss: 验证损失列表
            title: 图表标题
            figsize: 图表大小
            save_name: 保存文件名（优先使用save_path）
            save_path: 完整保存路径
            show: 是否显示图表
        """
        epochs = range(1, len(train_loss) + 1)
        
        plt.figure(figsize=figsize)
        plt.plot(epochs, train_loss, 'b-o', label='Train Loss', linewidth=2, markersize=4)
        plt.plot(epochs, val_loss, 'r-s', label='Val Loss', linewidth=2, markersize=4)
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Plot saved: {save_path}")
        elif save_name:
            filepath = self.save_dir / save_name
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"Plot saved: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_accuracy_curves(
        self,
        train_acc: List[float],
        val_acc: List[float],
        title: str = 'Training and Validation Accuracy',
        figsize: Tuple[int, int] = (10, 6),
        save_name: Optional[str] = None,
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """
        绘制训练和验证准确率曲线
        
        Args:
            train_acc: 训练准确率列表
            val_acc: 验证准确率列表
            title: 图表标题
            figsize: 图表大小
            save_name: 保存文件名（优先使用save_path）
            save_path: 完整保存路径
            show: 是否显示图表
        """
        epochs = range(1, len(train_acc) + 1)
        
        plt.figure(figsize=figsize)
        plt.plot(epochs, train_acc, 'b-o', label='Train Accuracy', linewidth=2, markersize=4)
        plt.plot(epochs, val_acc, 'r-s', label='Val Accuracy', linewidth=2, markersize=4)
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1])
        
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Plot saved: {save_path}")
        elif save_name:
            filepath = self.save_dir / save_name
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"Plot saved: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_learning_rate(
        self,
        learning_rates: List[float],
        title: str = 'Learning Rate Schedule',
        figsize: Tuple[int, int] = (10, 6),
        save_name: Optional[str] = None,
        show: bool = True
    ):
        """
        绘制学习率变化曲线
        
        Args:
            learning_rates: 学习率列表
            title: 图表标题
            figsize: 图表大小
            save_name: 保存文件名
            show: 是否显示图表
        """
        epochs = range(1, len(learning_rates) + 1)
        
        plt.figure(figsize=figsize)
        plt.plot(epochs, learning_rates, 'g-o', linewidth=2, markersize=4)
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Learning Rate', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # 使用对数刻度
        
        plt.tight_layout()
        
        # 保存图表
        if save_name:
            filepath = self.save_dir / save_name
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"Plot saved: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: Optional[List[str]] = None,
        title: str = 'Confusion Matrix',
        figsize: Tuple[int, int] = (10, 8),
        normalize: bool = False,
        save_name: Optional[str] = None,
        show: bool = True
    ):
        """
        绘制混淆矩阵热力图
        
        Args:
            cm: 混淆矩阵
            class_names: 类别名称列表
            title: 图表标题
            figsize: 图表大小
            normalize: 是否归一化
            save_name: 保存文件名
            show: 是否显示图表
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
            fmt = '.2f'
        else:
            fmt = 'd'
        
        num_classes = cm.shape[0]
        
        if class_names is None:
            class_names = [f'Class {i}' for i in range(num_classes)]
        
        plt.figure(figsize=figsize)
        
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
        )
        
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # 保存图表
        if save_name:
            filepath = self.save_dir / save_name
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"Plot saved: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_metrics_comparison(
        self,
        metrics_dict: Dict[str, Dict[str, float]],
        title: str = 'Metrics Comparison',
        figsize: Tuple[int, int] = (12, 6),
        save_name: Optional[str] = None,
        show: bool = True
    ):
        """
        绘制多个模型/实验的指标对比
        
        Args:
            metrics_dict: 嵌套字典，格式为 {model_name: {metric_name: value}}
            title: 图表标题
            figsize: 图表大小
            save_name: 保存文件名
            show: 是否显示图表
        """
        # 提取数据
        models = list(metrics_dict.keys())
        metric_names = list(next(iter(metrics_dict.values())).keys())
        
        # 准备数据
        data = {metric: [metrics_dict[model][metric] for model in models]
                for metric in metric_names}
        
        # 绘制柱状图
        x = np.arange(len(models))
        width = 0.8 / len(metric_names)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        for i, metric in enumerate(metric_names):
            offset = width * i - width * (len(metric_names) - 1) / 2
            ax.bar(x + offset, data[metric], width, label=metric)
        
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # 保存图表
        if save_name:
            filepath = self.save_dir / save_name
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"Plot saved: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_per_class_metrics(
        self,
        per_class_metrics: Dict[int, float],
        class_names: Optional[List[str]] = None,
        metric_name: str = 'Accuracy',
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6),
        save_name: Optional[str] = None,
        show: bool = True
    ):
        """
        绘制每个类别的指标
        
        Args:
            per_class_metrics: 每个类别的指标字典
            class_names: 类别名称列表
            metric_name: 指标名称
            title: 图表标题
            figsize: 图表大小
            save_name: 保存文件名
            show: 是否显示图表
        """
        classes = sorted(per_class_metrics.keys())
        values = [per_class_metrics[c] for c in classes]
        
        if class_names is None:
            class_names = [f'Class {c}' for c in classes]
        
        if title is None:
            title = f'Per-Class {metric_name}'
        
        plt.figure(figsize=figsize)
        
        bars = plt.bar(range(len(classes)), values, color='skyblue', edgecolor='navy')
        
        # 为每个柱子添加数值标签
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{val:.3f}',
                ha='center',
                va='bottom',
                fontsize=9
            )
        
        plt.xlabel('Class', fontsize=12)
        plt.ylabel(metric_name, fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xticks(range(len(classes)), class_names, rotation=45, ha='right')
        plt.ylim([0, max(values) * 1.1])
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # 保存图表
        if save_name:
            filepath = self.save_dir / save_name
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"Plot saved: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()


def plot_embeddings_2d(
    embeddings: np.ndarray,
    labels: Optional[np.ndarray] = None,
    method: str = 'tsne',
    title: str = '2D Embedding Visualization',
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    绘制2D嵌入可视化（使用t-SNE或PCA）
    
    Args:
        embeddings: 嵌入向量 (N, D)
        labels: 标签 (N,)
        method: 降维方法 ('tsne' 或 'pca')
        title: 图表标题
        figsize: 图表大小
        save_path: 保存路径
        show: 是否显示图表
    """
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    
    # 降维到2D
    if method.lower() == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    elif method.lower() == 'pca':
        reducer = PCA(n_components=2)
    else:
        raise ValueError(f"Unsupported method: {method}")
    
    embeddings_2d = reducer.fit_transform(embeddings)
    
    # 绘制
    plt.figure(figsize=figsize)
    
    if labels is not None:
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=[colors[i]],
                label=f'Class {label}',
                alpha=0.6,
                s=50
            )
        
        plt.legend(fontsize=10)
    else:
        plt.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            alpha=0.6,
            s=50
        )
    
    plt.xlabel(f'{method.upper()} Component 1', fontsize=12)
    plt.ylabel(f'{method.upper()} Component 2', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Plot saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


if __name__ == '__main__':
    # 测试可视化
    print("Testing Visualization")
    print("=" * 60)
    
    # 创建可视化器
    visualizer = TrainingVisualizer(save_dir='./test_plots')
    
    # 模拟训练历史
    history = {
        'train_loss': [2.3, 1.8, 1.5, 1.2, 1.0, 0.9, 0.8, 0.75, 0.7, 0.68],
        'val_loss': [2.2, 1.9, 1.6, 1.4, 1.3, 1.2, 1.15, 1.1, 1.08, 1.05],
        'train_acc': [0.3, 0.5, 0.6, 0.7, 0.75, 0.78, 0.82, 0.85, 0.87, 0.88],
        'val_acc': [0.35, 0.48, 0.58, 0.65, 0.68, 0.70, 0.72, 0.73, 0.74, 0.75],
        'learning_rate': [1e-3, 1e-3, 5e-4, 5e-4, 1e-4, 1e-4, 5e-5, 5e-5, 1e-5, 1e-5]
    }
    
    # 1. 绘制训练历史
    print("\n1. Plotting training history...")
    visualizer.plot_training_history(
        history,
        title='Training History',
        save_name='training_history.png',
        show=False
    )
    
    # 2. 绘制损失曲线
    print("\n2. Plotting loss curves...")
    visualizer.plot_loss_curves(
        history['train_loss'],
        history['val_loss'],
        save_name='loss_curves.png',
        show=False
    )
    
    # 3. 绘制准确率曲线
    print("\n3. Plotting accuracy curves...")
    visualizer.plot_accuracy_curves(
        history['train_acc'],
        history['val_acc'],
        save_name='accuracy_curves.png',
        show=False
    )
    
    # 4. 绘制学习率
    print("\n4. Plotting learning rate...")
    visualizer.plot_learning_rate(
        history['learning_rate'],
        save_name='learning_rate.png',
        show=False
    )
    
    # 5. 绘制混淆矩阵
    print("\n5. Plotting confusion matrix...")
    cm = np.array([
        [50, 2, 3, 1, 0],
        [3, 45, 2, 5, 1],
        [1, 2, 48, 3, 2],
        [2, 4, 3, 47, 0],
        [0, 1, 2, 1, 52]
    ])
    class_names = ['Cat', 'Dog', 'Bird', 'Fish', 'Rabbit']
    visualizer.plot_confusion_matrix(
        cm,
        class_names=class_names,
        save_name='confusion_matrix.png',
        show=False
    )
    
    # 6. 绘制每个类别的指标
    print("\n6. Plotting per-class metrics...")
    per_class_acc = {0: 0.89, 1: 0.80, 2: 0.86, 3: 0.84, 4: 0.93}
    visualizer.plot_per_class_metrics(
        per_class_acc,
        class_names=class_names,
        metric_name='Accuracy',
        save_name='per_class_accuracy.png',
        show=False
    )
    
    print("\n✓ All plots saved to ./test_plots/")
    print("=" * 60)
