"""
对比学习数据集
为对比学习预训练提供数据，每个样本返回两个不同的增强视图
支持NT-Xent（无监督）和SupCon（有监督）两种模式
"""
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import KFold
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessors import FFTTransform, STFTTransform, CWTTransform, IdentityTransform
from augmentation.augmentation_pipeline import ContrastiveAugmentation


class ContrastiveDataset(Dataset):
    """
    对比学习数据集
    
    关键特性：
    1. 每个样本生成两个独立的强增强视图
    2. 支持NT-Xent（无监督）和SupCon（有监督）
    3. 使用与BearingDataset相同的数据源和预处理
    """
    
    def __init__(self,
                 data_dir='raw_datasets/train',
                 window_size=512,
                 window_step=256,
                 mode='train',
                 fold=0,
                 n_folds=5,
                 timefreq_method='stft',
                 use_labels=False,  # True: SupCon, False: NT-Xent
                 augmentation=None,
                 cache_data=True):
        """
        Args:
            data_dir: 原始数据目录
            window_size: 窗口大小
            window_step: 滑动步长
            mode: 'train' 或 'val'
            fold: 当前fold编号
            n_folds: 总fold数
            timefreq_method: 时频变换方法
            use_labels: 是否使用标签（True=SupCon, False=NT-Xent）
            augmentation: 对比学习增强实例
            cache_data: 是否缓存数据
        """
        self.data_dir = data_dir
        self.window_size = window_size
        self.window_step = window_step
        self.mode = mode
        self.fold = fold
        self.n_folds = n_folds
        self.timefreq_method = timefreq_method
        self.use_labels = use_labels
        self.cache_data = cache_data

        # 故障类型映射
        self.fault_types = {
            'normal': 0,
            'inner_wear': 1,
            'inner_broken': 2,
            'roller_wear': 3,
            'roller_broken': 4,
            'outer_missing': 5
        }

        self.folder_mapping = {
            'normal': 'normal_train160',
            'inner_wear': 'inner_wear_train120',
            'inner_broken': 'inner_broken_train150',
            'roller_wear': 'roller_wear_train100',
            'roller_broken': 'roller_broken_train150',
            'outer_missing': 'outer_missing_train180'
        }

        # 创建信号变换
        self._create_transforms()

        # 创建对比学习增强
        if augmentation is not None:
            self.augmentation = augmentation
        else:
            # 使用默认的对比学习增强
            self.augmentation = ContrastiveAugmentation(strong_aug_prob=0.5)

        # 加载数据索引
        self.data_index = self._build_data_index()

        # k-fold划分
        self._split_kfold()

        # 缓存
        self.cache = {} if cache_data else None

        print(f"ContrastiveDataset initialized: mode={mode}, fold={fold}/{n_folds}, "
              f"samples={len(self.data_index)}, use_labels={use_labels}")

    def _create_transforms(self):
        """创建信号变换"""
        self.temporal_transform = IdentityTransform()
        self.frequency_transform = FFTTransform(use_log=True, normalize=True)

        if self.timefreq_method == 'stft':
            self.timefreq_transform = STFTTransform(
                n_fft=128,
                hop_length=32,
                target_size=(64, 128)
            )
        else:
            self.timefreq_transform = CWTTransform(
                scales=32,
                target_size=(64, 128)
            )

    def _build_data_index(self):
        """构建数据索引（与BearingDataset相同）"""
        data_index = []

        for fault_type, folder_name in self.folder_mapping.items():
            folder_path = os.path.join(self.data_dir, folder_name)

            if not os.path.exists(folder_path):
                print(f"Warning: {folder_path} not found, skipping...")
                continue

            # 遍历文件
            for filename in sorted(os.listdir(folder_path)):
                if not filename.endswith('.xlsx'):
                    continue

                file_path = os.path.join(folder_path, filename)

                # 读取文件长度
                df = pd.read_excel(file_path, header=None)
                signal_length = len(df)

                # 计算窗口数量
                num_windows = (signal_length - self.window_size) // self.window_step + 1

                # 为每个窗口创建索引
                for i in range(num_windows):
                    start_idx = i * self.window_step

                    data_index.append({
                        'file_path': file_path,
                        'start_idx': start_idx,
                        'label': self.fault_types[fault_type],
                        'fault_type': fault_type,
                        'file_id': f"{fault_type}_{filename}"
                    })

        return data_index

    def _split_kfold(self):
        """K-fold划分"""
        # 获取所有唯一的file_id
        file_ids = list(set([item['file_id'] for item in self.data_index]))

        # K-fold划分
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        splits = list(kf.split(file_ids))

        # 获取当前fold的file_ids
        if self.mode == 'train':
            train_idx, _ = splits[self.fold]
            current_file_ids = [file_ids[i] for i in train_idx]
        else:  # val
            _, val_idx = splits[self.fold]
            current_file_ids = [file_ids[i] for i in val_idx]

        # 过滤data_index
        self.data_index = [item for item in self.data_index
                           if item['file_id'] in current_file_ids]

    def _load_signal_window(self, item):
        """
        从Excel文件加载指定窗口的信号

        Args:
            item: 数据索引项
        Returns:
            signal: (window_size,) torch tensor
        """
        # 检查缓存
        cache_key = (item['file_path'], item['start_idx'])
        if self.cache is not None and cache_key in self.cache:
            return self.cache[cache_key].clone()

        # 读取Excel文件
        df = pd.read_excel(item['file_path'], header=None)

        # 第二列是垂直振动信号
        full_signal = df.iloc[:, 1].values.astype(np.float32)

        # 提取窗口
        start = item['start_idx']
        end = start + self.window_size
        window = full_signal[start:end]

        # 转换为tensor
        signal = torch.from_numpy(window).float()

        # 缓存
        if self.cache is not None:
            self.cache[cache_key] = signal.clone()

        return signal

    def _apply_transforms(self, signal):
        """
        应用三种变换

        Args:
            signal: (512,) tensor

        Returns:
            {
                'temporal': (1, 512),
                'frequency': (1, 257),
                'timefreq': (1, 64, 128)
            }
        """
        temporal = self.temporal_transform(signal).unsqueeze(0)      # (1, 512)
        frequency = self.frequency_transform(signal).unsqueeze(0)    # (1, 257)
        timefreq = self.timefreq_transform(signal).unsqueeze(0)      # (1, 64, 128)

        return {
            'temporal': temporal,
            'frequency': frequency,
            'timefreq': timefreq
        }

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        """
        返回对比学习样本

        Returns:
            如果use_labels=False (NT-Xent):
                {
                    'view1': {'temporal', 'frequency', 'timefreq'},
                    'view2': {'temporal', 'frequency', 'timefreq'}
                }

            如果use_labels=True (SupCon):
                {
                    'view1': {'temporal', 'frequency', 'timefreq'},
                    'view2': {'temporal', 'frequency', 'timefreq'},
                    'label': int
                }
        """
        item = self.data_index[idx]

        # 加载原始信号
        signal = self._load_signal_window(item)  # (512,)

        # 🔧 关键修复: 生成两个独立的增强视图
        # ContrastiveAugmentation.__call__会返回两个增强版本 (aug1, aug2)
        aug_signal1, aug_signal2 = self.augmentation(signal)

        # 对两个增强分别应用时频变换
        view1 = self._apply_transforms(aug_signal1)
        view2 = self._apply_transforms(aug_signal2)

        # 构建返回字典
        result = {
            'view1': view1,
            'view2': view2
        }

        # SupCon模式需要标签
        if self.use_labels:
            result['label'] = item['label']

        return result


def create_contrastive_dataloaders(data_dir='raw_datasets/train',
                                   batch_size=64,
                                   fold=0,
                                   n_folds=5,
                                   window_size=512,
                                   window_step=256,
                                   timefreq_method='stft',
                                   use_labels=False,
                                   num_workers=4):
    """
    创建对比学习DataLoader的工厂函数

    Args:
        data_dir: 数据目录
        batch_size: batch大小（对比学习通常用较大的batch）
        fold: 当前fold编号
        n_folds: 总fold数
        window_size: 窗口大小
        window_step: 滑动步长
        timefreq_method: 时频方法
        use_labels: 是否使用标签（SupCon=True, NT-Xent=False）
        num_workers: 数据加载worker数

    Returns:
        train_loader, val_loader
    """
    from torch.utils.data import DataLoader

    # 创建训练集
    train_dataset = ContrastiveDataset(
        data_dir=data_dir,
        window_size=window_size,
        window_step=window_step,
        mode='train',
        fold=fold,
        n_folds=n_folds,
        timefreq_method=timefreq_method,
        use_labels=use_labels,
        cache_data=True
    )

    # 创建验证集
    val_dataset = ContrastiveDataset(
        data_dir=data_dir,
        window_size=window_size,
        window_step=window_step,
        mode='val',
        fold=fold,
        n_folds=n_folds,
        timefreq_method=timefreq_method,
        use_labels=use_labels,
        cache_data=True
    )

    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # 对比学习建议drop最后不完整的batch
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


if __name__ == "__main__":
    """测试代码"""
    print("=" * 70)
    print("测试对比学习数据集")
    print("=" * 70)

    # 测试NT-Xent模式
    print("\n1. NT-Xent模式（无监督）")
    nt_xent_dataset = ContrastiveDataset(
        data_dir='raw_datasets/train',
        mode='train',
        fold=0,
        n_folds=5,
        use_labels=False
    )

    print(f"   数据集大小: {len(nt_xent_dataset)}")

    sample = nt_xent_dataset[0]
    print(f"   View1 temporal: {sample['view1']['temporal'].shape}")
    print(f"   View1 frequency: {sample['view1']['frequency'].shape}")
    print(f"   View1 timefreq: {sample['view1']['timefreq'].shape}")
    print(f"   View2 temporal: {sample['view2']['temporal'].shape}")
    print(f"   包含标签: {'label' in sample}")

    # 验证两个视图确实不同
    diff_temporal = (sample['view1']['temporal'] - sample['view2']['temporal']).abs().mean()
    print(f"   两个视图的差异: {diff_temporal:.4f} (应该>0)")

    # 测试SupCon模式
    print("\n2. SupCon模式（有监督）")
    supcon_dataset = ContrastiveDataset(
        data_dir='raw_datasets/train',
        mode='train',
        fold=0,
        n_folds=5,
        use_labels=True
    )

    sample = supcon_dataset[0]
    print(f"   数据集大小: {len(supcon_dataset)}")
    print(f"   包含标签: {'label' in sample}")
    if 'label' in sample:
        print(f"   Label: {sample['label']}")

    # 测试DataLoader
    print("\n3. 测试DataLoader")
    train_loader, val_loader = create_contrastive_dataloaders(
        batch_size=8,
        fold=0,
        use_labels=False,
        num_workers=0
    )

    batch = next(iter(train_loader))
    print(f"   Batch view1 temporal: {batch['view1']['temporal'].shape}")
    print(f"   Batch view2 temporal: {batch['view2']['temporal'].shape}")

    # 验证两个视图不同（增强效果）
    diff = (batch['view1']['temporal'] - batch['view2']['temporal']).abs().mean()
    print(f"   两个视图的平均差异: {diff:.4f} (应该>0)")

    print("\n✓ 对比学习数据集测试完成!")
