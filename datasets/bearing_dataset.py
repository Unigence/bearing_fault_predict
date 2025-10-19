"""
轴承故障诊断数据集
支持k-fold交叉验证、动态数据切分、三分支输入
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


class BearingDataset(Dataset):
    """
    轴承故障诊断数据集
    
    架构:
    1. 从raw_datasets读取1024长度的原始信号
    2. 动态切分为512长度的窗口（滑动窗口）
    3. 支持k-fold交叉验证
    4. 应用数据增强（可选）
    5. 生成三分支输入：时域、频域、时频域
    """
    
    def __init__(self,
                 data_dir='raw_datasets/train',
                 window_size=512,
                 window_step=256,
                 mode='train',
                 fold=0,
                 n_folds=5,
                 timefreq_method='stft',  # 'stft' or 'cwt'
                 augmentation=None,
                 cache_data=True):
        """
        Args:
            data_dir: 原始数据目录
            window_size: 窗口大小（训练使用512）
            window_step: 滑动步长
            mode: 'train' 或 'val'
            fold: 当前fold编号 (0 to n_folds-1)
            n_folds: 总fold数
            timefreq_method: 时频变换方法 ('stft' 或 'cwt')
            augmentation: 数据增强pipeline
            cache_data: 是否缓存数据到内存
        """
        self.data_dir = data_dir
        self.window_size = window_size
        self.window_step = window_step
        self.mode = mode
        self.fold = fold
        self.n_folds = n_folds
        self.timefreq_method = timefreq_method
        self.augmentation = augmentation
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
        
        # 文件夹映射
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
        
        # 加载数据索引
        self.data_index = self._build_data_index()
        
        # k-fold划分
        self._split_kfold()
        
        # 缓存
        self.cache = {} if cache_data else None
        
        print(f"Dataset initialized: mode={mode}, fold={fold}/{n_folds}, "
              f"samples={len(self.data_index)}")
    
    def _create_transforms(self):
        """创建信号变换"""
        # 时域：恒等变换
        self.temporal_transform = IdentityTransform()
        
        # 频域：FFT变换
        self.frequency_transform = FFTTransform(
            use_log=True,
            normalize=True
        )
        
        # 时频域：STFT或CWT
        if self.timefreq_method == 'stft':
            self.timefreq_transform = STFTTransform(
                n_fft=128,
                hop_length=32,
                target_size=(64, 128)
            )
        elif self.timefreq_method == 'cwt':
            self.timefreq_transform = CWTTransform(
                scales=32,
                target_size=(64, 128)
            )
        else:
            raise ValueError(f"Unknown timefreq method: {self.timefreq_method}")
    
    def _build_data_index(self):
        """
        构建数据索引
        每个索引项包含：(file_path, start_idx, fault_type, label)
        """
        data_index = []
        
        for fault_name, label in self.fault_types.items():
            folder_name = self.folder_mapping[fault_name]
            folder_path = os.path.join(self.data_dir, folder_name)
            
            if not os.path.exists(folder_path):
                print(f"Warning: folder {folder_path} not found")
                continue
            
            # 获取所有Excel文件
            excel_files = sorted([f for f in os.listdir(folder_path) 
                                 if f.endswith('.xlsx')])
            
            for excel_file in excel_files:
                file_path = os.path.join(folder_path, excel_file)
                
                # 读取文件长度（假设都是1024）
                # 实际应该读取文件获取长度，这里简化
                signal_length = 1024
                
                # 计算可以切分的窗口数
                n_windows = (signal_length - self.window_size) // self.window_step + 1
                
                # 为每个窗口创建索引
                for i in range(n_windows):
                    start_idx = i * self.window_step
                    data_index.append({
                        'file_path': file_path,
                        'start_idx': start_idx,
                        'fault_type': fault_name,
                        'label': label,
                        'file_id': excel_file  # 用于k-fold划分
                    })
        
        return data_index
    
    def _split_kfold(self):
        """
        k-fold划分
        关键：确保同一个文件的所有窗口在同一个fold中
        """
        # 获取所有唯一的file_id
        file_ids = list(set([item['file_id'] for item in self.data_index]))
        file_ids.sort()
        
        # k-fold划分file_ids
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
    
    def __len__(self):
        return len(self.data_index)
    
    def __getitem__(self, idx):
        """
        返回一个样本
        
        Returns:
            {
                'temporal': (1, 512) 时域输入,
                'frequency': (1, 257) 频域输入,
                'timefreq': (1, 64, 128) 时频输入,
                'label': int 标签,
                'fault_type': str 故障类型名称
            }
        """
        item = self.data_index[idx]
        
        # 加载原始信号
        signal = self._load_signal_window(item)  # (512,)
        
        # 数据增强（仅训练时）
        if self.mode == 'train' and self.augmentation is not None:
            signal = self.augmentation(signal)
        
        # 应用三种变换
        temporal = self.temporal_transform(signal)      # (512,)
        frequency = self.frequency_transform(signal)    # (257,)
        timefreq = self.timefreq_transform(signal)      # (64, 128)
        
        # 添加通道维度
        temporal = temporal.unsqueeze(0)    # (1, 512)
        frequency = frequency.unsqueeze(0)  # (1, 257)
        timefreq = timefreq.unsqueeze(0)    # (1, 64, 128)
        
        return {
            'temporal': temporal,
            'frequency': frequency,
            'timefreq': timefreq,
            'label': item['label'],
            'fault_type': item['fault_type']
        }


class BearingTestDataset(Dataset):
    """
    测试数据集
    不进行窗口切分，直接使用完整文件
    """
    
    def __init__(self,
                 data_dir='raw_datasets/test',
                 timefreq_method='stft'):
        """
        Args:
            data_dir: 测试数据目录
            timefreq_method: 时频变换方法
        """
        self.data_dir = data_dir
        self.timefreq_method = timefreq_method
        
        # 创建变换
        self.temporal_transform = IdentityTransform()
        self.frequency_transform = FFTTransform(use_log=True, normalize=True)
        
        if timefreq_method == 'stft':
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
        
        # 加载文件列表
        self.file_list = sorted([f for f in os.listdir(data_dir) 
                                 if f.endswith('.xlsx')])
        
        print(f"Test dataset initialized: {len(self.file_list)} files")
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        """
        返回测试样本
        注意：测试集可能长度不是512，需要处理
        """
        filename = self.file_list[idx]
        file_path = os.path.join(self.data_dir, filename)
        
        # 读取信号
        df = pd.read_excel(file_path, header=None)
        signal = df.iloc[:, 1].values.astype(np.float32)
        signal = torch.from_numpy(signal).float()
        
        # 如果长度不是512，截取或填充
        if len(signal) < 512:
            # 填充
            signal = torch.nn.functional.pad(signal, (0, 512 - len(signal)))
        elif len(signal) > 512:
            # 截取中间部分
            start = (len(signal) - 512) // 2
            signal = signal[start:start+512]
        
        # 应用变换
        temporal = self.temporal_transform(signal).unsqueeze(0)
        frequency = self.frequency_transform(signal).unsqueeze(0)
        timefreq = self.timefreq_transform(signal).unsqueeze(0)
        
        return {
            'temporal': temporal,
            'frequency': frequency,
            'timefreq': timefreq,
            'filename': filename
        }


def create_dataloaders(data_dir='raw_datasets/train',
                      batch_size=32,
                      fold=0,
                      n_folds=5,
                      window_size=512,
                      window_step=256,
                      timefreq_method='stft',
                      augmentation_train=None,
                      num_workers=4):
    """
    创建训练和验证DataLoader的工厂函数
    
    Args:
        data_dir: 数据目录
        batch_size: batch大小
        fold: 当前fold编号
        n_folds: 总fold数
        window_size: 窗口大小
        window_step: 滑动步长
        timefreq_method: 时频方法
        augmentation_train: 训练集增强
        num_workers: 数据加载worker数
    
    Returns:
        train_loader, val_loader
    """
    from torch.utils.data import DataLoader
    
    # 创建训练集
    train_dataset = BearingDataset(
        data_dir=data_dir,
        window_size=window_size,
        window_step=window_step,
        mode='train',
        fold=fold,
        n_folds=n_folds,
        timefreq_method=timefreq_method,
        augmentation=augmentation_train,
        cache_data=True
    )
    
    # 创建验证集
    val_dataset = BearingDataset(
        data_dir=data_dir,
        window_size=window_size,
        window_step=window_step,
        mode='val',
        fold=fold,
        n_folds=n_folds,
        timefreq_method=timefreq_method,
        augmentation=None,  # 验证集不增强
        cache_data=True
    )
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
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
    print("测试轴承数据集")
    print("=" * 70)
    
    # 测试数据集创建
    print("\n1. 创建数据集 (fold=0)")
    train_dataset = BearingDataset(
        data_dir='raw_datasets/train',
        mode='train',
        fold=0,
        n_folds=5
    )
    
    val_dataset = BearingDataset(
        data_dir='raw_datasets/train',
        mode='val',
        fold=0,
        n_folds=5
    )
    
    print(f"   训练集大小: {len(train_dataset)}")
    print(f"   验证集大小: {len(val_dataset)}")
    
    # 测试获取样本
    print("\n2. 获取样本")
    sample = train_dataset[0]
    print(f"   Temporal: {sample['temporal'].shape}")
    print(f"   Frequency: {sample['frequency'].shape}")
    print(f"   Timefreq: {sample['timefreq'].shape}")
    print(f"   Label: {sample['label']} ({sample['fault_type']})")
    
    # 测试DataLoader
    print("\n3. 测试DataLoader")
    train_loader, val_loader = create_dataloaders(
        batch_size=4,
        fold=0,
        num_workers=0
    )
    
    batch = next(iter(train_loader))
    print(f"   Batch temporal: {batch['temporal'].shape}")
    print(f"   Batch frequency: {batch['frequency'].shape}")
    print(f"   Batch timefreq: {batch['timefreq'].shape}")
    print(f"   Batch labels: {batch['label'].shape}")
    
    print("\n✓ 数据集测试完成!")
