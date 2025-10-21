# 轴承故障诊断系统 - 多模态深度学习框架

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

基于多模态深度学习的轴承故障诊断系统,融合时域、频域和时频域特征,采用对比学习预训练+有监督微调的两阶段训练策略。

---

## ? 目录

- [项目概述](#-项目概述)
- [核心特性](#-核心特性)
- [系统架构](#-系统架构)
- [文件结构](#-文件结构)
- [环境配置](#-环境配置)
- [快速开始](#-快速开始)
- [配置说明](#-配置说明)
- [模型详解](#-模型详解)
- [训练策略](#-训练策略)
- [评估与可视化](#-评估与可视化)
- [常见问题](#-常见问题)

---

## ? 项目概述

### 研究背景
轴承作为旋转机械的核心部件,其健康状态直接影响设备运行安全。本项目基于振动信号,通过多模态深度学习实现轴承故障的智能诊断。

### 故障类型
- **正常** (Normal)
- **内圈磨损** (Inner Wear)
- **内圈断裂** (Inner Broken)
- **滚珠磨损** (Roller Wear)
- **滚珠断裂** (Roller Broken)
- **外圈缺失** (Outer Missing)

---

## ? 核心特性

### ? 模型创新
- **三分支多模态架构**: 时域、频域、时频域特征互补融合
- **多级自适应融合**: Level 1(模态重要性) → Level 2(跨模态交互) → Level 3(特征压缩)
- **双头联合训练**: Softmax + ArcFace度量学习
- **对比学习预训练**: 自监督学习通用特征表示

### ?? 工程优化
- **渐进式数据增强**: Weak → Medium → Strong动态调整
- **动态损失权重**: Focal + ArcFace权重随训练调整
- **Mixup增强**: 时域/特征空间混合正则化
- **梯度裁剪与混合精度**: 稳定训练,提升效率

### ? 完整工作流
```
原始信号 → 多域变换 → 三分支提取 → 多级融合 → 双头分类
    ↓          ↓           ↓          ↓         ↓
 滑窗切分   FFT/STFT   CNN/GRU/EMA   注意力    Softmax/ArcFace
```

---

## ?? 系统架构

### 网络结构图

```
输入: 振动信号 x(t)
    │
    ├─→ [时域分支]
    │   ├─ 宽卷积去噪
    │   ├─ MSFEB×2 (多尺度特征提取)
    │   ├─ CBAM注意力
    │   ├─ Bi-GRU时序建模
    │   ├─ 时序自注意力
    │   └─→ 特征 f_t (256-dim)
    │
    ├─→ [频域分支]
    │   ├─ FFT变换 → 频谱
    │   ├─ 1D ResNet
    │   ├─ SE注意力
    │   └─→ 特征 f_f (256-dim)
    │
    └─→ [时频分支]
        ├─ STFT/CWT → 时频图
        ├─ 2D CNN (3 blocks)
        ├─ EMA注意力
        └─→ 特征 f_tf (256-dim)

特征融合: [f_t, f_f, f_tf]
    │
    ├─→ Level 1: 自适应模态重要性
    │   ├─ 全局平均池化
    │   ├─ 重要性网络
    │   └─ 加权求和 → f_weighted
    │
    ├─→ Level 2: 跨模态注意力交互
    │   ├─ Multi-Head Attention
    │   └─ 残差连接 → f_interact
    │
    └─→ Level 3: 特征压缩
        └─ MLP → f_fused (128-dim)

分类器: 双头设计
    │
    ├─→ [Softmax Head]
    │   └─ FC → Logits (6-dim)
    │
    └─→ [ArcFace Head]
        ├─ 特征L2归一化
        ├─ 权重L2归一化
        ├─ 角度计算: cos(θ)
        ├─ 添加margin: cos(θ + m)
        └─ 缩放: s × cos(θ + m)

训练损失:
L_total = λ1?L_focal + λ2?L_arcface

预训练损失 (对比学习):
L_contrast = NT-Xent Loss
```

---

## ? 文件结构

```
bearing-fault-diagnosis/
│
├── configs/                          # 配置文件
│   ├── model_config.yaml            # 模型结构配置
│   ├── train_config.yaml            # 训练超参数
│   └── augmentation_config.yaml     # 数据增强策略
│
├── raw_datasets/                     # 原始数据
│   ├── train/                       # 训练集
│   │   ├── normal_train160/
│   │   ├── inner_wear_train120/
│   │   ├── inner_broken_train150/
│   │   ├── roller_wear_train100/
│   │   ├── roller_broken_train150/
│   │   └── outer_missing_train180/
│   └── test/                        # 测试集
│
├── preprocessors/                    # 数据预处理
│   ├── signal_transform.py          # FFT/STFT/CWT变换
│   ├── feature_extraction.py        # MED/VMD特征提取
│   └── rawdata_processor.py         # 原始数据处理
│
├── models/                           # 模型定义
│   ├── multimodal_model.py          # 完整多模态模型
│   │
│   ├── backbone/                    # 三分支backbone
│   │   ├── temporal_branch.py       # 时域分支
│   │   ├── frequency_branch.py      # 频域分支
│   │   └── timefreq_branch.py       # 时频分支
│   │
│   ├── modules/                     # 可复用模块
│   │   ├── attention.py             # CBAM/SE/EMA注意力
│   │   ├── msfeb.py                 # 多尺度特征提取块
│   │   ├── residual.py              # 残差块
│   │   └── pooling.py               # 池化层
│   │
│   ├── fusion/                      # 特征融合
│   │   ├── adaptive_fusion.py       # 自适应融合
│   │   ├── cross_modal_attention.py # 跨模态注意力
│   │   └── hierarchical_fusion.py   # 多级融合
│   │
│   └── heads/                       # 分类头
│       ├── softmax_head.py          # Softmax分类器
│       ├── arcface_head.py          # ArcFace分类器
│       └── dual_head.py             # 双头设计
│
├── datasets/                         # 数据集
│   ├── bearing_dataset.py           # 主数据集
│   ├── contrastive_dataset.py       # 对比学习数据集
│   └── dataloader.py                # DataLoader创建
│
├── augmentation/                     # 数据增强
│   ├── time_domain_aug.py           # 时域增强
│   ├── frequency_aug.py             # 频域增强
│   ├── mixup.py                     # Mixup增强
│   └── augmentation_pipeline.py     # 增强管道
│
├── losses/                           # 损失函数
│   ├── focal_loss.py                # Focal Loss
│   ├── combined_loss.py             # 组合损失
│   └── contrastive_loss.py          # 对比学习损失
│
├── training/                         # 训练模块
│   ├── trainer_base.py              # 训练器基类
│   │
│   ├── pretrain/                    # 预训练
│   │   ├── contrastive_trainer.py   # 对比学习训练器
│   │   └── pretrain_launcher.py     # 预训练启动器
│   │
│   ├── finetune/                    # 微调
│   │   ├── supervised_trainer.py    # 有监督训练器
│   │   └── finetune_launcher.py     # 微调启动器
│   │
│   ├── optimizer_factory.py         # 优化器工厂
│   ├── scheduler_factory.py         # 学习率调度器
│   └── callbacks.py                 # 训练回调
│
├── utils/                            # 工具函数
│   ├── config_parser.py             # 配置解析
│   ├── metrics.py                   # 评估指标
│   ├── visualization.py             # 可视化
│   ├── checkpoint.py                # 模型保存/加载
│   └── seed.py                      # 随机种子
│
├── evaluation/                       # 评估模块
│   ├── evaluator.py                 # 评估器
│   ├── confusion_matrix.py          # 混淆矩阵
│   ├── tsne_visualization.py        # t-SNE可视化
│   └── attention_visualization.py   # 注意力可视化
│
├── tests/                            # 单元测试
│   ├── test_models.py
│   ├── test_augmentation.py
│   ├── test_losses.py
│   └── test_dataset.py
│
├── runs/                             # 训练记录(自动生成)
│   ├── pretrain_YYYYMMDD_HHMMSS/
│   │   ├── config.yaml
│   │   ├── logs/
│   │   ├── checkpoints/
│   │   └── visualizations/
│   └── finetune_YYYYMMDD_HHMMSS/
│
├── main.py                           # 主入口
├── requirements.txt                  # Python依赖
├── README.md                         # 本文档
└── CODE_REVIEW_REPORT.md            # 代码审查报告
```

---

## ? 环境配置

### 系统要求
- **Python**: 3.8 或更高
- **PyTorch**: 2.0 或更高
- **CUDA**: 11.0+ (GPU训练推荐)
- **内存**: 16GB+ (训练时)
- **硬盘**: 10GB+ (数据+模型)

### 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/your-repo/bearing-fault-diagnosis.git
cd bearing-fault-diagnosis

# 2. 创建虚拟环境(推荐)
conda create -n bearing python=3.8
conda activate bearing

# 3. 安装PyTorch (根据CUDA版本选择)
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU版本
pip install torch torchvision torchaudio

# 4. 安装其他依赖
pip install -r requirements.txt

# 5. 验证安装
python -c "import torch; print(f'PyTorch {torch.__version__} - CUDA available: {torch.cuda.is_available()}')"
```

### requirements.txt 内容
```
torch>=2.0.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
scipy>=1.7.0
pyyaml>=6.0
matplotlib>=3.4.0
seaborn>=0.11.0
tqdm>=4.62.0
tensorboard>=2.10.0
openpyxl>=3.0.0
pywavelets>=1.3.0
```

---

## ? 快速开始

### 1. 数据准备

```bash
# 确保数据目录结构正确
raw_datasets/
├── train/
│   ├── normal_train160/          # 正常轴承,160个文件
│   ├── inner_wear_train120/      # 内圈磨损,120个文件
│   ├── inner_broken_train150/    # 内圈断裂,150个文件
│   ├── roller_wear_train100/     # 滚珠磨损,100个文件
│   ├── roller_broken_train150/   # 滚珠断裂,150个文件
│   └── outer_missing_train180/   # 外圈缺失,180个文件
└── test/                          # 测试集

# 每个文件格式: Excel (.xlsx)
# 第1列: 索引
# 第2列: 振动信号值
# 长度: 1024个采样点
```

### 2. 配置文件检查

```bash
# 查看配置文件是否存在
ls configs/
# 应该有: model_config.yaml, train_config.yaml, augmentation_config.yaml

# 根据需要修改配置
vim configs/train_config.yaml
```

### 3. 完整训练流程

#### 方式1: 一键训练(推荐)
```bash
# 完整流程: 预训练 + 微调
python main.py --mode full --experiment_name my_exp

# 等价于:
# 1. 对比学习预训练 (30 epochs)
# 2. 有监督微调 (100 epochs)
```

#### 方式2: 分步训练
```bash
# Step 1: 仅预训练
python main.py --mode pretrain --experiment_name pretrain_exp

# Step 2: 使用预训练权重微调
python main.py --mode finetune \
    --pretrained_weights runs/pretrain_exp/pretrained_weights.pth \
    --experiment_name finetune_exp
```

#### 方式3: 跳过预训练(快速测试)
```bash
python main.py --mode full --skip_pretrain --experiment_name quick_test
```

### 4. 命令行参数

```bash
python main.py \
    --config_dir configs \              # 配置文件目录
    --mode full \                       # 训练模式: pretrain/finetune/full
    --experiment_name my_exp \          # 实验名称
    --device cuda \                     # 设备: cuda/cpu
    --seed 42 \                         # 随机种子
    --skip_pretrain                     # 跳过预训练(可选)
```

### 5. 监控训练

```bash
# 查看训练日志
tail -f runs/my_exp_pretrain/logs/training.log

# TensorBoard可视化(如果启用)
tensorboard --logdir runs/

# 检查checkpoint
ls runs/my_exp_finetune/checkpoints/
```

---

## ?? 配置说明

### model_config.yaml - 模型结构

```yaml
model:
  config: 'medium'              # 模型规模: small/medium/large
  seq_len: 512                  # 输入序列长度
  num_classes: 6                # 故障类别数
  fusion_type: 'hierarchical'   # 融合方式
  head_type: 'dual'             # 分类头类型

backbone:
  temporal:
    gru_hidden: 128
    gru_layers: 2
    attention_heads: 4
  
  frequency:
    freq_len: 257               # FFT频点数
  
  timefreq:
    input_size: [64, 128]       # STFT时频图尺寸
    timefreq_method: 'stft'     # 'stft' 或 'cwt'

fusion:
  modal_dim: 256                # 各模态输出维度
  fusion_output_dim: 128        # 融合后维度
  num_heads: 4                  # 注意力头数
  dropout: [0.25, 0.30, 0.35]   # 三个Level的dropout

classifier:
  head_hidden_dim: 256
  arcface_s: 30.0               # ArcFace scale
  arcface_m: 0.50               # ArcFace margin
  dropout: [0.40, 0.30]         # 分类器dropout
```

### train_config.yaml - 训练超参数

```yaml
# 通用设置
device: 'cuda'                  # 训练设备
seed: 42                        # 随机种子
use_amp: true                   # 混合精度训练
num_workers: 4                  # DataLoader workers

# 数据配置
data:
  train_dir: 'raw_datasets/train'
  test_dir: 'raw_datasets/test'
  window_size: 512              # 滑动窗口大小
  window_step: 256              # 滑动步长
  n_folds: 5                    # K-fold交叉验证
  current_fold: 0               # 当前fold

# 预训练配置
pretrain:
  epochs: 30
  batch_size: 64
  optimizer:
    type: 'adamw'
    lr: 0.001
    weight_decay: 0.0001
  scheduler:
    type: 'cosine_warmup'
    warmup_epochs: 3
    T_0: 10
  loss:
    type: 'ntxent'              # 'ntxent' 或 'supcon'
    temperature: 0.07
  early_stopping:
    patience: 8
    monitor: 'contrastive_loss'

# 微调配置
finetune:
  epochs: 100
  batch_size: 32
  optimizer:
    type: 'adamw'
    lr: 0.0005
    weight_decay: 0.0005
  scheduler:
    type: 'combined'
    warmup_epochs: 5
  loss:
    focal:
      gamma: 2.0
      label_smoothing: 0.1
    arcface:
      weight: 0.5
    weights_schedule:           # 动态权重
      - [0, 30, 0.3]           # epoch 0-30: ArcFace权重0.3
      - [30, 60, 0.5]          # epoch 30-60: 权重0.5
      - [60, 100, 0.7]         # epoch 60-100: 权重0.7
  early_stopping:
    patience: 15
    monitor: 'val_acc'
  mixup:
    enable: true
    alpha: 0.2
  gradient_clip:
    enable: true
    max_norm: 1.0
```

### augmentation_config.yaml - 数据增强

```yaml
# 预训练增强(强)
contrastive:
  strong_aug_prob: 0.5          # 强增强概率
  time_domain:
    gaussian_noise:
      sigma_range: [0.03, 0.05]
      prob: 0.8
    time_shift:
      shift_range: [-50, 50]
      prob: 0.6
    # ... 其他增强

# 有监督训练增强(渐进式)
supervised:
  progressive:                  # 渐进式增强
    enable: true
    stages:
      - [0, 30, 'weak']        # epoch 0-30: 弱增强
      - [30, 70, 'medium']     # epoch 30-70: 中等增强
      - [70, 100, 'strong']    # epoch 70-100: 强增强
  
  weak:
    time_domain:
      gaussian_noise:
        sigma_range: [0.01, 0.02]
        prob: 0.3
      # ...
  
  medium:
    time_domain:
      gaussian_noise:
        sigma_range: [0.02, 0.04]
        prob: 0.5
      # ...
  
  strong:
    time_domain:
      gaussian_noise:
        sigma_range: [0.03, 0.05]
        prob: 0.6
      # ...

# Mixup配置
mixup:
  time_domain:
    enable: true
    alpha: 0.2
    prob: 0.5
```

---

## ? 模型详解

### 模型规模对比

| 配置 | 参数量 | 时域GRU | 融合维度 | 适用场景 |
|------|--------|---------|----------|----------|
| Small | ~800K | 64×1层 | 64-dim | 快速原型/资源受限 |
| Medium | ~2.5M | 128×2层 | 128-dim | **推荐,平衡性能与效率** |
| Large | ~8M | 256×2层 | 256-dim | 追求最高精度 |

### 关键组件说明

#### 1. 时域分支 (Temporal Branch)
**输入**: 原始振动信号 (B, 1, 512)

**架构**:
```
Stage 1: 宽卷积去噪
├─ Conv1d(kernel=64) → BN → LeakyReLU → Dropout(0.1)

Stage 2: 多尺度特征提取 (MSFEB×2)
├─ MSFEB: 3个并行卷积(kernel=3,5,7)
├─ Concat → 1×1 Conv降维
└─ Dropout(0.15)

Stage 3: CBAM注意力
├─ Channel Attention (SE-like)
└─ Spatial Attention (1D卷积)

Stage 4: 时序建模
├─ Bi-GRU(hidden=128, layers=2, dropout=0.2)
├─ Temporal Self-Attention(heads=4)
└─ 捕捉长程依赖

Stage 5: 特征聚合
├─ Multi-Head Pooling(Max + Avg + Attention)
└─ MLP压缩 → 输出(B, 256)
```

**作用**: 捕捉冲击、周期性模式、局部异常

#### 2. 频域分支 (Frequency Branch)
**输入**: FFT频谱 (B, 1, 257)

**架构**:
```
Stage 1: 1D ResNet
├─ Conv1d(64) → BN → ReLU
├─ ResBlock×2 (通道:64 → 128)
└─ SE注意力

Stage 2: 全局池化
├─ Adaptive Avg Pooling
└─ MLP压缩 → 输出(B, 256)
```

**作用**: 捕捉谐波成分、频率特征、能量分布

#### 3. 时频分支 (Time-Frequency Branch)
**输入**: STFT时频图 (B, 1, 64, 128)

**架构**:
```
Stage 1: 2D CNN
├─ Conv Block 1: 32通道
├─ Conv Block 2: 64通道
├─ Conv Block 3: 128通道
└─ MaxPooling下采样

Stage 2: EMA注意力
├─ 高效多尺度特征聚合
└─ 参数量少,效果好

Stage 3: 自适应池化
├─ Adaptive Avg Pool(4×4)
├─ Flatten
└─ MLP压缩 → 输出(B, 256)
```

**作用**: 捕捉时变频率、瞬态冲击、能量演变

#### 4. 多级自适应融合
**输入**: [f_t, f_f, f_tf] 三个256-dim特征

**Level 1: 模态重要性学习**
```python
# 计算各模态的全局统计
g_t = GlobalAvgPool(f_t)  # (B, 256)
g_f = GlobalAvgPool(f_f)
g_tf = GlobalAvgPool(f_tf)

# 拼接
g_concat = concat([g_t, g_f, g_tf])  # (B, 768)

# 重要性网络
scores = MLP(g_concat)  # (B, 3)
weights = Softmax(scores)  # (B, 3) 权重和为1

# 加权融合
f_weighted = w_t?f_t + w_f?f_f + w_tf?f_tf  # (B, 256)
```

**Level 2: 跨模态注意力交互**
```python
# 拼接三个模态
f_all = concat([f_t, f_f, f_tf])  # (B, 768)

# Multi-Head Cross-Attention
f_interact = MultiHeadAttention(
    query=f_weighted,    # Level 1输出
    key=f_all,
    value=f_all
)  # (B, 768)

# 残差连接
f_interact = f_interact + f_weighted
```

**Level 3: 特征压缩**
```python
# MLP降维
f_fused = MLP(
    input_dim=768,
    hidden_dim=384,
    output_dim=128,
    dropout=[0.25, 0.35]
)  # (B, 128)

# 最终融合特征
```

#### 5. 双头分类器

**Softmax Head** (传统分类)
```python
x = Dropout(0.4)(f_fused)
x = Linear(128, 256)(x)
x = BatchNorm1d(256)(x)
x = ReLU()(x)
x = Dropout(0.3)(x)
logits = Linear(256, num_classes)(x)  # (B, 6)
```

**ArcFace Head** (度量学习)
```python
# 特征归一化
f_norm = F.normalize(f_fused, p=2, dim=1)

# 权重归一化
W_norm = F.normalize(W, p=2, dim=0)  # (128, 6)

# 计算余弦相似度
cos_theta = f_norm @ W_norm  # (B, 6)

# 目标类添加margin
cos_theta_m = cos(acos(cos_theta[y]) + m)

# 缩放
logits = s * cos_theta_m  # (B, 6)
```

**联合训练**
```python
if training:
    return softmax_logits, arcface_logits, features
else:
    # 推理时只用softmax
    return softmax_logits
```

---

## ? 训练策略

### 两阶段训练

#### 阶段1: 对比学习预训练 (30 epochs)
**目标**: 学习通用的判别性特征表示

```python
# 数据: 每个样本生成2个强增强版本
x_view1, x_view2 = strong_augment(x)

# 前向: 提取特征投影
z1 = ProjectionHead(Backbone(x_view1))  # (B, 128)
z2 = ProjectionHead(Backbone(x_view2))  # (B, 128)

# 损失: NT-Xent
loss = -log(exp(sim(z1,z2)/τ) / Σexp(sim(z1,zk)/τ))

# 优化: AdamW, lr=1e-3
# 调度器: CosineAnnealingWarmRestarts
```

**作用**:
- 学习不同增强下的不变特征
- 避免过早陷入分类任务
- 提供更好的初始化

#### 阶段2: 有监督微调 (100 epochs)
**目标**: 在预训练基础上学习分类

```python
# 加载预训练权重
model.load_pretrained_backbone('pretrained_weights.pth')

# 前向: 双头输出
softmax_logits, arcface_logits, features = model(batch)

# 损失: Focal + ArcFace
loss = focal_loss(softmax_logits, y) + 0.5 * arcface_loss(arcface_logits, y)

# 优化: AdamW, lr=5e-4 (更小)
# 调度器: Warmup + CosineAnnealing
```

**技巧**:
- 前5 epoch: Warmup,学习率线性增长
- 5-80 epoch: CosineAnnealing
- 80+ epoch: ReduceLROnPlateau
- 动态调整ArcFace权重: 0.3 → 0.5 → 0.7

### 数据增强策略

#### 预训练阶段 (强增强)
```python
contrastive_augmentation = [
    GaussianNoise(σ=0.05, p=0.8),
    TimeShift(offset=[-50,50], p=0.6),
    AmplitudeScale(scale=[0.7,1.3], p=0.7),
    TimeWarping(ratio=[0.85,1.15], p=0.5),
    RandomMasking(ratio=0.15, p=0.5),
    AddImpulse(num=[1,3], p=0.4),
]
```

#### 微调阶段 (渐进式)
```python
# Epoch 0-30: 弱增强
weak_aug = [
    GaussianNoise(σ=0.01, p=0.3),
    AmplitudeScale(scale=[0.9,1.1], p=0.3),
]

# Epoch 30-70: 中等增强
medium_aug = [
    GaussianNoise(σ=0.03, p=0.5),
    TimeShift(offset=[-30,30], p=0.4),
    AmplitudeScale(scale=[0.8,1.2], p=0.5),
]

# Epoch 70-100: 强增强
strong_aug = [
    GaussianNoise(σ=0.05, p=0.6),
    TimeShift(offset=[-50,50], p=0.5),
    AmplitudeScale(scale=[0.7,1.3], p=0.6),
    TimeWarping(ratio=[0.9,1.1], p=0.4),
    RandomMasking(ratio=0.1, p=0.4),
]
```

#### Mixup增强
```python
# 时域mixup (有监督阶段)
if use_mixup and training:
    x_mix, y_a, y_b, lam = mixup_data(x, y, alpha=0.2)
    logits = model(x_mix)
    loss = lam * loss(logits, y_a) + (1-lam) * loss(logits, y_b)
```

### 损失函数设计

#### Focal Loss (解决类别不平衡)
```python
FL(p_t) = -α_t × (1-p_t)^γ × log(p_t)

# 参数:
γ = 2.0                 # 聚焦参数
α = [权重向量]          # 类别权重,根据样本数自动计算
label_smoothing = 0.1   # 标签平滑
```

**类别权重计算**:
```python
# Effective Number方法
def compute_class_weights(class_counts, beta=0.9999):
    effective_num = 1.0 - beta^class_counts
    weights = (1 - beta) / effective_num
    weights = normalize(weights)
    return weights

# 示例:
# [160, 120, 180, 150, 100, 150] → [0.86, 1.05, 0.78, 0.93, 1.32, 0.93]
```

#### ArcFace Loss (度量学习)
```python
L_arc = -log(exp(s?cos(θ_yi + m)) / Σexp(s?cos(θ_j)))

# 参数:
s = 30.0   # scale,控制logits范围
m = 0.50   # margin,增大类间距
```

**效果**: 最小化类内距,最大化类间距

#### 组合损失 (渐进式权重)
```python
L_total = λ1?L_focal + λ2?L_arcface

# 动态权重:
epoch 0-30:   λ2 = 0.3
epoch 30-60:  λ2 = 0.5
epoch 60-100: λ2 = 0.7

# 后期加大度量学习权重
```

### 正则化技巧

```python
# 1. Label Smoothing
y_smooth = (1 - ε) * y_onehot + ε / num_classes
# ε = 0.1

# 2. Dropout (分层设计)
# 浅层: 0.1-0.15 (保留底层特征)
# 中层: 0.2-0.3 (防止过拟合)
# 深层: 0.3-0.4 (强正则化)

# 3. Gradient Clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 4. Weight Decay
weight_decay = 5e-4  # 比预训练大,防过拟合

# 5. Mixup
alpha = 0.2, prob = 0.5

# 6. Early Stopping
patience = 15, monitor = 'val_acc'
```

---

## ? 评估与可视化

### 评估指标

```python
from utils.metrics import compute_metrics

metrics = compute_metrics(y_true, y_pred)
# 返回:
# - accuracy: 总体准确率
# - precision: 各类精确率
# - recall: 各类召回率
# - f1_score: 各类F1分数
# - confusion_matrix: 混淆矩阵
```

### 混淆矩阵

```python
from evaluation.confusion_matrix import plot_confusion_matrix

plot_confusion_matrix(
    y_true, y_pred,
    class_names=['Normal', 'Inner Wear', 'Inner Broken', 
                 'Roller Wear', 'Roller Broken', 'Outer Missing'],
    save_path='runs/exp/confusion_matrix.png'
)
```

### t-SNE特征可视化

```python
from evaluation.tsne_visualization import visualize_features

visualize_features(
    model, test_loader,
    save_path='runs/exp/tsne.png'
)
```

### 注意力权重可视化

```python
from evaluation.attention_visualization import visualize_modal_weights

# 可视化模态重要性
visualize_modal_weights(
    model, test_loader,
    save_path='runs/exp/modal_weights.png'
)
```

### TensorBoard监控

```bash
# 启动TensorBoard
tensorboard --logdir runs/

# 查看:
# - 训练/验证损失曲线
# - 准确率曲线
# - 学习率变化
# - 梯度分布
# - 参数分布
```

---

