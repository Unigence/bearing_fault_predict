# ��й������ϵͳ - ��ģ̬���ѧϰ���

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

���ڶ�ģ̬���ѧϰ����й������ϵͳ,�ں�ʱ��Ƶ���ʱƵ������,���öԱ�ѧϰԤѵ��+�мල΢�������׶�ѵ�����ԡ�

---

## ? Ŀ¼

- [��Ŀ����](#-��Ŀ����)
- [��������](#-��������)
- [ϵͳ�ܹ�](#-ϵͳ�ܹ�)
- [�ļ��ṹ](#-�ļ��ṹ)
- [��������](#-��������)
- [���ٿ�ʼ](#-���ٿ�ʼ)
- [����˵��](#-����˵��)
- [ģ�����](#-ģ�����)
- [ѵ������](#-ѵ������)
- [��������ӻ�](#-��������ӻ�)
- [��������](#-��������)

---

## ? ��Ŀ����

### �о�����
�����Ϊ��ת��е�ĺ��Ĳ���,�佡��״ֱ̬��Ӱ���豸���а�ȫ������Ŀ�������ź�,ͨ����ģ̬���ѧϰʵ����й��ϵ�������ϡ�

### ��������
- **����** (Normal)
- **��Ȧĥ��** (Inner Wear)
- **��Ȧ����** (Inner Broken)
- **����ĥ��** (Roller Wear)
- **�������** (Roller Broken)
- **��Ȧȱʧ** (Outer Missing)

---

## ? ��������

### ? ģ�ʹ���
- **����֧��ģ̬�ܹ�**: ʱ��Ƶ��ʱƵ�����������ں�
- **�༶����Ӧ�ں�**: Level 1(ģ̬��Ҫ��) �� Level 2(��ģ̬����) �� Level 3(����ѹ��)
- **˫ͷ����ѵ��**: Softmax + ArcFace����ѧϰ
- **�Ա�ѧϰԤѵ��**: �Լලѧϰͨ��������ʾ

### ?? �����Ż�
- **����ʽ������ǿ**: Weak �� Medium �� Strong��̬����
- **��̬��ʧȨ��**: Focal + ArcFaceȨ����ѵ������
- **Mixup��ǿ**: ʱ��/�����ռ�������
- **�ݶȲü����Ͼ���**: �ȶ�ѵ��,����Ч��

### ? ����������
```
ԭʼ�ź� �� ����任 �� ����֧��ȡ �� �༶�ں� �� ˫ͷ����
    ��          ��           ��          ��         ��
 �����з�   FFT/STFT   CNN/GRU/EMA   ע����    Softmax/ArcFace
```

---

## ?? ϵͳ�ܹ�

### ����ṹͼ

```
����: ���ź� x(t)
    ��
    ������ [ʱ���֧]
    ��   ���� ����ȥ��
    ��   ���� MSFEB��2 (��߶�������ȡ)
    ��   ���� CBAMע����
    ��   ���� Bi-GRUʱ��ģ
    ��   ���� ʱ����ע����
    ��   ������ ���� f_t (256-dim)
    ��
    ������ [Ƶ���֧]
    ��   ���� FFT�任 �� Ƶ��
    ��   ���� 1D ResNet
    ��   ���� SEע����
    ��   ������ ���� f_f (256-dim)
    ��
    ������ [ʱƵ��֧]
        ���� STFT/CWT �� ʱƵͼ
        ���� 2D CNN (3 blocks)
        ���� EMAע����
        ������ ���� f_tf (256-dim)

�����ں�: [f_t, f_f, f_tf]
    ��
    ������ Level 1: ����Ӧģ̬��Ҫ��
    ��   ���� ȫ��ƽ���ػ�
    ��   ���� ��Ҫ������
    ��   ���� ��Ȩ��� �� f_weighted
    ��
    ������ Level 2: ��ģ̬ע��������
    ��   ���� Multi-Head Attention
    ��   ���� �в����� �� f_interact
    ��
    ������ Level 3: ����ѹ��
        ���� MLP �� f_fused (128-dim)

������: ˫ͷ���
    ��
    ������ [Softmax Head]
    ��   ���� FC �� Logits (6-dim)
    ��
    ������ [ArcFace Head]
        ���� ����L2��һ��
        ���� Ȩ��L2��һ��
        ���� �Ƕȼ���: cos(��)
        ���� ���margin: cos(�� + m)
        ���� ����: s �� cos(�� + m)

ѵ����ʧ:
L_total = ��1?L_focal + ��2?L_arcface

Ԥѵ����ʧ (�Ա�ѧϰ):
L_contrast = NT-Xent Loss
```

---

## ? �ļ��ṹ

```
bearing-fault-diagnosis/
��
������ configs/                          # �����ļ�
��   ������ model_config.yaml            # ģ�ͽṹ����
��   ������ train_config.yaml            # ѵ��������
��   ������ augmentation_config.yaml     # ������ǿ����
��
������ raw_datasets/                     # ԭʼ����
��   ������ train/                       # ѵ����
��   ��   ������ normal_train160/
��   ��   ������ inner_wear_train120/
��   ��   ������ inner_broken_train150/
��   ��   ������ roller_wear_train100/
��   ��   ������ roller_broken_train150/
��   ��   ������ outer_missing_train180/
��   ������ test/                        # ���Լ�
��
������ preprocessors/                    # ����Ԥ����
��   ������ signal_transform.py          # FFT/STFT/CWT�任
��   ������ feature_extraction.py        # MED/VMD������ȡ
��   ������ rawdata_processor.py         # ԭʼ���ݴ���
��
������ models/                           # ģ�Ͷ���
��   ������ multimodal_model.py          # ������ģ̬ģ��
��   ��
��   ������ backbone/                    # ����֧backbone
��   ��   ������ temporal_branch.py       # ʱ���֧
��   ��   ������ frequency_branch.py      # Ƶ���֧
��   ��   ������ timefreq_branch.py       # ʱƵ��֧
��   ��
��   ������ modules/                     # �ɸ���ģ��
��   ��   ������ attention.py             # CBAM/SE/EMAע����
��   ��   ������ msfeb.py                 # ��߶�������ȡ��
��   ��   ������ residual.py              # �в��
��   ��   ������ pooling.py               # �ػ���
��   ��
��   ������ fusion/                      # �����ں�
��   ��   ������ adaptive_fusion.py       # ����Ӧ�ں�
��   ��   ������ cross_modal_attention.py # ��ģ̬ע����
��   ��   ������ hierarchical_fusion.py   # �༶�ں�
��   ��
��   ������ heads/                       # ����ͷ
��       ������ softmax_head.py          # Softmax������
��       ������ arcface_head.py          # ArcFace������
��       ������ dual_head.py             # ˫ͷ���
��
������ datasets/                         # ���ݼ�
��   ������ bearing_dataset.py           # �����ݼ�
��   ������ contrastive_dataset.py       # �Ա�ѧϰ���ݼ�
��   ������ dataloader.py                # DataLoader����
��
������ augmentation/                     # ������ǿ
��   ������ time_domain_aug.py           # ʱ����ǿ
��   ������ frequency_aug.py             # Ƶ����ǿ
��   ������ mixup.py                     # Mixup��ǿ
��   ������ augmentation_pipeline.py     # ��ǿ�ܵ�
��
������ losses/                           # ��ʧ����
��   ������ focal_loss.py                # Focal Loss
��   ������ combined_loss.py             # �����ʧ
��   ������ contrastive_loss.py          # �Ա�ѧϰ��ʧ
��
������ training/                         # ѵ��ģ��
��   ������ trainer_base.py              # ѵ��������
��   ��
��   ������ pretrain/                    # Ԥѵ��
��   ��   ������ contrastive_trainer.py   # �Ա�ѧϰѵ����
��   ��   ������ pretrain_launcher.py     # Ԥѵ��������
��   ��
��   ������ finetune/                    # ΢��
��   ��   ������ supervised_trainer.py    # �мලѵ����
��   ��   ������ finetune_launcher.py     # ΢��������
��   ��
��   ������ optimizer_factory.py         # �Ż�������
��   ������ scheduler_factory.py         # ѧϰ�ʵ�����
��   ������ callbacks.py                 # ѵ���ص�
��
������ utils/                            # ���ߺ���
��   ������ config_parser.py             # ���ý���
��   ������ metrics.py                   # ����ָ��
��   ������ visualization.py             # ���ӻ�
��   ������ checkpoint.py                # ģ�ͱ���/����
��   ������ seed.py                      # �������
��
������ evaluation/                       # ����ģ��
��   ������ evaluator.py                 # ������
��   ������ confusion_matrix.py          # ��������
��   ������ tsne_visualization.py        # t-SNE���ӻ�
��   ������ attention_visualization.py   # ע�������ӻ�
��
������ tests/                            # ��Ԫ����
��   ������ test_models.py
��   ������ test_augmentation.py
��   ������ test_losses.py
��   ������ test_dataset.py
��
������ runs/                             # ѵ����¼(�Զ�����)
��   ������ pretrain_YYYYMMDD_HHMMSS/
��   ��   ������ config.yaml
��   ��   ������ logs/
��   ��   ������ checkpoints/
��   ��   ������ visualizations/
��   ������ finetune_YYYYMMDD_HHMMSS/
��
������ main.py                           # �����
������ requirements.txt                  # Python����
������ README.md                         # ���ĵ�
������ CODE_REVIEW_REPORT.md            # ������鱨��
```

---

## ? ��������

### ϵͳҪ��
- **Python**: 3.8 �����
- **PyTorch**: 2.0 �����
- **CUDA**: 11.0+ (GPUѵ���Ƽ�)
- **�ڴ�**: 16GB+ (ѵ��ʱ)
- **Ӳ��**: 10GB+ (����+ģ��)

### ��װ����

```bash
# 1. ��¡�ֿ�
git clone https://github.com/your-repo/bearing-fault-diagnosis.git
cd bearing-fault-diagnosis

# 2. �������⻷��(�Ƽ�)
conda create -n bearing python=3.8
conda activate bearing

# 3. ��װPyTorch (����CUDA�汾ѡ��)
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU�汾
pip install torch torchvision torchaudio

# 4. ��װ��������
pip install -r requirements.txt

# 5. ��֤��װ
python -c "import torch; print(f'PyTorch {torch.__version__} - CUDA available: {torch.cuda.is_available()}')"
```

### requirements.txt ����
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

## ? ���ٿ�ʼ

### 1. ����׼��

```bash
# ȷ������Ŀ¼�ṹ��ȷ
raw_datasets/
������ train/
��   ������ normal_train160/          # �������,160���ļ�
��   ������ inner_wear_train120/      # ��Ȧĥ��,120���ļ�
��   ������ inner_broken_train150/    # ��Ȧ����,150���ļ�
��   ������ roller_wear_train100/     # ����ĥ��,100���ļ�
��   ������ roller_broken_train150/   # �������,150���ļ�
��   ������ outer_missing_train180/   # ��Ȧȱʧ,180���ļ�
������ test/                          # ���Լ�

# ÿ���ļ���ʽ: Excel (.xlsx)
# ��1��: ����
# ��2��: ���ź�ֵ
# ����: 1024��������
```

### 2. �����ļ����

```bash
# �鿴�����ļ��Ƿ����
ls configs/
# Ӧ����: model_config.yaml, train_config.yaml, augmentation_config.yaml

# ������Ҫ�޸�����
vim configs/train_config.yaml
```

### 3. ����ѵ������

#### ��ʽ1: һ��ѵ��(�Ƽ�)
```bash
# ��������: Ԥѵ�� + ΢��
python main.py --mode full --experiment_name my_exp

# �ȼ���:
# 1. �Ա�ѧϰԤѵ�� (30 epochs)
# 2. �мල΢�� (100 epochs)
```

#### ��ʽ2: �ֲ�ѵ��
```bash
# Step 1: ��Ԥѵ��
python main.py --mode pretrain --experiment_name pretrain_exp

# Step 2: ʹ��Ԥѵ��Ȩ��΢��
python main.py --mode finetune \
    --pretrained_weights runs/pretrain_exp/pretrained_weights.pth \
    --experiment_name finetune_exp
```

#### ��ʽ3: ����Ԥѵ��(���ٲ���)
```bash
python main.py --mode full --skip_pretrain --experiment_name quick_test
```

### 4. �����в���

```bash
python main.py \
    --config_dir configs \              # �����ļ�Ŀ¼
    --mode full \                       # ѵ��ģʽ: pretrain/finetune/full
    --experiment_name my_exp \          # ʵ������
    --device cuda \                     # �豸: cuda/cpu
    --seed 42 \                         # �������
    --skip_pretrain                     # ����Ԥѵ��(��ѡ)
```

### 5. ���ѵ��

```bash
# �鿴ѵ����־
tail -f runs/my_exp_pretrain/logs/training.log

# TensorBoard���ӻ�(�������)
tensorboard --logdir runs/

# ���checkpoint
ls runs/my_exp_finetune/checkpoints/
```

---

## ?? ����˵��

### model_config.yaml - ģ�ͽṹ

```yaml
model:
  config: 'medium'              # ģ�͹�ģ: small/medium/large
  seq_len: 512                  # �������г���
  num_classes: 6                # ���������
  fusion_type: 'hierarchical'   # �ںϷ�ʽ
  head_type: 'dual'             # ����ͷ����

backbone:
  temporal:
    gru_hidden: 128
    gru_layers: 2
    attention_heads: 4
  
  frequency:
    freq_len: 257               # FFTƵ����
  
  timefreq:
    input_size: [64, 128]       # STFTʱƵͼ�ߴ�
    timefreq_method: 'stft'     # 'stft' �� 'cwt'

fusion:
  modal_dim: 256                # ��ģ̬���ά��
  fusion_output_dim: 128        # �ںϺ�ά��
  num_heads: 4                  # ע����ͷ��
  dropout: [0.25, 0.30, 0.35]   # ����Level��dropout

classifier:
  head_hidden_dim: 256
  arcface_s: 30.0               # ArcFace scale
  arcface_m: 0.50               # ArcFace margin
  dropout: [0.40, 0.30]         # ������dropout
```

### train_config.yaml - ѵ��������

```yaml
# ͨ������
device: 'cuda'                  # ѵ���豸
seed: 42                        # �������
use_amp: true                   # ��Ͼ���ѵ��
num_workers: 4                  # DataLoader workers

# ��������
data:
  train_dir: 'raw_datasets/train'
  test_dir: 'raw_datasets/test'
  window_size: 512              # �������ڴ�С
  window_step: 256              # ��������
  n_folds: 5                    # K-fold������֤
  current_fold: 0               # ��ǰfold

# Ԥѵ������
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
    type: 'ntxent'              # 'ntxent' �� 'supcon'
    temperature: 0.07
  early_stopping:
    patience: 8
    monitor: 'contrastive_loss'

# ΢������
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
    weights_schedule:           # ��̬Ȩ��
      - [0, 30, 0.3]           # epoch 0-30: ArcFaceȨ��0.3
      - [30, 60, 0.5]          # epoch 30-60: Ȩ��0.5
      - [60, 100, 0.7]         # epoch 60-100: Ȩ��0.7
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

### augmentation_config.yaml - ������ǿ

```yaml
# Ԥѵ����ǿ(ǿ)
contrastive:
  strong_aug_prob: 0.5          # ǿ��ǿ����
  time_domain:
    gaussian_noise:
      sigma_range: [0.03, 0.05]
      prob: 0.8
    time_shift:
      shift_range: [-50, 50]
      prob: 0.6
    # ... ������ǿ

# �мලѵ����ǿ(����ʽ)
supervised:
  progressive:                  # ����ʽ��ǿ
    enable: true
    stages:
      - [0, 30, 'weak']        # epoch 0-30: ����ǿ
      - [30, 70, 'medium']     # epoch 30-70: �е���ǿ
      - [70, 100, 'strong']    # epoch 70-100: ǿ��ǿ
  
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

# Mixup����
mixup:
  time_domain:
    enable: true
    alpha: 0.2
    prob: 0.5
```

---

## ? ģ�����

### ģ�͹�ģ�Ա�

| ���� | ������ | ʱ��GRU | �ں�ά�� | ���ó��� |
|------|--------|---------|----------|----------|
| Small | ~800K | 64��1�� | 64-dim | ����ԭ��/��Դ���� |
| Medium | ~2.5M | 128��2�� | 128-dim | **�Ƽ�,ƽ��������Ч��** |
| Large | ~8M | 256��2�� | 256-dim | ׷����߾��� |

### �ؼ����˵��

#### 1. ʱ���֧ (Temporal Branch)
**����**: ԭʼ���ź� (B, 1, 512)

**�ܹ�**:
```
Stage 1: ����ȥ��
���� Conv1d(kernel=64) �� BN �� LeakyReLU �� Dropout(0.1)

Stage 2: ��߶�������ȡ (MSFEB��2)
���� MSFEB: 3�����о��(kernel=3,5,7)
���� Concat �� 1��1 Conv��ά
���� Dropout(0.15)

Stage 3: CBAMע����
���� Channel Attention (SE-like)
���� Spatial Attention (1D���)

Stage 4: ʱ��ģ
���� Bi-GRU(hidden=128, layers=2, dropout=0.2)
���� Temporal Self-Attention(heads=4)
���� ��׽��������

Stage 5: �����ۺ�
���� Multi-Head Pooling(Max + Avg + Attention)
���� MLPѹ�� �� ���(B, 256)
```

**����**: ��׽�����������ģʽ���ֲ��쳣

#### 2. Ƶ���֧ (Frequency Branch)
**����**: FFTƵ�� (B, 1, 257)

**�ܹ�**:
```
Stage 1: 1D ResNet
���� Conv1d(64) �� BN �� ReLU
���� ResBlock��2 (ͨ��:64 �� 128)
���� SEע����

Stage 2: ȫ�ֳػ�
���� Adaptive Avg Pooling
���� MLPѹ�� �� ���(B, 256)
```

**����**: ��׽г���ɷ֡�Ƶ�������������ֲ�

#### 3. ʱƵ��֧ (Time-Frequency Branch)
**����**: STFTʱƵͼ (B, 1, 64, 128)

**�ܹ�**:
```
Stage 1: 2D CNN
���� Conv Block 1: 32ͨ��
���� Conv Block 2: 64ͨ��
���� Conv Block 3: 128ͨ��
���� MaxPooling�²���

Stage 2: EMAע����
���� ��Ч��߶������ۺ�
���� ��������,Ч����

Stage 3: ����Ӧ�ػ�
���� Adaptive Avg Pool(4��4)
���� Flatten
���� MLPѹ�� �� ���(B, 256)
```

**����**: ��׽ʱ��Ƶ�ʡ�˲̬����������ݱ�

#### 4. �༶����Ӧ�ں�
**����**: [f_t, f_f, f_tf] ����256-dim����

**Level 1: ģ̬��Ҫ��ѧϰ**
```python
# �����ģ̬��ȫ��ͳ��
g_t = GlobalAvgPool(f_t)  # (B, 256)
g_f = GlobalAvgPool(f_f)
g_tf = GlobalAvgPool(f_tf)

# ƴ��
g_concat = concat([g_t, g_f, g_tf])  # (B, 768)

# ��Ҫ������
scores = MLP(g_concat)  # (B, 3)
weights = Softmax(scores)  # (B, 3) Ȩ�غ�Ϊ1

# ��Ȩ�ں�
f_weighted = w_t?f_t + w_f?f_f + w_tf?f_tf  # (B, 256)
```

**Level 2: ��ģ̬ע��������**
```python
# ƴ������ģ̬
f_all = concat([f_t, f_f, f_tf])  # (B, 768)

# Multi-Head Cross-Attention
f_interact = MultiHeadAttention(
    query=f_weighted,    # Level 1���
    key=f_all,
    value=f_all
)  # (B, 768)

# �в�����
f_interact = f_interact + f_weighted
```

**Level 3: ����ѹ��**
```python
# MLP��ά
f_fused = MLP(
    input_dim=768,
    hidden_dim=384,
    output_dim=128,
    dropout=[0.25, 0.35]
)  # (B, 128)

# �����ں�����
```

#### 5. ˫ͷ������

**Softmax Head** (��ͳ����)
```python
x = Dropout(0.4)(f_fused)
x = Linear(128, 256)(x)
x = BatchNorm1d(256)(x)
x = ReLU()(x)
x = Dropout(0.3)(x)
logits = Linear(256, num_classes)(x)  # (B, 6)
```

**ArcFace Head** (����ѧϰ)
```python
# ������һ��
f_norm = F.normalize(f_fused, p=2, dim=1)

# Ȩ�ع�һ��
W_norm = F.normalize(W, p=2, dim=0)  # (128, 6)

# �����������ƶ�
cos_theta = f_norm @ W_norm  # (B, 6)

# Ŀ�������margin
cos_theta_m = cos(acos(cos_theta[y]) + m)

# ����
logits = s * cos_theta_m  # (B, 6)
```

**����ѵ��**
```python
if training:
    return softmax_logits, arcface_logits, features
else:
    # ����ʱֻ��softmax
    return softmax_logits
```

---

## ? ѵ������

### ���׶�ѵ��

#### �׶�1: �Ա�ѧϰԤѵ�� (30 epochs)
**Ŀ��**: ѧϰͨ�õ��б���������ʾ

```python
# ����: ÿ����������2��ǿ��ǿ�汾
x_view1, x_view2 = strong_augment(x)

# ǰ��: ��ȡ����ͶӰ
z1 = ProjectionHead(Backbone(x_view1))  # (B, 128)
z2 = ProjectionHead(Backbone(x_view2))  # (B, 128)

# ��ʧ: NT-Xent
loss = -log(exp(sim(z1,z2)/��) / ��exp(sim(z1,zk)/��))

# �Ż�: AdamW, lr=1e-3
# ������: CosineAnnealingWarmRestarts
```

**����**:
- ѧϰ��ͬ��ǿ�µĲ�������
- ������������������
- �ṩ���õĳ�ʼ��

#### �׶�2: �мල΢�� (100 epochs)
**Ŀ��**: ��Ԥѵ��������ѧϰ����

```python
# ����Ԥѵ��Ȩ��
model.load_pretrained_backbone('pretrained_weights.pth')

# ǰ��: ˫ͷ���
softmax_logits, arcface_logits, features = model(batch)

# ��ʧ: Focal + ArcFace
loss = focal_loss(softmax_logits, y) + 0.5 * arcface_loss(arcface_logits, y)

# �Ż�: AdamW, lr=5e-4 (��С)
# ������: Warmup + CosineAnnealing
```

**����**:
- ǰ5 epoch: Warmup,ѧϰ����������
- 5-80 epoch: CosineAnnealing
- 80+ epoch: ReduceLROnPlateau
- ��̬����ArcFaceȨ��: 0.3 �� 0.5 �� 0.7

### ������ǿ����

#### Ԥѵ���׶� (ǿ��ǿ)
```python
contrastive_augmentation = [
    GaussianNoise(��=0.05, p=0.8),
    TimeShift(offset=[-50,50], p=0.6),
    AmplitudeScale(scale=[0.7,1.3], p=0.7),
    TimeWarping(ratio=[0.85,1.15], p=0.5),
    RandomMasking(ratio=0.15, p=0.5),
    AddImpulse(num=[1,3], p=0.4),
]
```

#### ΢���׶� (����ʽ)
```python
# Epoch 0-30: ����ǿ
weak_aug = [
    GaussianNoise(��=0.01, p=0.3),
    AmplitudeScale(scale=[0.9,1.1], p=0.3),
]

# Epoch 30-70: �е���ǿ
medium_aug = [
    GaussianNoise(��=0.03, p=0.5),
    TimeShift(offset=[-30,30], p=0.4),
    AmplitudeScale(scale=[0.8,1.2], p=0.5),
]

# Epoch 70-100: ǿ��ǿ
strong_aug = [
    GaussianNoise(��=0.05, p=0.6),
    TimeShift(offset=[-50,50], p=0.5),
    AmplitudeScale(scale=[0.7,1.3], p=0.6),
    TimeWarping(ratio=[0.9,1.1], p=0.4),
    RandomMasking(ratio=0.1, p=0.4),
]
```

#### Mixup��ǿ
```python
# ʱ��mixup (�мල�׶�)
if use_mixup and training:
    x_mix, y_a, y_b, lam = mixup_data(x, y, alpha=0.2)
    logits = model(x_mix)
    loss = lam * loss(logits, y_a) + (1-lam) * loss(logits, y_b)
```

### ��ʧ�������

#### Focal Loss (������ƽ��)
```python
FL(p_t) = -��_t �� (1-p_t)^�� �� log(p_t)

# ����:
�� = 2.0                 # �۽�����
�� = [Ȩ������]          # ���Ȩ��,�����������Զ�����
label_smoothing = 0.1   # ��ǩƽ��
```

**���Ȩ�ؼ���**:
```python
# Effective Number����
def compute_class_weights(class_counts, beta=0.9999):
    effective_num = 1.0 - beta^class_counts
    weights = (1 - beta) / effective_num
    weights = normalize(weights)
    return weights

# ʾ��:
# [160, 120, 180, 150, 100, 150] �� [0.86, 1.05, 0.78, 0.93, 1.32, 0.93]
```

#### ArcFace Loss (����ѧϰ)
```python
L_arc = -log(exp(s?cos(��_yi + m)) / ��exp(s?cos(��_j)))

# ����:
s = 30.0   # scale,����logits��Χ
m = 0.50   # margin,��������
```

**Ч��**: ��С�����ھ�,�������

#### �����ʧ (����ʽȨ��)
```python
L_total = ��1?L_focal + ��2?L_arcface

# ��̬Ȩ��:
epoch 0-30:   ��2 = 0.3
epoch 30-60:  ��2 = 0.5
epoch 60-100: ��2 = 0.7

# ���ڼӴ����ѧϰȨ��
```

### ���򻯼���

```python
# 1. Label Smoothing
y_smooth = (1 - ��) * y_onehot + �� / num_classes
# �� = 0.1

# 2. Dropout (�ֲ����)
# ǳ��: 0.1-0.15 (�����ײ�����)
# �в�: 0.2-0.3 (��ֹ�����)
# ���: 0.3-0.4 (ǿ����)

# 3. Gradient Clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 4. Weight Decay
weight_decay = 5e-4  # ��Ԥѵ����,�������

# 5. Mixup
alpha = 0.2, prob = 0.5

# 6. Early Stopping
patience = 15, monitor = 'val_acc'
```

---

## ? ��������ӻ�

### ����ָ��

```python
from utils.metrics import compute_metrics

metrics = compute_metrics(y_true, y_pred)
# ����:
# - accuracy: ����׼ȷ��
# - precision: ���ྫȷ��
# - recall: �����ٻ���
# - f1_score: ����F1����
# - confusion_matrix: ��������
```

### ��������

```python
from evaluation.confusion_matrix import plot_confusion_matrix

plot_confusion_matrix(
    y_true, y_pred,
    class_names=['Normal', 'Inner Wear', 'Inner Broken', 
                 'Roller Wear', 'Roller Broken', 'Outer Missing'],
    save_path='runs/exp/confusion_matrix.png'
)
```

### t-SNE�������ӻ�

```python
from evaluation.tsne_visualization import visualize_features

visualize_features(
    model, test_loader,
    save_path='runs/exp/tsne.png'
)
```

### ע����Ȩ�ؿ��ӻ�

```python
from evaluation.attention_visualization import visualize_modal_weights

# ���ӻ�ģ̬��Ҫ��
visualize_modal_weights(
    model, test_loader,
    save_path='runs/exp/modal_weights.png'
)
```

### TensorBoard���

```bash
# ����TensorBoard
tensorboard --logdir runs/

# �鿴:
# - ѵ��/��֤��ʧ����
# - ׼ȷ������
# - ѧϰ�ʱ仯
# - �ݶȷֲ�
# - �����ֲ�
```

---

