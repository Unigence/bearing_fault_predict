"""
多模态轴承故障诊断模型 - 完整集成（支持对比学习）
整合三分支Backbone + 多级融合 + 双头分类器 + 对比学习预训练
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """
    对比学习投影头
    用于预训练阶段,将融合特征映射到对比学习空间
    微调时会被移除或不使用
    
    结构: FC(in→hidden) → BN → ReLU → FC(hidden→out) → L2_normalize
    """
    
    def __init__(self, in_features=128, hidden_dim=256, out_features=128):
        """
        Args:
            in_features: 输入特征维度(融合特征维度)
            hidden_dim: 隐藏层维度
            out_features: 输出特征维度(投影空间维度)
        """
        super(ProjectionHead, self).__init__()
        
        self.projection = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            
            nn.Linear(hidden_dim, out_features)
        )
        
    def forward(self, x):
        """
        前向传播并L2归一化
        
        Args:
            x: 融合特征 (B, in_features)
        
        Returns:
            z: 投影后的归一化特征 (B, out_features)
        """
        z = self.projection(x)
        z = F.normalize(z, p=2, dim=1)  # L2归一化到单位球面
        return z


class MultimodalBearingDiagnosisModel(nn.Module):
    """
    多模态轴承故障诊断模型（支持对比学习）
    
    架构流程:
    输入特征 → 三分支Backbone (时域/频域/时频) → 多级融合 → 分类/对比学习
    
    训练流程:
    1. 对比学习预训练: 使用projection_head,不使用classifier
    2. 有监督微调: 使用classifier,不使用projection_head
    
    关键参数:
    - seq_len: 输入序列长度 (默认512)
    - num_classes: 故障类别数 (默认6)
    - modal_dim: 每个模态的输出维度 (默认256)
    - fusion_output_dim: 融合后的特征维度 (默认128)
    - enable_contrastive: 是否启用对比学习模式
    """
    
    def __init__(self,
                 # 数据参数
                 seq_len=512,
                 num_classes=6,
                 
                 # Backbone参数
                 temporal_config='medium',
                 frequency_config='medium',
                 timefreq_config='medium',
                 modal_dim=256,
                 
                 # Fusion参数
                 fusion_type='hierarchical',  # 'hierarchical' or 'hierarchical_v2'
                 fusion_output_dim=128,
                 fusion_num_heads=4,
                 fusion_dropout_l1=0.25,
                 fusion_dropout_l2=0.35,
                 fusion_dropout_l3=0.3,
                 
                 # Head参数
                 head_type='dual',  # 'dual' or 'dual_shared' or 'ensemble'
                 head_hidden_dim=256,
                 arcface_s=30.0,
                 arcface_m=0.50,
                 head_dropout1=0.4,
                 head_dropout2=0.3,
                 
                 # 对比学习参数
                 enable_contrastive=False,
                 projection_hidden_dim=256,
                 projection_out_dim=128):
        """
        Args:
            seq_len: 输入序列长度
            num_classes: 类别数
            
            Backbone参数:
            temporal_config: 时域分支配置 ('small'|'medium'|'large')
            frequency_config: 频域分支配置
            timefreq_config: 时频分支配置
            modal_dim: 每个分支的输出维度
            
            Fusion参数:
            fusion_type: 融合类型
            fusion_output_dim: 融合输出维度
            fusion_num_heads: 注意力头数
            fusion_dropout_*: 各级dropout率
            
            Head参数:
            head_type: 分类头类型
            head_hidden_dim: 隐藏层维度
            arcface_s: ArcFace scale参数
            arcface_m: ArcFace margin参数
            head_dropout*: Dropout率
            
            对比学习参数:
            enable_contrastive: 是否启用对比学习模式
            projection_hidden_dim: 投影头隐藏层维度
            projection_out_dim: 投影特征维度
        """
        super(MultimodalBearingDiagnosisModel, self).__init__()
        
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.modal_dim = modal_dim
        self.fusion_output_dim = fusion_output_dim
        self.enable_contrastive = enable_contrastive
        
        # ==================== 1. 三分支Backbone ====================
        from models.backbone import (
            create_temporal_branch,
            create_frequency_branch,
            create_timefreq_branch
        )
        
        self.temporal_branch = create_temporal_branch(
            config=temporal_config,
            seq_len=seq_len,
            output_dim=modal_dim
        )
        
        self.frequency_branch = create_frequency_branch(
            config=frequency_config,
            output_dim=modal_dim
        )
        
        self.timefreq_branch = create_timefreq_branch(
            config=timefreq_config,
            output_dim=modal_dim
        )
        
        # ==================== 2. 多级特征融合 ====================
        from models.fusion import HierarchicalFusion, HierarchicalFusionV2
        
        if fusion_type == 'hierarchical':
            self.fusion = HierarchicalFusion(
                modal_dim=modal_dim,
                output_dim=fusion_output_dim,
                num_heads=fusion_num_heads,
                dropout_l1=fusion_dropout_l1,
                dropout_l2=fusion_dropout_l2,
                dropout_l3=fusion_dropout_l3
            )
        elif fusion_type == 'hierarchical_v2':
            self.fusion = HierarchicalFusionV2(
                modal_dim=modal_dim,
                output_dim=fusion_output_dim,
                num_heads=fusion_num_heads
            )
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")
        
        # ==================== 3. 分类头 ====================
        from models.heads import (
            DualHead,
            DualHeadWithSharedBackbone,
            EnsembleDualHead
        )
        
        if head_type == 'dual':
            self.classifier = DualHead(
                in_features=fusion_output_dim,
                num_classes=num_classes,
                hidden_dim=head_hidden_dim,
                s=arcface_s,
                m=arcface_m,
                dropout1=head_dropout1,
                dropout2=head_dropout2
            )
        elif head_type == 'dual_shared':
            self.classifier = DualHeadWithSharedBackbone(
                in_features=fusion_output_dim,
                num_classes=num_classes,
                hidden_dim=head_hidden_dim,
                s=arcface_s,
                m=arcface_m,
                dropout=head_dropout1
            )
        elif head_type == 'ensemble':
            self.classifier = EnsembleDualHead(
                in_features=fusion_output_dim,
                num_classes=num_classes,
                hidden_dim=head_hidden_dim,
                s=arcface_s,
                m=arcface_m,
                ensemble_weight=0.7
            )
        else:
            raise ValueError(f"Unknown head_type: {head_type}")
        
        # ==================== 4. 对比学习投影头（可选） ====================
        if enable_contrastive:
            self.projection_head = ProjectionHead(
                in_features=fusion_output_dim,
                hidden_dim=projection_hidden_dim,
                out_features=projection_out_dim
            )
        else:
            self.projection_head = None
    
    def forward(self, batch, mode='supervised', return_features=False):
        """
        前向传播
        
        Args:
            batch: 包含三个模态输入的字典
                   对于有监督模式:
                   {
                       'temporal': (B, 1, 512),
                       'frequency': (B, 1, 257),
                       'timefreq': (B, 1, 64, 128),
                       'labels': (B,)  # 训练模式需要
                   }
                   
                   对于对比学习模式:
                   {
                       'view1': {'temporal', 'frequency', 'timefreq'},
                       'view2': {'temporal', 'frequency', 'timefreq'},
                       'label': (B,)  # SupCon需要,NT-Xent不需要
                   }
            
            mode: 前向传播模式
                - 'supervised': 有监督分类模式
                    - 训练: 返回(softmax_logits, arcface_logits, features)
                    - 推理: 返回logits
                - 'contrastive': 对比学习模式
                    - 返回(z1, z2)投影特征
            
            return_features: 是否返回中间特征（仅supervised模式）
        
        Returns:
            mode='supervised' + training:
                (softmax_logits, arcface_logits, features)
            mode='supervised' + eval:
                logits 或 (logits, features)
            mode='contrastive':
                (z1, z2) 两个视图的投影特征
        """
        if mode == 'supervised':
            return self._forward_supervised(batch, return_features)
        elif mode == 'contrastive':
            return self._forward_contrastive(batch)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'supervised' or 'contrastive'")
    
    def _forward_supervised(self, batch, return_features=False):
        """
        有监督分类前向传播
        
        Args:
            batch: {'temporal', 'frequency', 'timefreq', 'labels'}
            return_features: 是否返回特征
        
        Returns:
            training: (softmax_logits, arcface_logits, features)
            eval: logits 或 (logits, features)
        """
        # ==================== 1. 三分支特征提取 ====================
        feat_temporal = self.temporal_branch(batch['temporal'])       # (B, 256)
        feat_frequency = self.frequency_branch(batch['frequency'])    # (B, 256)
        feat_timefreq = self.timefreq_branch(batch['timefreq'])      # (B, 256)
        
        # ==================== 2. 多级特征融合 ====================
        fused_feat = self.fusion(
            feat_temporal, 
            feat_frequency, 
            feat_timefreq
        )  # (B, fusion_output_dim)
        
        # ==================== 3. 分类 ====================
        if self.training:
            # 训练模式: 返回双头输出
            labels = batch.get('labels', None)
            softmax_logits, arcface_logits, features = self.classifier(
                fused_feat, 
                labels=labels, 
                mode='train'
            )
            return softmax_logits, arcface_logits, features
        else:
            # 推理模式: 只返回logits
            logits = self.classifier(fused_feat, mode='eval')
            if return_features:
                return logits, fused_feat
            return logits
    
    def _forward_contrastive(self, batch):
        """
        对比学习前向传播
        
        Args:
            batch: {'view1': {...}, 'view2': {...}, 'label': ...}
        
        Returns:
            z1, z2: 两个视图的投影特征 (B, projection_out_dim)
        """
        if self.projection_head is None:
            raise RuntimeError(
                "Projection head not initialized. "
                "Set enable_contrastive=True when creating the model."
            )
        
        # ==================== View 1 ====================
        feat_t1 = self.temporal_branch(batch['view1']['temporal'])
        feat_f1 = self.frequency_branch(batch['view1']['frequency'])
        feat_tf1 = self.timefreq_branch(batch['view1']['timefreq'])
        fused_feat1 = self.fusion(feat_t1, feat_f1, feat_tf1)
        z1 = self.projection_head(fused_feat1)  # (B, projection_out_dim)
        
        # ==================== View 2 ====================
        feat_t2 = self.temporal_branch(batch['view2']['temporal'])
        feat_f2 = self.frequency_branch(batch['view2']['frequency'])
        feat_tf2 = self.timefreq_branch(batch['view2']['timefreq'])
        fused_feat2 = self.fusion(feat_t2, feat_f2, feat_tf2)
        z2 = self.projection_head(fused_feat2)  # (B, projection_out_dim)
        
        return z1, z2
    
    def get_fusion_weights(self, batch):
        """
        获取模态融合权重 (用于可视化和分析)
        
        Args:
            batch: 输入batch
        
        Returns:
            modal_weights: (B, 3) 模态权重 [时域, 频域, 时频]
        """
        # 提取特征
        feat_temporal = self.temporal_branch(batch['temporal'])
        feat_frequency = self.frequency_branch(batch['frequency'])
        feat_timefreq = self.timefreq_branch(batch['timefreq'])
        
        # 只计算Level 1的模态权重
        _, modal_weights = self.fusion.level1_modal_importance(
            feat_temporal, feat_frequency, feat_timefreq
        )
        
        return modal_weights
    
    def get_all_features(self, batch):
        """
        获取所有中间特征 (用于可视化和分析)
        
        Args:
            batch: 输入batch
        
        Returns:
            all_features: 包含所有中间特征的字典
        """
        # 三分支特征
        feat_temporal = self.temporal_branch(batch['temporal'])
        feat_frequency = self.frequency_branch(batch['frequency'])
        feat_timefreq = self.timefreq_branch(batch['timefreq'])
        
        # 融合特征(带中间信息)
        fused_feat, fusion_intermediate = self.fusion(
            feat_temporal, feat_frequency, feat_timefreq,
            return_intermediate=True
        )
        
        all_features = {
            # 原始分支特征
            'feat_temporal': feat_temporal,
            'feat_frequency': feat_frequency,
            'feat_timefreq': feat_timefreq,
            
            # 融合特征
            'fused_feat': fused_feat,
            'weighted_feat': fusion_intermediate['weighted_feat'],
            'interact_feat': fusion_intermediate['interact_feat'],
            
            # 权重信息
            'modal_weights': fusion_intermediate['modal_weights'],
            'attn_weights': fusion_intermediate['attn_weights'],
        }
        
        return all_features
    
    def freeze_backbone(self, freeze_ratio=0.5):
        """
        冻结部分Backbone层 (用于微调)
        
        Args:
            freeze_ratio: 冻结的比例 [0, 1]
                0.0: 不冻结
                0.5: 冻结前50%的层
                1.0: 冻结所有backbone
        """
        # 获取所有backbone参数
        backbone_params = []
        for module in [self.temporal_branch, self.frequency_branch, 
                      self.timefreq_branch, self.fusion]:
            backbone_params.extend(list(module.parameters()))
        
        # 计算要冻结的参数数量
        num_freeze = int(len(backbone_params) * freeze_ratio)
        
        # 冻结前N个参数
        for i, param in enumerate(backbone_params):
            param.requires_grad = (i >= num_freeze)
        
        print(f"✓ Frozen {num_freeze}/{len(backbone_params)} backbone parameters")
    
    def unfreeze_all(self):
        """解冻所有参数"""
        for param in self.parameters():
            param.requires_grad = True
        print("✓ All parameters unfrozen")
    
    def load_pretrained_backbone(self, checkpoint_path, strict=False):
        """
        加载预训练的backbone权重
        
        Args:
            checkpoint_path: checkpoint文件路径
            strict: 是否严格加载（True会报错如果有不匹配的键）
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 提取state_dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # 只加载backbone的权重,排除projection_head和classifier
        model_dict = self.state_dict()
        pretrained_dict = {}
        
        for k, v in state_dict.items():
            # 跳过投影头和分类器
            if 'projection_head' in k or 'classifier' in k:
                continue
            # 只加载存在于当前模型的键
            if k in model_dict:
                pretrained_dict[k] = v
        
        # 更新模型字典
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict, strict=strict)
        
        print(f"✓ Loaded pretrained backbone from {checkpoint_path}")
        print(f"  Loaded {len(pretrained_dict)}/{len(state_dict)} parameters")
    
    def count_parameters(self):
        """
        统计模型参数
        
        Returns:
            param_dict: 包含各部分参数量的字典
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        temporal_params = sum(p.numel() for p in self.temporal_branch.parameters())
        frequency_params = sum(p.numel() for p in self.frequency_branch.parameters())
        timefreq_params = sum(p.numel() for p in self.timefreq_branch.parameters())
        fusion_params = sum(p.numel() for p in self.fusion.parameters())
        classifier_params = sum(p.numel() for p in self.classifier.parameters())
        
        param_dict = {
            'total': total_params,
            'trainable': trainable_params,
            'temporal_branch': temporal_params,
            'frequency_branch': frequency_params,
            'timefreq_branch': timefreq_params,
            'fusion': fusion_params,
            'classifier': classifier_params
        }
        
        if self.projection_head is not None:
            projection_params = sum(p.numel() for p in self.projection_head.parameters())
            param_dict['projection_head'] = projection_params
        
        return param_dict


def create_model(config='default', enable_contrastive=False, **kwargs):
    """
    模型工厂函数
    
    Args:
        config: 'small' | 'medium' | 'large' | 'default'
        enable_contrastive: 是否启用对比学习模式
        **kwargs: 覆盖默认参数
    
    Returns:
        model: MultimodalBearingDiagnosisModel实例
    """
    configs = {
        'small': {
            'modal_dim': 128,
            'fusion_output_dim': 64,
            'temporal_config': 'small',
            'frequency_config': 'small',
            'timefreq_config': 'small',
            'head_hidden_dim': 128,
            'projection_hidden_dim': 128,
            'projection_out_dim': 64
        },
        'medium': {
            'modal_dim': 256,
            'fusion_output_dim': 128,
            'temporal_config': 'medium',
            'frequency_config': 'medium',
            'timefreq_config': 'medium',
            'head_hidden_dim': 256,
            'projection_hidden_dim': 256,
            'projection_out_dim': 128
        },
        'large': {
            'modal_dim': 512,
            'fusion_output_dim': 256,
            'temporal_config': 'large',
            'frequency_config': 'large',
            'timefreq_config': 'large',
            'head_hidden_dim': 512,
            'projection_hidden_dim': 512,
            'projection_out_dim': 256
        }
    }
    
    params = configs.get(config, configs['medium'])
    params['enable_contrastive'] = enable_contrastive
    params.update(kwargs)
    
    return MultimodalBearingDiagnosisModel(**params)


# 单元测试
if __name__ == '__main__':
    print("=" * 80)
    print("测试多模态轴承故障诊断模型（支持对比学习）")
    print("=" * 80)
    
    try:
        # ==================== 测试1: 有监督模式 ====================
        print("\n[测试1] 有监督分类模式")
        print("-" * 80)
        
        model_supervised = create_model(
            config='medium',
            enable_contrastive=False
        )
        
        print(f"✓ 模型创建成功 (有监督模式)")
        param_dict = model_supervised.count_parameters()
        print(f"  总参数: {param_dict['total']:,}")
        print(f"  投影头: {'是' if 'projection_head' in param_dict else '否'}")
        
        # 创建测试数据
        batch_size = 4
        batch_supervised = {
            'temporal': torch.randn(batch_size, 1, 512),
            'frequency': torch.randn(batch_size, 1, 257),
            'timefreq': torch.randn(batch_size, 1, 64, 128),
            'labels': torch.randint(0, 6, (batch_size,))
        }
        
        # 训练模式
        model_supervised.train()
        softmax_logits, arcface_logits, features = model_supervised(
            batch_supervised, mode='supervised'
        )
        
        print(f"✓ 训练模式前向传播成功")
        print(f"  Softmax logits: {softmax_logits.shape}")
        print(f"  ArcFace logits: {arcface_logits.shape}")
        print(f"  Features: {features.shape}")
        
        # 推理模式
        model_supervised.eval()
        with torch.no_grad():
            logits = model_supervised(batch_supervised, mode='supervised')
        
        print(f"✓ 推理模式前向传播成功")
        print(f"  Logits: {logits.shape}")
        
        # ==================== 测试2: 对比学习模式 ====================
        print("\n[测试2] 对比学习模式")
        print("-" * 80)
        
        model_contrastive = create_model(
            config='medium',
            enable_contrastive=True
        )
        
        print(f"✓ 模型创建成功 (对比学习模式)")
        param_dict = model_contrastive.count_parameters()
        print(f"  总参数: {param_dict['total']:,}")
        print(f"  投影头参数: {param_dict.get('projection_head', 0):,}")
        
        # 创建对比学习数据
        batch_contrastive = {
            'view1': {
                'temporal': torch.randn(batch_size, 1, 512),
                'frequency': torch.randn(batch_size, 1, 257),
                'timefreq': torch.randn(batch_size, 1, 64, 128)
            },
            'view2': {
                'temporal': torch.randn(batch_size, 1, 512),
                'frequency': torch.randn(batch_size, 1, 257),
                'timefreq': torch.randn(batch_size, 1, 64, 128)
            },
            'label': torch.randint(0, 6, (batch_size,))
        }
        
        model_contrastive.train()
        z1, z2 = model_contrastive(batch_contrastive, mode='contrastive')
        
        print(f"✓ 对比学习前向传播成功")
        print(f"  z1: {z1.shape}")
        print(f"  z2: {z2.shape}")
        print(f"  z1归一化: {torch.norm(z1, p=2, dim=1).mean():.4f} (应接近1.0)")
        print(f"  z2归一化: {torch.norm(z2, p=2, dim=1).mean():.4f} (应接近1.0)")
        
        # ==================== 测试3: 权重加载 ====================
        print("\n[测试3] 测试权重加载")
        print("-" * 80)
        
        # 保存对比学习模型的权重
        torch.save({
            'model_state_dict': model_contrastive.state_dict(),
            'epoch': 30
        }, '/tmp/test_pretrained.pth')
        
        # 创建新的有监督模型并加载预训练权重
        model_finetune = create_model(
            config='medium',
            enable_contrastive=False
        )
        
        model_finetune.load_pretrained_backbone('/tmp/test_pretrained.pth')
        
        # 测试冻结
        model_finetune.freeze_backbone(freeze_ratio=0.5)
        param_dict = model_finetune.count_parameters()
        print(f"  冻结后可训练参数: {param_dict['trainable']:,}")
        
        model_finetune.unfreeze_all()
        param_dict = model_finetune.count_parameters()
        print(f"  解冻后可训练参数: {param_dict['trainable']:,}")
        
        # ==================== 测试4: 不同配置 ====================
        print("\n[测试4] 不同配置")
        print("-" * 80)
        
        for config in ['small', 'medium', 'large']:
            model_test = create_model(config=config, enable_contrastive=True)
            params = model_test.count_parameters()
            print(f"  {config.capitalize():8s} - 总参数: {params['total']:,} | "
                  f"投影头: {params.get('projection_head', 0):,}")
        
        print("\n" + "=" * 80)
        print("✅ 所有测试通过!")
        print("=" * 80)
        
    except ImportError as e:
        print(f"\n⚠️  ImportError: {e}")
        print("注意: 测试需要所有依赖模块在正确的路径")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
