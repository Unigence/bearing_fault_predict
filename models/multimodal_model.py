"""
多模态轴承故障诊断模型 - 完整集成
整合三分支Backbone + 多级融合 + 双头分类器
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultimodalBearingDiagnosisModel(nn.Module):
    """
    多模态轴承故障诊断模型
    
    架构流程:
    输入信号 → 三分支Backbone (时域/频域/时频) → 多级融合 → 双头分类器 → 输出
    
    关键参数:
    - seq_len: 输入序列长度 (默认512)
    - num_classes: 故障类别数 (默认6)
    - modal_dim: 每个模态的输出维度 (默认256)
    - fusion_output_dim: 融合后的特征维度 (默认128)
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
                 head_dropout2=0.3):
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
        """
        super(MultimodalBearingDiagnosisModel, self).__init__()
        
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.modal_dim = modal_dim
        self.fusion_output_dim = fusion_output_dim
        
        # ==================== 导入必要的模块 ====================
        # 注意: 这里假设模块已经在合适的路径
        # 实际使用时需要调整import路径
        
        # ==================== 1. 三分支Backbone ====================
        from models.backbone import (
            create_temporal_branch,
            create_frequency_branch,
            create_timefreq_branch
        )
        
        # 时域分支
        self.temporal_branch = create_temporal_branch(
            config=temporal_config,
            seq_len=seq_len,
            output_dim=modal_dim
        )
        
        # 频域分支
        freq_len = seq_len // 2 + 1  # FFT后的频谱长度
        self.frequency_branch = create_frequency_branch(
            config=frequency_config,
            freq_len=freq_len,
            output_dim=modal_dim
        )
        
        # 时频分支
        # STFT参数: n_fft=128, hop_length=64, 结果约为(64, 8)
        # 可以根据实际调整
        self.timefreq_branch = create_timefreq_branch(
            config=timefreq_config,
            input_size=(64, 128),  # 根据STFT参数调整
            output_dim=modal_dim
        )
        
        # ==================== 2. 特征融合模块 ====================
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
                num_heads=fusion_num_heads * 2,  # V2使用更多头
                num_interaction_layers=2
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
                ensemble_weight=0.7  # Softmax权重
            )
        else:
            raise ValueError(f"Unknown head_type: {head_type}")
    
    def forward(self, batch, mode='train', return_features=False):
        """
        前向传播
        
        Args:
            batch: 包含三个模态输入的字典
                   {
                       'temporal': (B, 1, 512),      # 时域信号
                       'frequency': (B, 1, 257),     # FFT频谱
                       'timefreq': (B, 1, 64, 128)   # 时频图
                   }
            mode: 'train' 或 'eval'
            return_features: 是否返回中间特征
        
        Returns:
            如果mode='train':
                (softmax_logits, arcface_logits, features)
            如果mode='eval':
                logits 或 (logits, features)
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
        )  # (B, 128)
        
        # ==================== 3. 分类 ====================
        if mode == 'train':
            # 训练模式: 返回双头输出
            # labels参数在损失计算时提供给classifier
            softmax_logits, arcface_logits, normalized_feat = self.classifier(
                fused_feat, 
                labels=batch.get('labels', None),
                mode='train'
            )
            
            if return_features:
                feature_dict = {
                    'feat_temporal': feat_temporal,
                    'feat_frequency': feat_frequency,
                    'feat_timefreq': feat_timefreq,
                    'fused_feat': fused_feat,
                    'normalized_feat': normalized_feat
                }
                return softmax_logits, arcface_logits, feature_dict
            
            return softmax_logits, arcface_logits, normalized_feat
        
        else:
            # 推理模式: 只返回Softmax输出
            logits = self.classifier(fused_feat, mode='eval')
            
            if return_features:
                feature_dict = {
                    'feat_temporal': feat_temporal,
                    'feat_frequency': feat_frequency,
                    'feat_timefreq': feat_timefreq,
                    'fused_feat': fused_feat
                }
                return logits, feature_dict
            
            return logits
    
    def get_fusion_weights(self, batch):
        """
        获取模态融合权重 (用于可视化分析)
        
        Args:
            batch: 输入batch
        
        Returns:
            modal_weights: 模态权重 (B, 3)
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
                         0: 不冻结
                         0.5: 冻结前50%的层
                         1.0: 冻结所有层
        """
        if freeze_ratio <= 0:
            return
        
        def freeze_module(module, ratio):
            """冻结模块的前ratio比例的参数"""
            params = list(module.parameters())
            num_freeze = int(len(params) * ratio)
            
            for i, param in enumerate(params):
                if i < num_freeze:
                    param.requires_grad = False
        
        # 冻结三个分支
        freeze_module(self.temporal_branch, freeze_ratio)
        freeze_module(self.frequency_branch, freeze_ratio)
        freeze_module(self.timefreq_branch, freeze_ratio)
        
        print(f"✓ 已冻结Backbone的前{freeze_ratio*100:.0f}%参数")
    
    def unfreeze_all(self):
        """解冻所有参数"""
        for param in self.parameters():
            param.requires_grad = True
        print("✓ 已解冻所有参数")
    
    def count_parameters(self):
        """统计模型参数量"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # 分模块统计
        temporal_params = sum(p.numel() for p in self.temporal_branch.parameters())
        frequency_params = sum(p.numel() for p in self.frequency_branch.parameters())
        timefreq_params = sum(p.numel() for p in self.timefreq_branch.parameters())
        fusion_params = sum(p.numel() for p in self.fusion.parameters())
        classifier_params = sum(p.numel() for p in self.classifier.parameters())
        
        param_dict = {
            'total': total,
            'trainable': trainable,
            'temporal_branch': temporal_params,
            'frequency_branch': frequency_params,
            'timefreq_branch': timefreq_params,
            'fusion': fusion_params,
            'classifier': classifier_params
        }
        
        return param_dict


def create_model(config='default', **kwargs):
    """
    模型工厂函数
    
    Args:
        config: 'small' | 'medium' | 'large' | 'default'
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
            'head_hidden_dim': 128
        },
        'medium': {
            'modal_dim': 256,
            'fusion_output_dim': 128,
            'temporal_config': 'medium',
            'frequency_config': 'medium',
            'timefreq_config': 'medium',
            'head_hidden_dim': 256
        },
        'large': {
            'modal_dim': 512,
            'fusion_output_dim': 256,
            'temporal_config': 'large',
            'frequency_config': 'large',
            'timefreq_config': 'large',
            'head_hidden_dim': 512
        }
    }
    
    params = configs.get(config, configs['medium'])
    params.update(kwargs)
    
    return MultimodalBearingDiagnosisModel(**params)


# 单元测试
if __name__ == '__main__':
    print("=" * 70)
    print("测试多模态轴承故障诊断模型")
    print("=" * 70)
    
    # 注意: 这个测试需要所有依赖模块都在正确的路径
    # 实际运行时可能需要调整import路径
    
    try:
        # 创建模型
        model = create_model(config='medium')
        
        print("\n1. 模型结构:")
        print(f"✓ 模型创建成功")
        
        # 参数统计
        param_dict = model.count_parameters()
        print(f"\n2. 参数统计:")
        print(f"✓ 总参数: {param_dict['total']:,}")
        print(f"✓ 可训练: {param_dict['trainable']:,}")
        print(f"✓ 时域分支: {param_dict['temporal_branch']:,}")
        print(f"✓ 频域分支: {param_dict['frequency_branch']:,}")
        print(f"✓ 时频分支: {param_dict['timefreq_branch']:,}")
        print(f"✓ 融合模块: {param_dict['fusion']:,}")
        print(f"✓ 分类器: {param_dict['classifier']:,}")
        
        # 创建测试数据
        batch_size = 4
        batch = {
            'temporal': torch.randn(batch_size, 1, 512),
            'frequency': torch.randn(batch_size, 1, 257),
            'timefreq': torch.randn(batch_size, 1, 64, 128),
            'labels': torch.randint(0, 6, (batch_size,))
        }
        
        # 测试训练模式
        print("\n3. 测试训练模式...")
        model.train()
        softmax_logits, arcface_logits, features = model(batch, mode='train')
        
        print(f"✓ Softmax logits: {softmax_logits.shape}")
        print(f"✓ ArcFace logits: {arcface_logits.shape}")
        print(f"✓ Features: {features.shape}")
        
        # 测试推理模式
        print("\n4. 测试推理模式...")
        model.eval()
        with torch.no_grad():
            logits = model(batch, mode='eval')
        
        print(f"✓ Logits: {logits.shape}")
        probs = F.softmax(logits, dim=1)
        print(f"✓ Probabilities sum: {probs.sum(dim=1)}")
        
        # 测试获取模态权重
        print("\n5. 测试获取模态权重...")
        with torch.no_grad():
            modal_weights = model.get_fusion_weights(batch)
        
        print(f"✓ Modal weights shape: {modal_weights.shape}")
        print(f"✓ Sample weights: {modal_weights[0]}")
        print(f"   时域: {modal_weights[0, 0]:.3f}")
        print(f"   频域: {modal_weights[0, 1]:.3f}")
        print(f"   时频: {modal_weights[0, 2]:.3f}")
        
        # 测试冻结/解冻
        print("\n6. 测试冻结/解冻功能...")
        model.freeze_backbone(freeze_ratio=0.5)
        param_dict = model.count_parameters()
        print(f"✓ 冻结后可训练参数: {param_dict['trainable']:,}")
        
        model.unfreeze_all()
        param_dict = model.count_parameters()
        print(f"✓ 解冻后可训练参数: {param_dict['trainable']:,}")
        
        # 测试不同配置
        print("\n7. 测试不同配置...")
        for config in ['small', 'medium', 'large']:
            model_test = create_model(config=config)
            params = model_test.count_parameters()
            print(f"✓ {config.capitalize():8s} - 参数量: {params['total']:,}")
        
        print("\n" + "=" * 70)
        print("✅ 所有测试通过!")
        print("=" * 70)
        
    except ImportError as e:
        print(f"\n⚠️  ImportError: {e}")
        print("注意: 测试需要所有依赖模块在正确的路径")
        print("实际使用时请确保模块导入路径正确")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
