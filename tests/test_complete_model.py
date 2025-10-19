"""
完整模型测试脚本
测试整个训练流程: 数据加载 → 模型前向 → 损失计算 → 反向传播
"""
import torch
import torch.nn.functional as F


def test_complete_pipeline():
    """测试完整的训练流程"""

    print("=" * 80)
    print("轴承故障诊断模型 - 完整流程测试")
    print("=" * 80)

    # ==================== 1. 模型创建 ====================
    print("\n[Step 1] 创建模型...")

    from models.multimodal_model import create_model

    model = create_model(
        config='medium',
        seq_len=512,
        num_classes=6,
        fusion_type='hierarchical',
        head_type='dual'
    )

    # 参数统计
    param_dict = model.count_parameters()
    print(f"✓ 模型创建成功")
    print(f"  - 总参数量: {param_dict['total']:,}")
    print(f"  - 可训练参数: {param_dict['trainable']:,}")

    # ==================== 2. 数据准备 ====================
    print("\n[Step 2] 准备测试数据...")

    batch_size = 8
    seq_len = 512
    num_classes = 6

    # 模拟一个batch的数据
    batch = {
        'temporal': torch.randn(batch_size, 1, seq_len),  # 时域信号
        'frequency': torch.randn(batch_size, 1, 257),  # FFT频谱
        'timefreq': torch.randn(batch_size, 1, 64, 128),  # 时频图(STFT)
        'labels': torch.randint(0, num_classes, (batch_size,))  # 标签
    }

    print(f"✓ 数据准备完成")
    print(f"  - Batch size: {batch_size}")
    print(f"  - 时域信号: {batch['temporal'].shape}")
    print(f"  - 频域谱: {batch['frequency'].shape}")
    print(f"  - 时频图: {batch['timefreq'].shape}")
    print(f"  - 标签分布: {torch.bincount(batch['labels'])}")

    # ==================== 3. 前向传播 ====================
    print("\n[Step 3] 测试前向传播...")

    model.train()

    # 训练模式前向传播
    softmax_logits, arcface_logits, features = model(batch, mode='train')

    print(f"✓ 前向传播成功")
    print(f"  - Softmax logits: {softmax_logits.shape}")
    print(f"  - ArcFace logits: {arcface_logits.shape}")
    print(f"  - Features: {features.shape}")
    print(f"  - Features L2 norm: {torch.norm(features, p=2, dim=1).mean():.4f}")

    # ==================== 4. 损失计算 ====================
    print("\n[Step 4] 计算损失...")

    from losses.combined_loss import CombinedLoss
    from losses.focal_loss import compute_class_weights

    # 模拟类别不平衡的数据分布
    class_counts = [160, 120, 180, 150, 100, 150]
    class_weights = compute_class_weights(class_counts, method='effective_num')

    # 创建组合损失
    criterion = CombinedLoss(
        num_classes=num_classes,
        focal_alpha=class_weights,
        focal_gamma=2.0,
        focal_weight=1.0,
        arcface_weight=0.5,
        label_smoothing=0.1
    )

    # 计算损失
    total_loss, loss_dict = criterion(
        softmax_logits,
        arcface_logits,
        batch['labels']
    )

    print(f"✓ 损失计算成功")
    print(f"  - 总损失: {loss_dict['total']:.4f}")
    print(f"  - Focal Loss: {loss_dict['focal']:.4f}")
    print(f"  - ArcFace Loss: {loss_dict['arcface']:.4f}")

    # ==================== 5. 反向传播 ====================
    print("\n[Step 5] 测试反向传播...")

    # 创建优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-4,
        weight_decay=5e-4
    )

    # 梯度清零
    optimizer.zero_grad()

    # 反向传播
    total_loss.backward()

    # 梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # 统计梯度
    grad_norms = []
    for param in model.parameters():
        if param.grad is not None:
            grad_norms.append(param.grad.norm().item())

    print(f"✓ 反向传播成功")
    print(f"  - 平均梯度范数: {sum(grad_norms) / len(grad_norms):.6f}")
    print(f"  - 最大梯度范数: {max(grad_norms):.6f}")
    print(f"  - 最小梯度范数: {min(grad_norms):.6f}")

    # 更新参数
    optimizer.step()
    print(f"✓ 参数更新成功")

    # ==================== 6. 推理模式 ====================
    print("\n[Step 6] 测试推理模式...")

    model.eval()

    with torch.no_grad():
        # 推理
        logits = model(batch, mode='eval')

        # 计算概率
        probs = F.softmax(logits, dim=1)

        # 预测
        preds = torch.argmax(probs, dim=1)

        # 计算准确率
        accuracy = (preds == batch['labels']).float().mean()

    print(f"✓ 推理成功")
    print(f"  - Logits: {logits.shape}")
    print(f"  - Predictions: {preds}")
    print(f"  - Ground truth: {batch['labels']}")
    print(f"  - Accuracy: {accuracy.item() * 100:.2f}%")

    # ==================== 7. 特征可视化准备 ====================
    print("\n[Step 7] 提取可视化特征...")

    with torch.no_grad():
        # 获取所有中间特征
        all_features = model.get_all_features(batch)

        # 获取模态权重
        modal_weights = model.get_fusion_weights(batch)

    print(f"✓ 特征提取成功")
    print(f"  - 时域特征: {all_features['feat_temporal'].shape}")
    print(f"  - 频域特征: {all_features['feat_frequency'].shape}")
    print(f"  - 时频特征: {all_features['feat_timefreq'].shape}")
    print(f"  - 融合特征: {all_features['fused_feat'].shape}")

    print(f"\n  示例模态权重:")
    print(f"    时域: {modal_weights[0, 0]:.3f}")
    print(f"    频域: {modal_weights[0, 1]:.3f}")
    print(f"    时频: {modal_weights[0, 2]:.3f}")

    # ==================== 8. 模型冻结测试 ====================
    print("\n[Step 8] 测试模型冻结...")

    # 冻结50%的backbone
    model.freeze_backbone(freeze_ratio=0.5)
    param_dict_frozen = model.count_parameters()

    print(f"✓ Backbone冻结成功")
    print(f"  - 冻结前可训练: {param_dict['trainable']:,}")
    print(f"  - 冻结后可训练: {param_dict_frozen['trainable']:,}")
    print(f"  - 减少: {param_dict['trainable'] - param_dict_frozen['trainable']:,}")

    # 解冻
    model.unfreeze_all()
    param_dict_unfrozen = model.count_parameters()
    print(f"✓ 解冻成功")
    print(f"  - 解冻后可训练: {param_dict_unfrozen['trainable']:,}")

    # ==================== 9. 渐进式训练测试 ====================
    print("\n[Step 9] 测试渐进式训练策略...")

    from losses.combined_loss import ProgressiveCombinedLoss

    progressive_criterion = ProgressiveCombinedLoss(
        num_classes=num_classes,
        focal_alpha=class_weights,
        focal_gamma_init=2.0,
        focal_gamma_min=1.0,
        arcface_weight_init=0.3,
        arcface_weight_max=0.7
    )

    print(f"✓ 模拟训练过程中的损失变化:")

    model.train()
    for epoch in [0, 25, 50, 75, 100]:
        progress = epoch / 100
        progressive_criterion.update_schedule(progress)

        # 前向传播
        softmax_logits, arcface_logits, _ = model(batch, mode='train')

        # 计算损失
        total_loss, loss_dict = progressive_criterion(
            softmax_logits, arcface_logits, batch['labels']
        )

        print(f"  Epoch {epoch:3d}: "
              f"Gamma={loss_dict['focal_gamma']:.2f}, "
              f"ArcFace_w={loss_dict['arcface_weight']:.2f}, "
              f"Loss={loss_dict['total']:.4f}")

    # ==================== 总结 ====================
    print("\n" + "=" * 80)
    print("✅ 所有测试通过!")
    print("=" * 80)

    print("\n模型总结:")
    print(f"  • 模型规模: Medium")
    print(f"  • 总参数量: {param_dict['total']:,}")
    print(f"  • 模型大小: ~{param_dict['total'] * 4 / 1024 / 1024:.2f} MB (FP32)")
    print(f"  • 支持功能:")
    print(f"    - 三分支多模态输入 ✓")
    print(f"    - 多级自适应融合 ✓")
    print(f"    - 双头联合训练 ✓")
    print(f"    - Focal Loss + ArcFace ✓")
    print(f"    - 渐进式训练策略 ✓")
    print(f"    - Backbone冻结/微调 ✓")
    print(f"    - 特征可视化支持 ✓")


if __name__ == '__main__':
    import sys
    import os

    # 添加父目录到路径
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    try:
        test_complete_pipeline()
    except ImportError as e:
        print(f"\n⚠️  ImportError: {e}")
        print("\n解决方案:")
        print("  1. 确保所有模块文件在正确的位置")
        print("  2. 检查import路径是否正确")
        print("  3. 运行前请先阅读README_FIXES.md")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()
