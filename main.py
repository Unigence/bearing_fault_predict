"""
主入口脚本
实现完整的训练流程:
1. 对比学习预训练(可选)
2. 有监督微调
3. 保存权重和可视化结果
"""
import argparse
import torch
import os
import sys
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.config_parser import ModelConfigParser, TrainConfigParser, AugmentationConfigParser
from training.pretrain.pretrain_launcher import launch_pretrain
from training.finetune.finetune_launcher import launch_finetune
from utils.seed import set_seed


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='轴承故障诊断模型训练')
    
    parser.add_argument(
        '--config_dir',
        type=str,
        default='configs',
        help='配置文件目录'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['pretrain', 'finetune', 'full'],
        default='full',
        help='训练模式: pretrain(仅预训练), finetune(仅微调), full(完整流程)'
    )
    
    parser.add_argument(
        '--pretrained_weights',
        type=str,
        default=None,
        help='预训练权重路径(用于finetune模式)'
    )
    
    parser.add_argument(
        '--experiment_name',
        type=str,
        default=None,
        help='实验名称(默认使用时间戳)'
    )
    
    parser.add_argument(
        '--skip_pretrain',
        action='store_true',
        help='跳过预训练,直接有监督训练'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='训练设备(cuda/cpu,覆盖配置文件)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='随机种子(覆盖配置文件)'
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    print("=" * 80)
    print("轴承故障诊断模型训练")
    print("=" * 80)
    print(f"训练模式: {args.mode}")
    print(f"配置目录: {args.config_dir}")
    
    # 加载配置
    print("\n加载配置文件...")
    try:
        model_config = ModelConfigParser(f'{args.config_dir}/model_config.yaml')
        train_config = TrainConfigParser(f'{args.config_dir}/train_config.yaml')
        aug_config = AugmentationConfigParser(f'{args.config_dir}/augmentation_config.yaml')
        print("✓ 配置文件加载成功")
    except FileNotFoundError as e:
        print(f"❌ 配置文件加载失败: {e}")
        sys.exit(1)
    
    # 覆盖配置(如果命令行指定)
    if args.device:
        train_config.set('device.type', args.device)
        print(f"✓ 覆盖设备配置: {args.device}")
    
    if args.seed:
        train_config.set('seed', args.seed)
        print(f"✓ 覆盖随机种子: {args.seed}")
    
    # 检查训练模式
    training_pipeline = train_config.get_training_mode()
    use_pretrain = training_pipeline == 'pretrain_finetune' and not args.skip_pretrain
    
    if args.mode == 'pretrain':
        use_pretrain = True
        skip_finetune = True
    elif args.mode == 'finetune':
        use_pretrain = False
        skip_finetune = False
    else:  # full
        skip_finetune = False
    
    print(f"\n训练流程:")
    print(f"  - 对比学习预训练: {'启用' if use_pretrain else '跳过'}")
    print(f"  - 有监督微调: {'启用' if not skip_finetune else '跳过'}")
    
    # 创建实验名称
    if args.experiment_name:
        experiment_name = args.experiment_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"exp_{timestamp}"
    
    pretrained_weights_path = args.pretrained_weights
    
    # ============================================================
    # 阶段1: 对比学习预训练
    # ============================================================
    if use_pretrain:
        print("\n" + "=" * 80)
        print("阶段1: 对比学习预训练")
        print("=" * 80)
        
        pretrain_exp_name = f"{experiment_name}_pretrain"
        
        try:
            model, pretrain_dir, pretrained_weights_path = launch_pretrain(
                model_config=model_config,
                train_config=train_config,
                aug_config=aug_config,
                experiment_name=pretrain_exp_name
            )
            
            print(f"\n✓ 预训练阶段完成")
            print(f"  - 实验目录: {pretrain_dir}")
            print(f"  - 权重路径: {pretrained_weights_path}")
            
        except Exception as e:
            print(f"\n❌ 预训练阶段失败: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    # ============================================================
    # 阶段2: 有监督微调
    # ============================================================
    if not skip_finetune:
        print("\n" + "=" * 80)
        print("阶段2: 有监督微调")
        print("=" * 80)
        
        finetune_exp_name = f"{experiment_name}_finetune"
        
        try:
            model, finetune_dir, final_model_path = launch_finetune(
                model_config=model_config,
                train_config=train_config,
                aug_config=aug_config,
                pretrained_weights_path=pretrained_weights_path,
                experiment_name=finetune_exp_name
            )
            
            print(f"\n✓ 微调阶段完成")
            print(f"  - 实验目录: {finetune_dir}")
            print(f"  - 模型路径: {final_model_path}")
            
        except Exception as e:
            print(f"\n❌ 微调阶段失败: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    # ============================================================
    # 训练完成总结
    # ============================================================
    print("\n" + "=" * 80)
    print("✅ 训练流程完成!")
    print("=" * 80)
    
    if use_pretrain and not skip_finetune:
        print("\n完整训练流程已完成:")
        print(f"1. 预训练: {pretrain_dir}")
        print(f"   权重: {pretrained_weights_path}")
        print(f"2. 微调: {finetune_dir}")
        print(f"   模型: {final_model_path}")
    elif use_pretrain:
        print("\n预训练已完成:")
        print(f"- 目录: {pretrain_dir}")
        print(f"- 权重: {pretrained_weights_path}")
    elif not skip_finetune:
        print("\n微调已完成:")
        print(f"- 目录: {finetune_dir}")
        print(f"- 模型: {final_model_path}")
    
    print("\n训练结果包含:")
    print("  - 模型权重(.pth文件)")
    print("  - 训练曲线(visualizations/)")
    print("  - Checkpoint(checkpoints/)")
    print("  - 配置文件(config.yaml)")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
