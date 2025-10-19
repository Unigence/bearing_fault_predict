"""
随机种子设置
确保实验可复现
"""
import torch
import numpy as np
import random
import os


def set_seed(seed: int = 42):
    """
    设置所有随机种子
    
    Args:
        seed: 随机种子值
    """
    # Python random
    random.seed(seed)
    
    # Numpy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU
    
    # cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Python hash
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"✓ 随机种子已设置: {seed}")


if __name__ == '__main__':
    """测试代码"""
    print("=" * 70)
    print("随机种子测试")
    print("=" * 70)
    
    # 测试设置种子
    set_seed(42)
    
    # 生成一些随机数验证
    print("\n生成随机数验证:")
    print(f"  random.random(): {random.random():.6f}")
    print(f"  np.random.rand(): {np.random.rand():.6f}")
    print(f"  torch.rand(1): {torch.rand(1).item():.6f}")
    
    print("\n" + "=" * 70)
    print("✓ 随机种子模块测试完成")
    print("=" * 70)
