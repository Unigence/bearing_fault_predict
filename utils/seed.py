"""
Random Seed Utilities
统一设置随机种子以保证实验可复现性
"""

import random
import numpy as np
import torch
import os
from typing import Optional


def set_seed(seed: int = 42, deterministic: bool = True, benchmark: bool = False):
    """
    设置所有相关库的随机种子
    
    Args:
        seed: 随机种子
        deterministic: 是否使用确定性算法（牺牲性能换取可复现性）
        benchmark: 是否启用cudnn benchmark（提升性能但可能影响可复现性）
    
    Note:
        - deterministic=True时，cuDNN会选择确定性算法，但可能会降低性能
        - benchmark=True时，cuDNN会自动寻找最优算法，但可能导致结果不可复现
        - 如果需要完全的可复现性，建议设置 deterministic=True, benchmark=False
    """
    # Python内置random
    random.seed(seed)
    
    # Numpy
    np.random.seed(seed)
    
    # PyTorch CPU
    torch.manual_seed(seed)
    
    # PyTorch GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 对所有GPU设置种子
    
    # cuDNN
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = benchmark
    
    # 设置环境变量（某些库可能依赖这个）
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"Random seed set to {seed}")
    print(f"  - Deterministic: {deterministic}")
    print(f"  - cuDNN Benchmark: {benchmark}")


def get_random_state():
    """
    获取当前随机状态
    
    Returns:
        dict: 包含各个库随机状态的字典
    """
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    
    if torch.cuda.is_available():
        state['torch_cuda'] = torch.cuda.get_rng_state_all()
    
    return state


def set_random_state(state: dict):
    """
    恢复随机状态
    
    Args:
        state: 由get_random_state()返回的状态字典
    """
    if 'python' in state:
        random.setstate(state['python'])
    
    if 'numpy' in state:
        np.random.set_state(state['numpy'])
    
    if 'torch' in state:
        torch.set_rng_state(state['torch'])
    
    if 'torch_cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['torch_cuda'])
    
    print("Random state restored")


def worker_init_fn(worker_id: int, seed: Optional[int] = None):
    """
    DataLoader worker初始化函数
    为每个worker设置不同但可复现的随机种子
    
    Args:
        worker_id: worker的ID
        seed: 基础随机种子，如果为None则使用torch初始种子
    
    Usage:
        DataLoader(..., worker_init_fn=lambda x: worker_init_fn(x, seed=42))
    """
    if seed is None:
        seed = torch.initial_seed() % 2**32
    
    worker_seed = seed + worker_id
    
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    
    return worker_seed


class RandomSeedContext:
    """
    随机种子上下文管理器
    在with块内使用指定的随机种子，退出后恢复原状态
    
    Usage:
        with RandomSeedContext(seed=42):
            # 使用固定种子的代码
            data = torch.randn(10, 10)
        # 恢复原来的随机状态
    """
    
    def __init__(self, seed: int):
        """
        Args:
            seed: 随机种子
        """
        self.seed = seed
        self.state = None
    
    def __enter__(self):
        """进入上下文：保存当前状态并设置新种子"""
        self.state = get_random_state()
        set_seed(self.seed, deterministic=False, benchmark=False)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文：恢复原状态"""
        if self.state is not None:
            set_random_state(self.state)
        return False


def verify_reproducibility(func, seed: int = 42, n_runs: int = 2):
    """
    验证函数的可复现性
    
    Args:
        func: 要测试的函数（无参数）
        seed: 随机种子
        n_runs: 运行次数
    
    Returns:
        bool: 是否可复现
        
    Example:
        def test_func():
            return torch.randn(5)
        
        is_reproducible = verify_reproducibility(test_func, seed=42)
    """
    results = []
    
    for i in range(n_runs):
        set_seed(seed, deterministic=True, benchmark=False)
        result = func()
        
        # 转换为numpy数组以便比较
        if isinstance(result, torch.Tensor):
            result = result.cpu().numpy()
        
        results.append(result)
    
    # 检查所有结果是否相同
    is_reproducible = True
    for i in range(1, n_runs):
        if not np.allclose(results[0], results[i], rtol=1e-5, atol=1e-8):
            is_reproducible = False
            break
    
    if is_reproducible:
        print(f"✓ Function is reproducible with seed={seed}")
    else:
        print(f"✗ Function is NOT reproducible with seed={seed}")
    
    return is_reproducible


def get_seed_from_time():
    """
    从当前时间生成随机种子
    
    Returns:
        int: 随机种子
    """
    import time
    seed = int(time.time() * 1000) % (2**32)
    return seed


def print_cuda_info():
    """打印CUDA相关信息，有助于调试可复现性问题"""
    print("\n=== CUDA Information ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.current_device()}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"cuDNN deterministic: {torch.backends.cudnn.deterministic}")
        print(f"cuDNN benchmark: {torch.backends.cudnn.benchmark}")
    
    print("========================\n")


if __name__ == '__main__':
    # 测试随机种子设置
    print("Testing seed setting:")
    set_seed(42, deterministic=True, benchmark=False)
    
    # 生成一些随机数
    print("\nRandom numbers with seed=42:")
    print(f"Python random: {random.random()}")
    print(f"Numpy random: {np.random.rand()}")
    print(f"PyTorch random: {torch.rand(1).item()}")
    
    # 重置种子，应该得到相同的结果
    print("\nResetting seed to 42:")
    set_seed(42, deterministic=True, benchmark=False)
    print(f"Python random: {random.random()}")
    print(f"Numpy random: {np.random.rand()}")
    print(f"PyTorch random: {torch.rand(1).item()}")
    
    # 测试上下文管理器
    print("\n\nTesting RandomSeedContext:")
    print(f"Before context: {torch.rand(1).item()}")
    
    with RandomSeedContext(seed=123):
        print(f"Inside context (seed=123): {torch.rand(1).item()}")
        print(f"Inside context (seed=123): {torch.rand(1).item()}")
    
    print(f"After context: {torch.rand(1).item()}")
    
    # 测试可复现性
    print("\n\nTesting reproducibility:")
    def test_function():
        return torch.randn(3, 3)
    
    verify_reproducibility(test_function, seed=42, n_runs=3)
    
    # 打印CUDA信息
    print_cuda_info()
    
    # 测试从时间生成种子
    time_seed = get_seed_from_time()
    print(f"Seed from time: {time_seed}")
