"""
MED滤波算法实现
"""
import numpy as np
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')


class MED:
    """
    最小熵解卷积 (Minimum Entropy Deconvolution)
    """

    def __init__(self, filter_length=50, max_iter=100, objective='l1_norm'):
        """
        Args:
            filter_length: 滤波器长度
            max_iter: 最大迭代次数
            objective: 目标函数类型 ('entropy', 'l1_norm', 'd_norm')
        """
        self.filter_length = filter_length
        self.max_iter = max_iter
        self.objective_type = objective

    def compute_entropy(self, signal):
        """计算信号的熵"""
        signal_centered = signal - np.mean(signal)
        l2_norm = np.linalg.norm(signal_centered, 2)

        if l2_norm < 1e-10:
            return 0.0

        signal_norm = signal_centered / l2_norm
        l1_norm = np.linalg.norm(signal_norm, 1)
        N = len(signal)

        entropy = -np.log(l1_norm / np.sqrt(N) + 1e-10)
        return entropy

    def compute_l1_norm_objective(self, signal):
        """L1范数目标函数（最大化L4/L2^2比值）"""
        signal_centered = signal - np.mean(signal)
        l2_norm = np.linalg.norm(signal_centered, 2)

        if l2_norm < 1e-10:
            return 0.0

        l4_norm = np.sum(signal_centered ** 4)
        l2_squared = l2_norm ** 2

        return -(l4_norm / (l2_squared + 1e-10))

    def compute_d_norm_objective(self, signal):
        """D范数目标函数（最大化峰度）"""
        signal_centered = signal - np.mean(signal)
        std = np.std(signal_centered)

        if std < 1e-10:
            return 0.0

        kurtosis = np.mean((signal_centered / std) ** 4)
        return -kurtosis

    def objective_function(self, filter_coeffs, signal):
        """MED目标函数"""
        output = np.convolve(signal, filter_coeffs, mode='same')

        if self.objective_type == 'entropy':
            return self.compute_entropy(output)
        elif self.objective_type == 'l1_norm':
            return self.compute_l1_norm_objective(output)
        elif self.objective_type == 'd_norm':
            return self.compute_d_norm_objective(output)
        else:
            raise ValueError(f"Unknown objective type: {self.objective_type}")

    def filter(self, signal, verbose=False):
        """
        对信号进行MED滤波

        Args:
            signal: 输入信号
            verbose: 是否打印信息

        Returns:
            filtered_signal: 滤波后的信号
        """
        initial_filter = np.zeros(self.filter_length)
        initial_filter[self.filter_length // 2] = 1.0

        if verbose:
            init_obj = self.objective_function(initial_filter, signal)
            print(f"  MED初始目标值: {init_obj:.6f}")

        result = minimize(
            fun=self.objective_function,
            x0=initial_filter,
            args=(signal,),
            method='L-BFGS-B',
            options={'maxiter': self.max_iter, 'disp': False}
        )

        optimal_filter = result.x / (np.linalg.norm(result.x) + 1e-10)
        filtered_signal = np.convolve(signal, optimal_filter, mode='same')

        if verbose:
            print(f"  MED最终目标值: {result.fun:.6f}")
            improvement = (init_obj - result.fun) / init_obj * 100
            print(f"  熵降低: {improvement:.1f}%")

        return filtered_signal

# =============================================================================
# 测试代码
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("测试 MED")
    print("=" * 70)

    # 生成测试信号
    fs = 20480
    t = np.linspace(0, 1, fs)

    # 冲击信号
    impulse_signal = np.zeros_like(t)
    for i in range(10):
        t_impulse = i * 0.1
        idx = int(t_impulse * fs)
        if idx < len(t):
            decay = np.exp(-200 * np.abs(t - t_impulse))
            impulse_signal += 5 * decay * np.cos(2 * np.pi * 3000 * t)

    noise = 3 * np.random.randn(len(t))
    noisy_signal = impulse_signal + noise

    print(f"\n测试信号: {len(noisy_signal)} 采样点")

    # MED滤波
    print("\n" + "-" * 70)
    print("MED滤波")
    print("-" * 70)
    med = MED(filter_length=50, max_iter=100)
    med_filtered = med.filter(noisy_signal, verbose=True)

    print("\n" + "=" * 70)
    print("✅ 测试完成!")
    print("=" * 70)
