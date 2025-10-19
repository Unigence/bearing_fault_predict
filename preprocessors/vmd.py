"""
VMD算法实现 + IMF选择
"""
import numpy as np
from scipy.optimize import minimize
from scipy.signal import hilbert
from scipy.stats import pearsonr
from vmdpy import VMD as VMDPy
import warnings

class VMD:
    """
    变分模态分解 (Variational Mode Decomposition)
    """

    def __init__(self, K=5, alpha=2000, tau=0, DC=0, init=1, tol=1e-6):
        """
        Args:
            K: 模态数量
            alpha: 二次惩罚因子
            tau: 噪声容忍度
            DC: 是否包含直流分量
            init: 中心频率初始化方式
            tol: 收敛容差
        """
        self.K = K
        self.alpha = alpha
        self.tau = tau
        self.DC = DC
        self.init = init
        self.tol = tol

    def decompose(self, signal, verbose=False):
        """
        VMD分解
        Args:
            signal: 输入信号
            verbose: 是否打印信息
        Returns:
            imfs: 分解得到的IMF (K, T)
            center_freqs: 各IMF的中心频率（归一化）
        """
        if verbose:
            print(f"  VMD分解: K={self.K}, α={self.alpha}")

        try:
            u, u_hat, omega = VMDPy(
                signal,
                alpha=self.alpha,
                tau=self.tau,
                K=self.K,
                DC=self.DC,
                init=self.init,
                tol=self.tol
            )

            imfs = u
            center_freqs = omega[-1, :] if omega.ndim == 2 else omega

            if verbose:
                print(f"  VMD完成: IMF形状={imfs.shape}")

            if imfs.ndim == 1:
                imfs = imfs.reshape(1, -1)

            return imfs, center_freqs

        except Exception as e:
            print(f"  ❌ VMD分解失败: {e}")
            raise


class FeatureSelector:
    """IMF特征选择器 - 多指标加权选择"""

    @staticmethod
    def compute_envelope_kurtosis(signal):
        """
        计算包络峭度

        Args:
            signal: 输入信号

        Returns:
            kurtosis: 包络峭度值（越高表示周期冲击越明显）
        """
        analytic_signal = hilbert(signal)
        envelope = np.abs(analytic_signal)

        mean_val = np.mean(envelope)
        std_val = np.std(envelope)

        if std_val < 1e-10:
            return 0.0

        kurtosis = np.mean(((envelope - mean_val) / std_val) ** 4)
        return kurtosis

    @staticmethod
    def compute_energy_ratio(imf, all_imfs):
        """
        计算能量占比
        Args:
            imf: 单个IMF
            all_imfs: 所有IMF (K, T)
        Returns:
            energy_ratio: 该IMF的能量占比（0-1）
        """
        imf_energy = np.sum(imf ** 2)
        total_energy = np.sum(all_imfs ** 2)

        if total_energy < 1e-10:
            return 0.0

        return imf_energy / total_energy

    @staticmethod
    def compute_orthogonality_score(imf_idx, all_imfs):
        """
        计算与其他IMF的正交性得分（独立性）
        Args:
            imf_idx: IMF索引
            all_imfs: 所有IMF (K, T)
        Returns:
            orthogonality: 正交性得分（0-1，越高表示越独立）
        """
        K = all_imfs.shape[0]
        if K < 2:
            return 1.0

        correlations = []
        for i in range(K):
            if i != imf_idx:
                try:
                    corr, _ = pearsonr(all_imfs[imf_idx, :], all_imfs[i, :])
                    correlations.append(abs(corr))
                except:
                    pass

        if len(correlations) == 0:
            return 1.0

        avg_corr = np.mean(correlations)
        orthogonality = 1.0 - avg_corr

        return max(0.0, orthogonality)

    @staticmethod
    def compute_mode_separation_score(center_freq, all_center_freqs, imf_idx):
        """
        计算模态分离度得分
        Args:
            center_freq: 当前IMF的中心频率
            all_center_freqs: 所有IMF的中心频率
            imf_idx: 当前IMF索引
        Returns:
            separation_score: 分离度得分（越高表示频率分离越好）
        """
        if len(all_center_freqs) < 2:
            return 1.0

        distances = []
        for i, freq in enumerate(all_center_freqs):
            if i != imf_idx:
                distances.append(abs(center_freq - freq))

        if len(distances) == 0:
            return 1.0

        min_distance = min(distances)
        separation_score = min(1.0, min_distance / 0.1)

        return separation_score

    @staticmethod
    def compute_frequency_quality(center_freq, fs=20480):
        """
        计算中心频率质量得分
        Args:
            center_freq: 归一化中心频率（0-0.5）
            fs: 采样率
        Returns:
            quality: 频率质量得分（0-1）
        """
        freq_hz = center_freq * fs

        # 梯形窗口参数
        f1, f2, f3, f4 = 500, 1000, 5000, 7000  # Hz
        #         |--------|========|--------|
        #      低频过渡   核心频段   高频过渡

        if freq_hz < f1:
            # 极低频区域（< 500Hz）：线性上升但保持较低得分
            return max(0, freq_hz / f1) * 0.3
        elif f1 <= freq_hz < f2:
            # 低频过渡区（500-1000Hz）：线性上升到满分
            return 0.3 + 0.7 * (freq_hz - f1) / (f2 - f1)
        elif f2 <= freq_hz <= f3:
            # 核心频段（1000-5000Hz）：满分
            return 1.0
        elif f3 < freq_hz <= f4:
            # 高频过渡区（5000-7000Hz）：线性下降
            return 1.0 - 0.7 * (freq_hz - f3) / (f4 - f3)
        else:
            # 极高频区域（> 7000Hz）：快速下降
            return max(0, 0.3 * (1 - (freq_hz - f4) / 3000))

    @staticmethod
    def compute_quality_matrix(imfs, center_freqs, fs=20480):
        """
        计算所有IMF的综合质量评分矩阵
        Args:
            imfs: 所有IMF (K, T)
            center_freqs: 中心频率数组 (K,)
            fs: 采样率
        Returns:
            quality_matrix: 质量评分矩阵 (K, 5)
                列0: 包络峭度
                列1: 能量占比
                列2: 正交性得分
                列3: 模态分离度
                列4: 频率质量
        """
        K = imfs.shape[0]
        quality_matrix = np.zeros((K, 5))

        for i in range(K):
            quality_matrix[i, 0] = FeatureSelector.compute_envelope_kurtosis(imfs[i, :])
            quality_matrix[i, 1] = FeatureSelector.compute_energy_ratio(imfs[i, :], imfs)
            quality_matrix[i, 2] = FeatureSelector.compute_orthogonality_score(i, imfs)
            quality_matrix[i, 3] = FeatureSelector.compute_mode_separation_score(
                center_freqs[i], center_freqs, i
            )
            quality_matrix[i, 4] = FeatureSelector.compute_frequency_quality(center_freqs[i], fs)

        return quality_matrix

    @staticmethod
    def select_top_imfs(imfs, center_freqs, n_select=5,
                        weights=None, fs=20480,
                        return_details=False):
        """
        多指标加权选择Top-N个最优IMF
        Args:
            imfs: 所有IMF (K, T)
            center_freqs: 中心频率 (K,)
            n_select: 选择数量
            weights: 权重向量 [w_kurtosis, w_energy, w_orthogonality, w_separation, w_frequency]
            fs: 采样率
            return_details: 是否返回详细信息
        Returns:
            selected_indices: 选中的IMF索引 (按综合得分从高到低排序)
            composite_scores: 各IMF的综合得分
            quality_matrix (可选): 质量评分矩阵 (K, 5)
        """
        K = imfs.shape[0]

        # 默认权重
        if weights is None:
            weights = np.array([0.45, 0.15, 0.10, 0.10, 0.20])
        else:
            weights = np.array(weights)
            weights = weights / np.sum(weights)  # 归一化

        # 计算质量评分矩阵
        quality_matrix = FeatureSelector.compute_quality_matrix(imfs, center_freqs, fs)

        # Min-Max归一化每个指标到0-1
        normalized_matrix = np.zeros_like(quality_matrix)
        for col in range(quality_matrix.shape[1]):
            col_min = quality_matrix[:, col].min()
            col_max = quality_matrix[:, col].max()

            if col_max - col_min > 1e-10:
                normalized_matrix[:, col] = (quality_matrix[:, col] - col_min) / (col_max - col_min)
            else:
                normalized_matrix[:, col] = 0.5

        # 计算加权综合得分
        composite_scores = np.dot(normalized_matrix, weights)

        # 选择得分最高的n_select个
        n_select = min(n_select, K)
        selected_indices = np.argsort(composite_scores)[-n_select:][::-1]

        if return_details:
            return selected_indices, composite_scores, quality_matrix
        else:
            return selected_indices, composite_scores


# =============================================================================
# 测试代码
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("测试 VMD + 多指标IMF选择")
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

    # VMD分解
    print("\n" + "-" * 70)
    print("VMD分解")
    print("-" * 70)
    vmd = VMD(K=8, alpha=2000)
    imfs, center_freqs = vmd.decompose(med_filtered, verbose=True)

    # 多指标选择
    print("\n" + "-" * 70)
    print("多指标选择Top-5 IMF")
    print("-" * 70)
    selector = FeatureSelector()
    selected_indices, composite_scores, quality_matrix = \
        selector.select_top_imfs(imfs, center_freqs, n_select=5, return_details=True)

    print(f"\n✓ 选中的IMF: {selected_indices}")
    print(f"\n质量指标详情:")
    print(
        f"{'IMF':<5} {'包络峭度':<10} {'能量占比':<10} {'正交性':<10} {'分离度':<10} {'频率质量':<10} {'综合得分':<10}")
    print("-" * 75)

    for i in range(len(imfs)):
        marker = "✓" if i in selected_indices else " "
        print(f"{marker} IMF{i:<2} {quality_matrix[i, 0]:<10.3f} {quality_matrix[i, 1]:<10.3f} "
              f"{quality_matrix[i, 2]:<10.3f} {quality_matrix[i, 3]:<10.3f} "
              f"{quality_matrix[i, 4]:<10.3f} {composite_scores[i]:<10.3f}")

    print("\n" + "=" * 70)
    print("✅ 测试完成!")
    print("=" * 70)
