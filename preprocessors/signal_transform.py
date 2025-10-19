"""
信号变换模块 (Signal Transform)
提供FFT, STFT, CWT等信号变换功能
"""
import torch
import numpy as np
import pywt


class SignalTransform:
    """信号变换基类"""
    
    def __call__(self, signal):
        raise NotImplementedError


class FFTTransform(SignalTransform):
    """
    快速傅里叶变换 (FFT)
    时域 → 频域
    """
    
    def __init__(self, 
                 use_log=True, 
                 normalize=True,
                 return_phase=False):
        """
        Args:
            use_log: 是否使用对数幅度谱
            normalize: 是否进行Z-score标准化
            return_phase: 是否返回相位信息
        """
        self.use_log = use_log
        self.normalize = normalize
        self.return_phase = return_phase
    
    def __call__(self, signal):
        """
        Args:
            signal: (seq_len,) 时域信号，numpy array或torch tensor
        Returns:
            spectrum: (freq_len,) 频谱
        """
        # 转换为torch tensor
        if isinstance(signal, np.ndarray):
            signal = torch.from_numpy(signal).float()
        
        # FFT变换
        fft_result = torch.fft.rfft(signal)  # 单边频谱
        
        # 提取幅度
        magnitude = torch.abs(fft_result)
        
        # 对数变换
        if self.use_log:
            magnitude = torch.log(magnitude + 1e-8)
        
        # 标准化
        if self.normalize:
            mean = magnitude.mean()
            std = magnitude.std() + 1e-8
            magnitude = (magnitude - mean) / std
        
        # 是否返回相位
        if self.return_phase:
            phase = torch.angle(fft_result)
            return magnitude, phase
        
        return magnitude
    
    def inverse(self, magnitude, phase=None, original_length=None):
        """
        逆FFT (可选功能，用于可视化或重建)
        
        Args:
            magnitude: 幅度谱
            phase: 相位谱 (如果为None，使用零相位)
            original_length: 原始信号长度
        Returns:
            signal: 重建的时域信号
        """
        if phase is None:
            phase = torch.zeros_like(magnitude)
        
        # 构建复数频谱
        if self.use_log:
            magnitude = torch.exp(magnitude) - 1e-8
        
        complex_spectrum = magnitude * torch.exp(1j * phase)
        
        # 逆FFT
        signal = torch.fft.irfft(complex_spectrum, n=original_length)
        
        return signal


class STFTTransform(SignalTransform):
    """
    短时傅里叶变换 (STFT)
    时域 → 时频域
    """
    
    def __init__(self,
                 n_fft=128,
                 hop_length=32,
                 window='hann',
                 use_log=True,
                 normalize=True,
                 target_size=(64, 128)):
        """
        Args:
            n_fft: FFT窗口大小
            hop_length: 跳跃长度
            window: 窗函数类型 ('hann', 'hamming', 'blackman')
            use_log: 是否使用对数幅度
            normalize: 是否标准化
            target_size: 目标尺寸 (H, W)，None表示不resize
        """
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window_type = window
        self.use_log = use_log
        self.normalize = normalize
        self.target_size = target_size
    
    def __call__(self, signal):
        """
        Args:
            signal: (seq_len,) 时域信号
        Returns:
            spectrogram: (H, W) 时频图
        """
        # 转换为torch tensor
        if isinstance(signal, np.ndarray):
            signal = torch.from_numpy(signal).float()
        
        # 创建窗函数
        if self.window_type == 'hann':
            window = torch.hann_window(self.n_fft)
        elif self.window_type == 'hamming':
            window = torch.hamming_window(self.n_fft)
        elif self.window_type == 'blackman':
            window = torch.blackman_window(self.n_fft)
        else:
            raise ValueError(f"Unknown window type: {self.window_type}")
        
        # STFT变换
        stft_result = torch.stft(
            signal,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            return_complex=True,
            center=True,
            pad_mode='reflect'
        )
        
        # 提取幅度
        magnitude = torch.abs(stft_result)  # (freq_bins, time_frames)
        
        # 对数变换
        if self.use_log:
            magnitude = torch.log(magnitude + 1e-8)
        
        # 标准化
        if self.normalize:
            mean = magnitude.mean()
            std = magnitude.std() + 1e-8
            magnitude = (magnitude - mean) / std
        
        # Resize到目标尺寸
        if self.target_size is not None:
            magnitude = magnitude.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            magnitude = torch.nn.functional.interpolate(
                magnitude,
                size=self.target_size,
                mode='bilinear',
                align_corners=False
            )
            magnitude = magnitude.squeeze(0).squeeze(0)  # (H, W)
        
        return magnitude


class CWTTransform(SignalTransform):
    """
    连续小波变换 (CWT)
    时域 → 时频域
    """
    
    def __init__(self,
                 wavelet='morl',
                 scales=32,
                 sampling_rate=20480,
                 freq_range=(100, 10000),
                 target_size=(64, 128),
                 normalize=True):
        """
        Args:
            wavelet: 小波类型 ('morl'=Morlet, 'mexh'=Mexican hat, etc.)
            scales: 小波尺度数量
            sampling_rate: 采样率 (Hz)
            freq_range: 频率范围 (min_freq, max_freq) in Hz
            target_size: 目标尺寸 (H, W)
            normalize: 是否标准化
        """
        self.wavelet = wavelet
        self.scales = scales
        self.sampling_rate = sampling_rate
        self.freq_range = freq_range
        self.target_size = target_size
        self.normalize = normalize
        
        # 计算尺度数组
        # 从高频到低频: 小尺度到大尺度
        min_freq, max_freq = freq_range
        max_scale = self.sampling_rate / (2 * min_freq)
        min_scale = self.sampling_rate / (2 * max_freq)
        
        self.scale_array = np.logspace(
            np.log10(min_scale),
            np.log10(max_scale),
            num=scales
        )
    
    def __call__(self, signal):
        """
        Args:
            signal: (seq_len,) 时域信号
        Returns:
            scalogram: (H, W) 小波时频图
        """
        # 转换为numpy (pywt需要)
        if isinstance(signal, torch.Tensor):
            signal = signal.numpy()
        
        # CWT变换
        coefficients, _ = pywt.cwt(
            signal,
            self.scale_array,
            self.wavelet,
            sampling_period=1.0/self.sampling_rate
        )
        
        # 取绝对值
        scalogram = np.abs(coefficients)
        
        # 标准化到[0, 1]
        if self.normalize:
            if scalogram.max() > 0:
                scalogram = scalogram / scalogram.max()
        
        # 转换为torch tensor
        scalogram = torch.from_numpy(scalogram).float()
        
        # Resize到目标尺寸
        if self.target_size is not None:
            scalogram = scalogram.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            scalogram = torch.nn.functional.interpolate(
                scalogram,
                size=self.target_size,
                mode='bilinear',
                align_corners=False
            )
            scalogram = scalogram.squeeze(0).squeeze(0)  # (H, W)
        
        return scalogram


class IdentityTransform(SignalTransform):
    """
    恒等变换 (不做任何变换)
    用于时域分支
    """
    
    def __call__(self, signal):
        """
        Args:
            signal: (seq_len,) 时域信号
        Returns:
            signal: 原样返回
        """
        if isinstance(signal, np.ndarray):
            signal = torch.from_numpy(signal).float()
        return signal


def create_transform(transform_type, **kwargs):
    """
    创建信号变换的工厂函数
    
    Args:
        transform_type: 'identity', 'fft', 'stft', 'cwt'
        **kwargs: 变换参数
    
    Returns:
        SignalTransform实例
    """
    transforms = {
        'identity': IdentityTransform,
        'fft': FFTTransform,
        'stft': STFTTransform,
        'cwt': CWTTransform
    }
    
    if transform_type not in transforms:
        raise ValueError(f"Unknown transform type: {transform_type}")
    
    return transforms[transform_type](**kwargs)


if __name__ == "__main__":
    """测试代码"""
    import matplotlib.pyplot as plt
    
    # 创建测试信号
    t = torch.linspace(0, 1, 512)
    signal = torch.sin(2 * np.pi * 100 * t) + 0.5 * torch.sin(2 * np.pi * 300 * t)
    signal = signal.numpy()
    
    print("测试信号变换模块")
    print("=" * 70)
    
    # 测试FFT
    print("\n1. FFT Transform")
    fft_transform = FFTTransform(use_log=True, normalize=True)
    spectrum = fft_transform(signal)
    print(f"   输入: {signal.shape} → 输出: {spectrum.shape}")
    
    # 测试STFT
    print("\n2. STFT Transform")
    stft_transform = STFTTransform(n_fft=128, hop_length=32, target_size=(64, 128))
    spectrogram = stft_transform(signal)
    print(f"   输入: {signal.shape} → 输出: {spectrogram.shape}")
    
    # 测试CWT
    print("\n3. CWT Transform")
    try:
        cwt_transform = CWTTransform(scales=32, target_size=(64, 128))
        scalogram = cwt_transform(signal)
        print(f"   输入: {signal.shape} → 输出: {scalogram.shape}")
    except Exception as e:
        print(f"   CWT测试失败 (可能需要安装pywt): {e}")
    
    # 测试Identity
    print("\n4. Identity Transform")
    identity_transform = IdentityTransform()
    output = identity_transform(signal)
    print(f"   输入: {signal.shape} → 输出: {output.shape}")
    
    print("\n✓ 信号变换模块测试完成!")
