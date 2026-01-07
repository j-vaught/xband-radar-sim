"""Radar waveform generation."""
from dataclasses import dataclass
import numpy as np
from typing import Tuple


@dataclass
class WaveformConfig:
    """Configuration for radar waveform.
    
    Attributes:
        pulse_width_s: Pulse duration in seconds
        bandwidth_hz: Chirp bandwidth in Hz
        center_frequency_hz: Center frequency in Hz
        sample_rate_hz: ADC sample rate in Hz
    """
    pulse_width_s: float = 10e-6      # 10 μs
    bandwidth_hz: float = 50e6        # 50 MHz
    center_frequency_hz: float = 9.41e9
    sample_rate_hz: float = 100e6     # 100 MSPS
    
    @property
    def time_bandwidth_product(self) -> float:
        """Pulse compression gain (BT product)."""
        return self.bandwidth_hz * self.pulse_width_s
    
    @property
    def n_samples(self) -> int:
        """Number of samples in pulse."""
        return int(self.pulse_width_s * self.sample_rate_hz)
    
    @property
    def compression_gain_db(self) -> float:
        """Theoretical compression gain in dB."""
        return 10 * np.log10(self.time_bandwidth_product)


def generate_lfm_chirp(config: WaveformConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Linear Frequency Modulated (LFM) chirp waveform.
    
    Args:
        config: Waveform configuration
        
    Returns:
        time: Time axis in seconds
        signal: Complex chirp signal (baseband)
    """
    n_samples = config.n_samples
    t = np.arange(n_samples) / config.sample_rate_hz
    
    # Chirp rate K = B/T
    K = config.bandwidth_hz / config.pulse_width_s
    
    # LFM: s(t) = exp(j π K t²)
    phase = np.pi * K * t**2
    signal = np.exp(1j * phase)
    
    return t, signal


def generate_lfm_upchirp(config: WaveformConfig) -> np.ndarray:
    """Generate up-chirp (frequency increasing)."""
    _, signal = generate_lfm_chirp(config)
    return signal


def generate_lfm_downchirp(config: WaveformConfig) -> np.ndarray:
    """Generate down-chirp (frequency decreasing)."""
    _, signal = generate_lfm_chirp(config)
    return np.conj(signal)  # Conjugate reverses frequency sweep


def apply_window(signal: np.ndarray, window_type: str = "hamming") -> np.ndarray:
    """
    Apply window function to reduce sidelobes.
    
    Args:
        signal: Input signal
        window_type: "hamming", "hanning", "blackman", "taylor", or "rectangular"
        
    Returns:
        Windowed signal
    """
    n = len(signal)
    
    if window_type == "hamming":
        window = np.hamming(n)
    elif window_type == "hanning":
        window = np.hanning(n)
    elif window_type == "blackman":
        window = np.blackman(n)
    elif window_type == "taylor":
        # Taylor window with -40 dB sidelobes
        window = _taylor_window(n, nbar=4, sll=-40)
    else:
        window = np.ones(n)
    
    return signal * window


def _taylor_window(n: int, nbar: int = 4, sll: float = -30) -> np.ndarray:
    """Generate Taylor window.
    
    Args:
        n: Window length
        nbar: Number of nearly equal-level sidelobes
        sll: Desired sidelobe level in dB (negative)
    """
    # Simplified Taylor window approximation
    # Full implementation would use FM coefficients
    B = 10 ** (-sll / 20)
    A = np.arccosh(B) / np.pi
    
    # Use Chebyshev approximation
    window = np.ones(n)
    for i in range(n):
        x = 2 * i / (n - 1) - 1
        window[i] = _chebyshev_poly(n - 1, x * np.cosh(A * np.pi / n))
    
    window = np.abs(window)
    window /= np.max(window)
    return window


def _chebyshev_poly(n: int, x: float) -> float:
    """Evaluate Chebyshev polynomial of first kind."""
    if np.abs(x) <= 1:
        return np.cos(n * np.arccos(x))
    else:
        return np.cosh(n * np.arccosh(np.abs(x))) * np.sign(x) ** n


def compute_instantaneous_frequency(
    signal: np.ndarray,
    sample_rate_hz: float
) -> np.ndarray:
    """Compute instantaneous frequency of signal.
    
    Args:
        signal: Complex signal
        sample_rate_hz: Sample rate
        
    Returns:
        Instantaneous frequency in Hz
    """
    phase = np.unwrap(np.angle(signal))
    freq = np.diff(phase) * sample_rate_hz / (2 * np.pi)
    return np.append(freq, freq[-1])
