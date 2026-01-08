"""Matched filter and pulse compression."""
import numpy as np
from scipy import signal as sp_signal
from typing import Tuple

from .waveform import WaveformConfig


def matched_filter(
    rx_signal: np.ndarray,
    tx_waveform: np.ndarray
) -> np.ndarray:
    """
    Apply matched filter via correlation.
    
    Args:
        rx_signal: Received signal
        tx_waveform: Transmitted waveform (reference)
        
    Returns:
        Compressed signal
    """
    compressed = np.correlate(rx_signal, tx_waveform, mode='same')
    return compressed


def matched_filter_fft(
    rx_signal: np.ndarray,
    tx_waveform: np.ndarray
) -> np.ndarray:
    """
    FFT-based matched filter (faster for long signals).
    
    Uses: MF output = IFFT(FFT(rx) × conj(FFT(tx)))
    
    Args:
        rx_signal: Received signal
        tx_waveform: Transmitted waveform
        
    Returns:
        Compressed signal
    """
    n = len(rx_signal)
    n_fft = 2 ** int(np.ceil(np.log2(n + len(tx_waveform) - 1)))
    
    RX = np.fft.fft(rx_signal, n_fft)
    TX = np.fft.fft(tx_waveform, n_fft)
    
    compressed = np.fft.ifft(RX * np.conj(TX))
    
    # Center the output
    return compressed[:n]


def matched_filter_fft_windowed(
    rx_signal: np.ndarray,
    tx_waveform: np.ndarray,
    window_type: str = "taylor"
) -> np.ndarray:
    """
    FFT-based matched filter with windowing for sidelobe reduction.
    
    Windowing reduces range sidelobes from ~-13dB (sinc) to:
    - Taylor: ~-35dB to -40dB
    - Hamming: ~-42dB
    - Blackman: ~-58dB
    
    Args:
        rx_signal: Received signal
        tx_waveform: Transmitted waveform
        window_type: 'taylor', 'hamming', 'blackman', or 'none'
        
    Returns:
        Compressed signal with reduced sidelobes
    """
    n = len(rx_signal)
    n_tx = len(tx_waveform)
    n_fft = 2 ** int(np.ceil(np.log2(n + n_tx - 1)))
    
    # Apply window to transmit waveform for matched filter
    if window_type == "taylor":
        # Taylor window with -35dB sidelobes
        nbar = 4
        sll = 35  # sidelobe level in dB
        # Approximate Taylor with cosine-sum
        a = np.zeros(nbar)
        a[0] = 1.0
        for m in range(1, nbar):
            a[m] = (-1)**(m+1) * np.prod([(1 - (m/sll)**2) / (1 - (m/k)**2) 
                                          for k in range(1, nbar) if k != m])
        t = np.linspace(-0.5, 0.5, n_tx)
        window = sum(a[m] * np.cos(2*np.pi*m*t) for m in range(nbar))
        window = window / np.max(window)
    elif window_type == "hamming":
        window = np.hamming(n_tx)
    elif window_type == "blackman":
        window = np.blackman(n_tx)
    else:
        window = np.ones(n_tx)
    
    # Windowed reference for matched filter
    tx_windowed = tx_waveform * window
    
    RX = np.fft.fft(rx_signal, n_fft)
    TX = np.fft.fft(tx_windowed, n_fft)
    
    compressed = np.fft.ifft(RX * np.conj(TX))
    
    return compressed[:n]


def compute_compression_gain(
    input_signal: np.ndarray,
    compressed: np.ndarray
) -> float:
    """
    Compute pulse compression gain in dB.
    
    Gain = 20 log10(peak_compressed / peak_input)
    
    Args:
        input_signal: Original signal
        compressed: Compressed signal
        
    Returns:
        Compression gain in dB
    """
    peak_in = np.max(np.abs(input_signal))
    peak_out = np.max(np.abs(compressed))
    
    if peak_in > 0:
        return 20 * np.log10(peak_out / peak_in)
    return 0.0


def measure_resolution(
    compressed: np.ndarray,
    sample_rate_hz: float
) -> Tuple[float, float]:
    """
    Measure range resolution from compressed pulse.
    
    Args:
        compressed: Compressed signal
        sample_rate_hz: Sample rate
        
    Returns:
        resolution_m: 3 dB width in meters
        sidelobe_db: First sidelobe level in dB
    """
    c = 299792458.0
    
    # Find peak
    peak_idx = np.argmax(np.abs(compressed))
    peak_val = np.abs(compressed[peak_idx])
    half_power = peak_val / np.sqrt(2)
    
    # Find 3 dB points
    above_half = np.abs(compressed) >= half_power
    indices = np.where(above_half)[0]
    
    if len(indices) > 1:
        width_samples = indices[-1] - indices[0]
    else:
        width_samples = 1
    
    width_seconds = width_samples / sample_rate_hz
    resolution_m = c * width_seconds / 2  # Two-way
    
    # First sidelobe
    main_lobe_width = max(width_samples * 2, 10)
    left_idx = max(0, peak_idx - main_lobe_width)
    right_idx = min(len(compressed), peak_idx + main_lobe_width)
    
    left_region = np.abs(compressed[:left_idx]) if left_idx > 0 else np.array([0])
    right_region = np.abs(compressed[right_idx:]) if right_idx < len(compressed) else np.array([0])
    
    sidelobe_val = max(
        np.max(left_region) if len(left_region) > 0 else 0,
        np.max(right_region) if len(right_region) > 0 else 0
    )
    
    if peak_val > 0 and sidelobe_val > 0:
        sidelobe_db = 20 * np.log10(sidelobe_val / peak_val)
    else:
        sidelobe_db = -np.inf
    
    return resolution_m, sidelobe_db


def pulse_compress(
    rx_signal: np.ndarray,
    config: WaveformConfig,
    window_type: str = "hamming"
) -> np.ndarray:
    """
    Complete pulse compression with windowing.
    
    Args:
        rx_signal: Received signal
        config: Waveform configuration
        window_type: Window to apply
        
    Returns:
        Compressed range profile
    """
    from .waveform import generate_lfm_chirp, apply_window
    
    # Generate reference waveform
    _, tx = generate_lfm_chirp(config)
    
    # Apply window for sidelobe control
    tx_windowed = apply_window(tx, window_type)
    
    # Matched filter
    compressed = matched_filter_fft(rx_signal, tx_windowed)
    
    return compressed
