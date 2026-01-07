"""Target detection algorithms."""
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class CFARConfig:
    """Configuration for CFAR detector.
    
    Attributes:
        guard_cells: Guard cells on each side of CUT
        training_cells: Training cells on each side
        pfa: Probability of false alarm
    """
    guard_cells: int = 4
    training_cells: int = 16
    pfa: float = 1e-6
    
    @property
    def threshold_factor(self) -> float:
        """CFAR threshold multiplier for given Pfa.
        
        For CA-CFAR with N training cells:
        α = N × (Pfa^(-1/N) - 1)
        """
        N = 2 * self.training_cells
        return N * (self.pfa ** (-1/N) - 1)


def cfar_1d(
    signal: np.ndarray,
    config: CFARConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cell-Averaging CFAR detector.
    
    Args:
        signal: Input signal (complex or magnitude)
        config: CFAR configuration
        
    Returns:
        detections: Boolean array of detections
        threshold: Adaptive threshold array
    """
    n = len(signal)
    power = np.abs(signal) ** 2
    
    detections = np.zeros(n, dtype=bool)
    threshold = np.zeros(n)
    
    total_cells = config.guard_cells + config.training_cells
    
    for i in range(total_cells, n - total_cells):
        # Training cells (left and right, excluding guard cells)
        left_start = i - total_cells
        left_end = i - config.guard_cells
        right_start = i + config.guard_cells + 1
        right_end = i + total_cells + 1
        
        training_left = power[left_start:left_end]
        training_right = power[right_start:right_end]
        
        # Average noise estimate (Cell-Averaging)
        noise_estimate = np.mean(np.concatenate([training_left, training_right]))
        
        # Adaptive threshold
        threshold[i] = config.threshold_factor * noise_estimate
        
        # Detection decision
        detections[i] = power[i] > threshold[i]
    
    return detections, threshold


def cfar_os(
    signal: np.ndarray,
    config: CFARConfig,
    k: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ordered-Statistic CFAR detector.
    
    More robust to clutter edges than CA-CFAR.
    
    Args:
        signal: Input signal
        config: CFAR configuration
        k: Order statistic index (default: 3N/4)
        
    Returns:
        detections, threshold
    """
    n = len(signal)
    power = np.abs(signal) ** 2
    
    detections = np.zeros(n, dtype=bool)
    threshold = np.zeros(n)
    
    total_cells = config.guard_cells + config.training_cells
    N = 2 * config.training_cells
    
    if k is None:
        k = int(0.75 * N)  # 3N/4 typical choice
    
    for i in range(total_cells, n - total_cells):
        left_start = i - total_cells
        left_end = i - config.guard_cells
        right_start = i + config.guard_cells + 1
        right_end = i + total_cells + 1
        
        training = np.concatenate([
            power[left_start:left_end],
            power[right_start:right_end]
        ])
        
        # Order statistic: k-th smallest value
        sorted_training = np.sort(training)
        noise_estimate = sorted_training[min(k, len(sorted_training) - 1)]
        
        threshold[i] = config.threshold_factor * noise_estimate
        detections[i] = power[i] > threshold[i]
    
    return detections, threshold


def find_peaks(
    signal: np.ndarray,
    detections: np.ndarray,
    min_separation: int = 5
) -> List[int]:
    """
    Find peak locations among detections.
    
    Args:
        signal: Input signal (magnitude)
        detections: Boolean detection mask
        min_separation: Minimum samples between peaks
        
    Returns:
        List of peak indices
    """
    peaks = []
    magnitude = np.abs(signal)
    
    detection_indices = np.where(detections)[0]
    
    for idx in detection_indices:
        start = max(0, idx - min_separation)
        end = min(len(signal), idx + min_separation + 1)
        
        # Check if this is a local maximum
        if magnitude[idx] == np.max(magnitude[start:end]):
            # Check distance from previous peak
            if not peaks or idx - peaks[-1] >= min_separation:
                peaks.append(idx)
    
    return peaks


def peaks_to_ranges(
    peak_indices: List[int],
    sample_rate_hz: float,
    start_range_m: float = 0.0
) -> np.ndarray:
    """
    Convert peak indices to range values.
    
    Args:
        peak_indices: List of sample indices
        sample_rate_hz: Sample rate
        start_range_m: Range offset
        
    Returns:
        Array of ranges in meters
    """
    c = 299792458.0
    
    # Range = c × t / 2 (two-way)
    times = np.array(peak_indices) / sample_rate_hz
    ranges = start_range_m + c * times / 2
    
    return ranges


def compute_snr(
    signal: np.ndarray,
    peak_idx: int,
    noise_region: Tuple[int, int]
) -> float:
    """
    Compute SNR at detected peak.
    
    Args:
        signal: Input signal
        peak_idx: Peak index
        noise_region: (start, end) indices for noise estimation
        
    Returns:
        SNR in dB
    """
    peak_power = np.abs(signal[peak_idx]) ** 2
    noise_samples = signal[noise_region[0]:noise_region[1]]
    noise_power = np.mean(np.abs(noise_samples) ** 2)
    
    if noise_power > 0:
        return 10 * np.log10(peak_power / noise_power)
    return np.inf
