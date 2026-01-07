"""End-to-end radar simulation pipeline."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import RadarConfig
from interfaces import AntennaPattern, RayBundle, ScatterResult, RangeProfile
from signal.waveform import WaveformConfig, generate_lfm_chirp
from signal.matched_filter import matched_filter_fft, measure_resolution
from signal.detection import CFARConfig, cfar_1d, find_peaks, peaks_to_ranges


@dataclass
class SimulationResult:
    """Complete simulation results."""
    config: RadarConfig
    range_profile: RangeProfile
    detected_targets: List[dict]
    snr_db: float
    resolution_m: float


def synthesize_received_signal(
    scatter: ScatterResult,
    wf_config: WaveformConfig,
    max_range_m: float
) -> np.ndarray:
    """
    Synthesize received signal from scatter results.
    
    Each scatterer contributes delayed, attenuated copy of TX waveform.
    
    Args:
        scatter: ScatterResult from WP-3
        wf_config: Waveform configuration
        max_range_m: Maximum range
        
    Returns:
        Complex received signal
    """
    c = 299792458.0
    
    # Total signal duration
    max_delay = 2 * max_range_m / c
    n_samples = int(max_delay * wf_config.sample_rate_hz) + wf_config.n_samples
    
    rx_signal = np.zeros(n_samples, dtype=complex)
    
    _, tx_waveform = generate_lfm_chirp(wf_config)
    
    for i in range(scatter.n_scatterers):
        # Time delay (two-way)
        delay_s = 2 * scatter.path_to_rx_m[i] / c
        delay_samples = int(delay_s * wf_config.sample_rate_hz)
        
        if delay_samples + len(tx_waveform) > n_samples:
            continue
        
        # Add delayed, attenuated, phase-shifted echo
        amplitude = np.sqrt(scatter.scattered_power_w[i])
        phase = scatter.phase_rad[i]
        
        echo = amplitude * np.exp(1j * phase) * tx_waveform
        rx_signal[delay_samples:delay_samples + len(echo)] += echo
    
    return rx_signal


def add_noise(signal: np.ndarray, snr_db: float = 20.0) -> np.ndarray:
    """
    Add AWGN to signal.
    
    Args:
        signal: Input signal
        snr_db: Desired SNR in dB
        
    Returns:
        Noisy signal
    """
    signal_power = np.mean(np.abs(signal) ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    
    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))
    )
    
    return signal + noise


def run_pipeline(
    config: RadarConfig,
    scatter_result: ScatterResult,
    add_noise_flag: bool = True,
    noise_snr_db: float = 30.0
) -> SimulationResult:
    """
    Execute complete radar simulation pipeline.
    
    Args:
        config: Radar configuration
        scatter_result: Scattering results from WP-3
        add_noise_flag: Whether to add noise
        noise_snr_db: SNR of added noise
        
    Returns:
        SimulationResult with range profile and detections
    """
    # Generate waveform
    wf_config = WaveformConfig(
        bandwidth_hz=config.bandwidth_hz,
        center_frequency_hz=config.center_frequency_hz
    )
    _, tx_waveform = generate_lfm_chirp(wf_config)
    
    # Synthesize received signal
    rx_signal = synthesize_received_signal(
        scatter_result, wf_config, config.max_range_m
    )
    
    # Add noise if requested
    if add_noise_flag and np.max(np.abs(rx_signal)) > 0:
        rx_signal = add_noise(rx_signal, noise_snr_db)
    elif add_noise_flag:
        # Pure noise if no signal
        noise_power = 1e-12
        rx_signal = np.sqrt(noise_power / 2) * (
            np.random.randn(len(rx_signal)) + 1j * np.random.randn(len(rx_signal))
        )
    
    # Apply matched filter
    compressed = matched_filter_fft(rx_signal, tx_waveform)
    
    # Measure performance
    resolution_m, sidelobe_db = measure_resolution(
        compressed, wf_config.sample_rate_hz
    )
    
    # Detection
    cfar_config = CFARConfig()
    detections, threshold = cfar_1d(compressed, cfar_config)
    peaks = find_peaks(np.abs(compressed), detections)
    detected_ranges = peaks_to_ranges(peaks, wf_config.sample_rate_hz)
    
    # Compute SNR
    if len(peaks) > 0:
        peak_power = np.max(np.abs(compressed[peaks]) ** 2)
        noise_samples = np.abs(compressed[:100]) ** 2
        noise_power = np.mean(noise_samples) if len(noise_samples) > 0 else 1e-20
        snr_db = 10 * np.log10(peak_power / noise_power) if noise_power > 0 else 0
    else:
        snr_db = 0.0
    
    # Create range profile
    n_bins = len(compressed)
    range_bins = np.linspace(0, config.max_range_m, n_bins)
    amplitude_db = 20 * np.log10(np.abs(compressed) + 1e-12)
    
    range_profile = RangeProfile(
        range_bins_m=range_bins,
        amplitude_db=amplitude_db,
        snr_db=snr_db,
        detected_ranges_m=detected_ranges
    )
    
    return SimulationResult(
        config=config,
        range_profile=range_profile,
        detected_targets=[
            {'range_m': float(r), 'snr_db': snr_db} for r in detected_ranges
        ],
        snr_db=snr_db,
        resolution_m=resolution_m
    )


def run_simple_simulation(
    target_range_m: float = 2000.0,
    target_rcs_m2: float = 0.082,
    tx_power_w: float = 25.0
) -> SimulationResult:
    """
    Run simple single-target simulation.
    
    Args:
        target_range_m: Target range in meters
        target_rcs_m2: Target RCS in m²
        tx_power_w: Transmit power in watts
        
    Returns:
        SimulationResult
    """
    config = RadarConfig(tx_power_w=tx_power_w)
    
    # Compute received power using radar equation
    wavelength = config.wavelength_m
    G = 10 ** (config.antenna_gain_dbi / 10)  # Linear gain
    
    # Pr = (Pt × G² × λ² × σ) / ((4π)³ × R⁴)
    Pr = (tx_power_w * G**2 * wavelength**2 * target_rcs_m2) / \
         ((4 * np.pi)**3 * target_range_m**4)
    
    # Phase from two-way path
    phase = 2 * config.wavenumber * target_range_m
    
    scatter = ScatterResult(
        n_scatterers=1,
        scatter_points=np.array([[target_range_m, 0, 0]]),
        rcs_m2=np.array([target_rcs_m2]),
        scattered_power_w=np.array([Pr]),
        path_to_rx_m=np.array([target_range_m]),
        phase_rad=np.array([phase])
    )
    
    return run_pipeline(config, scatter)
