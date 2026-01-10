"""
Magnetron Waveform Generation

Simple rectangular pulse generation for magnetron radar.
No chirp, no pulse compression - range resolution = c * tau / 2
"""
import numpy as np
from dataclasses import dataclass
from typing import Tuple

C = 299792458.0  # Speed of light


@dataclass
class MagnetronPulseConfig:
    """Configuration for magnetron pulse."""
    pulse_width_s: float = 0.5e-6     # 500 ns pulse
    sample_rate_hz: float = 100e6     # 100 MHz sampling
    carrier_frequency_hz: float = 9.5e9

    @property
    def n_samples(self) -> int:
        """Number of samples in pulse."""
        return max(1, int(self.pulse_width_s * self.sample_rate_hz))

    @property
    def range_resolution_m(self) -> float:
        """Range resolution = c * tau / 2."""
        return C * self.pulse_width_s / 2

    @property
    def range_per_sample_m(self) -> float:
        """Range represented by each sample."""
        return C / (2 * self.sample_rate_hz)


def generate_rectangular_pulse(
    config: MagnetronPulseConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate simple rectangular pulse.

    Magnetron produces constant-amplitude pulse with no frequency modulation.

    Args:
        config: Pulse configuration

    Returns:
        Tuple of (time_axis, pulse_signal)
        - time_axis: Time values in seconds
        - pulse_signal: Complex baseband pulse
    """
    n = config.n_samples
    t = np.arange(n) / config.sample_rate_hz

    # Simple rectangular envelope (constant amplitude)
    pulse = np.ones(n, dtype=np.complex128)

    return t, pulse


def generate_gaussian_pulse(
    config: MagnetronPulseConfig,
    taper_fraction: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate pulse with Gaussian tapering on edges.

    More realistic than perfect rectangular pulse.

    Args:
        config: Pulse configuration
        taper_fraction: Fraction of pulse width for rise/fall

    Returns:
        Tuple of (time_axis, pulse_signal)
    """
    n = config.n_samples
    t = np.arange(n) / config.sample_rate_hz

    # Gaussian tapered edges
    taper_samples = max(1, int(n * taper_fraction))

    envelope = np.ones(n)

    # Rising edge
    if taper_samples > 0:
        rise = np.linspace(0, 1, taper_samples)
        rise = 1 - np.cos(rise * np.pi / 2)  # Cosine taper
        envelope[:taper_samples] = rise

        # Falling edge
        envelope[-taper_samples:] = rise[::-1]

    pulse = envelope.astype(np.complex128)

    return t, pulse


def envelope_detect(rx_signal: np.ndarray) -> np.ndarray:
    """Simple envelope detection (magnitude).

    For magnetron radar, no matched filter needed - just take envelope.

    Args:
        rx_signal: Complex received signal

    Returns:
        Envelope (magnitude) of signal
    """
    return np.abs(rx_signal)


def range_bin_signal(
    envelope: np.ndarray,
    config: MagnetronPulseConfig,
    max_range_m: float,
    n_bins: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Bin envelope signal into range cells.

    Args:
        envelope: Detected envelope signal
        config: Pulse configuration
        max_range_m: Maximum range to bin
        n_bins: Number of range bins (default: max_range / resolution)

    Returns:
        Tuple of (range_bins_m, binned_power)
    """
    if n_bins is None:
        n_bins = int(max_range_m / config.range_resolution_m) + 1

    range_bins = np.linspace(0, max_range_m, n_bins)
    binned_power = np.zeros(n_bins)

    # Map samples to range bins
    for i, val in enumerate(envelope):
        range_m = i * config.range_per_sample_m
        bin_idx = int(range_m / max_range_m * (n_bins - 1))
        if 0 <= bin_idx < n_bins:
            binned_power[bin_idx] = max(binned_power[bin_idx], val)

    return range_bins, binned_power


def synthesize_received_signal(
    target_ranges_m: np.ndarray,
    target_amplitudes: np.ndarray,
    config: MagnetronPulseConfig,
    max_range_m: float,
    noise_power: float = 0.0
) -> np.ndarray:
    """Synthesize received signal from multiple targets.

    Args:
        target_ranges_m: Array of target ranges
        target_amplitudes: Array of target amplitudes (sqrt of power)
        config: Pulse configuration
        max_range_m: Maximum range (determines signal length)
        noise_power: Noise power to add

    Returns:
        Complex received signal
    """
    # Signal length based on max range
    max_delay_s = 2 * max_range_m / C
    n_samples = int(max_delay_s * config.sample_rate_hz) + config.n_samples + 100

    # Generate reference pulse
    _, tx_pulse = generate_rectangular_pulse(config)

    # Initialize received signal
    rx_signal = np.zeros(n_samples, dtype=np.complex128)

    # Add returns from each target
    for r, amp in zip(target_ranges_m, target_amplitudes):
        delay_s = 2 * r / C
        delay_samples = int(delay_s * config.sample_rate_hz)

        if delay_samples + len(tx_pulse) < n_samples:
            rx_signal[delay_samples:delay_samples + len(tx_pulse)] += amp * tx_pulse

    # Add noise
    if noise_power > 0:
        noise = np.sqrt(noise_power / 2) * (
            np.random.randn(n_samples) + 1j * np.random.randn(n_samples)
        )
        rx_signal += noise

    return rx_signal


def process_magnetron_return(
    rx_signal: np.ndarray,
    config: MagnetronPulseConfig,
    max_range_m: float,
    n_range_bins: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Process received signal to range profile.

    Simple processing for magnetron:
    1. Envelope detection
    2. Range binning

    Args:
        rx_signal: Complex received signal
        config: Pulse configuration
        max_range_m: Maximum range
        n_range_bins: Number of output range bins

    Returns:
        Tuple of (range_bins_m, power_profile)
    """
    # Envelope detection
    envelope = envelope_detect(rx_signal)

    # Range binning
    ranges, power = range_bin_signal(
        envelope, config, max_range_m, n_range_bins
    )

    return ranges, power


def compute_snr(
    target_rcs_m2: float,
    range_m: float,
    peak_power_w: float,
    antenna_gain: float,
    wavelength_m: float,
    pulse_width_s: float,
    noise_figure_db: float = 5.0,
    system_losses_db: float = 3.0
) -> float:
    """Compute single-pulse SNR using radar equation.

    For magnetron (no pulse compression), processing gain = 1.

    Args:
        target_rcs_m2: Target RCS in m²
        range_m: Target range in meters
        peak_power_w: Peak transmit power in watts
        antenna_gain: Antenna gain (linear)
        wavelength_m: Wavelength in meters
        pulse_width_s: Pulse width in seconds
        noise_figure_db: Receiver noise figure
        system_losses_db: System losses

    Returns:
        SNR in dB
    """
    # Constants
    k = 1.38e-23  # Boltzmann
    T0 = 290.0    # Reference temperature

    # Noise power
    bandwidth = 1 / pulse_width_s  # Approximate receiver bandwidth
    noise_figure = 10 ** (noise_figure_db / 10)
    losses = 10 ** (system_losses_db / 10)
    noise_power = k * T0 * bandwidth * noise_figure

    # Received power (radar equation)
    numerator = peak_power_w * antenna_gain**2 * wavelength_m**2 * target_rcs_m2
    denominator = (4 * np.pi)**3 * range_m**4 * losses

    received_power = numerator / denominator

    # SNR
    snr_linear = received_power / noise_power
    snr_db = 10 * np.log10(snr_linear + 1e-30)

    return snr_db
