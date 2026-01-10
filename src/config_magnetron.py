"""
Magnetron Radar Configuration

X-band magnetron radar for coastal/lake scenarios.
Key difference from solid-state: no pulse compression, resolution = c*tau/2
"""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

C = 299792458.0  # Speed of light m/s


@dataclass
class MagnetronConfig:
    """Configuration for X-band magnetron radar.

    Magnetron characteristics:
    - Higher peak power (kW range)
    - No pulse compression (simple rectangular pulse)
    - Range resolution determined by pulse width: delta_r = c * tau / 2
    - Simpler signal processing (envelope detection only)
    """
    name: str = "X-Band Magnetron"

    # Frequency (X-band: 9-10 GHz)
    center_frequency_hz: float = 9.5e9

    # Pulse parameters
    pulse_width_s: float = 0.5e-6  # 0.5 microsecond for ~75m resolution
    prf_hz: float = 3000.0  # Pulse repetition frequency

    # Power
    peak_power_w: float = 10000.0  # 10 kW peak

    # Antenna
    horizontal_beamwidth_deg: float = 1.8
    vertical_beamwidth_deg: float = 22.0
    antenna_gain_dbi: float = 28.0
    rotation_rate_rpm: float = 24.0
    antenna_height_m: float = 10.0  # Height above water

    # Operational
    max_range_m: float = 2000.0

    # Receiver
    noise_figure_db: float = 5.0
    system_losses_db: float = 3.0

    @property
    def wavelength_m(self) -> float:
        """Wavelength in meters."""
        return C / self.center_frequency_hz

    @property
    def range_resolution_m(self) -> float:
        """Range resolution = c * tau / 2 (no pulse compression)."""
        return C * self.pulse_width_s / 2

    @property
    def blind_range_m(self) -> float:
        """Minimum range (blind zone during transmit)."""
        return C * self.pulse_width_s / 2

    @property
    def max_unambiguous_range_m(self) -> float:
        """Maximum unambiguous range from PRF."""
        return C / (2 * self.prf_hz)

    @property
    def antenna_gain_linear(self) -> float:
        """Antenna gain as linear ratio."""
        return 10 ** (self.antenna_gain_dbi / 10)

    @property
    def horizontal_beamwidth_rad(self) -> float:
        """Horizontal beamwidth in radians."""
        return np.radians(self.horizontal_beamwidth_deg)

    @property
    def vertical_beamwidth_rad(self) -> float:
        """Vertical beamwidth in radians."""
        return np.radians(self.vertical_beamwidth_deg)

    def range_to_sample(self, range_m: float, sample_rate_hz: float) -> int:
        """Convert range to sample index."""
        delay_s = 2 * range_m / C
        return int(delay_s * sample_rate_hz)

    def sample_to_range(self, sample_idx: int, sample_rate_hz: float) -> float:
        """Convert sample index to range."""
        delay_s = sample_idx / sample_rate_hz
        return C * delay_s / 2


# Pre-configured radar modes
def short_range_config() -> MagnetronConfig:
    """Short range, high resolution mode (0-500m)."""
    return MagnetronConfig(
        name="Short Range",
        pulse_width_s=0.1e-6,  # 100ns -> 15m resolution
        prf_hz=5000.0,
        max_range_m=500.0,
    )


def medium_range_config() -> MagnetronConfig:
    """Medium range mode (0-2km)."""
    return MagnetronConfig(
        name="Medium Range",
        pulse_width_s=0.5e-6,  # 500ns -> 75m resolution
        prf_hz=3000.0,
        max_range_m=2000.0,
    )


def long_range_config() -> MagnetronConfig:
    """Long range mode (0-5km)."""
    return MagnetronConfig(
        name="Long Range",
        pulse_width_s=1.0e-6,  # 1us -> 150m resolution
        prf_hz=1500.0,
        max_range_m=5000.0,
    )
