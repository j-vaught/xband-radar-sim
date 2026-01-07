"""Radar simulation configuration management."""
from dataclasses import dataclass
from typing import Literal, Union, List
from pathlib import Path
import yaml
import math


@dataclass
class RadarConfig:
    """Configuration for Furuno DRS4D-NXT radar simulation.
    
    Attributes:
        name: Radar system identifier
        center_frequency_hz: Operating frequency (9.41 GHz for X-band)
        bandwidth_hz: Pulse compression bandwidth
        horizontal_beamwidth_deg: Azimuth 3dB beamwidth
        vertical_beamwidth_deg: Elevation 3dB beamwidth
        antenna_gain_dbi: Peak antenna gain
        polarization: Antenna polarization (H or V)
        tx_power_w: Transmitter power
        max_range_m: Maximum instrumented range
        range_resolution_m: Range resolution
        n_rays: Number of rays for propagation simulation
        enable_gpu: Whether to use GPU acceleration
    """
    
    # Identification
    name: str = "Furuno DRS4D-NXT"
    
    # Frequency (Hz)
    center_frequency_hz: float = 9.41e9
    bandwidth_hz: float = 50e6
    
    # Antenna
    antenna_type: Literal["slotted_waveguide"] = "slotted_waveguide"
    horizontal_beamwidth_deg: float = 3.9
    vertical_beamwidth_deg: float = 25.0
    antenna_gain_dbi: float = 22.0
    polarization: Literal["H", "V"] = "H"
    radome_diameter_m: float = 0.61  # 24 inches
    
    # Transmitter
    tx_power_w: float = 25.0
    pulse_compression: bool = True
    
    # Operational
    max_range_m: float = 3000.0
    range_resolution_m: float = 1.5
    
    # Target
    min_target_rcs_m2: float = 0.082  # 3" corner at 9.41 GHz
    
    # Simulation
    n_rays: int = 100_000
    mesh_resolution_lambda: float = 0.1
    enable_gpu: bool = True
    
    @property
    def wavelength_m(self) -> float:
        """Compute wavelength from frequency."""
        return 299792458.0 / self.center_frequency_hz
    
    @property
    def wavenumber(self) -> float:
        """Compute wavenumber k = 2π/λ."""
        return 2 * math.pi / self.wavelength_m
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "RadarConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        from dataclasses import asdict
        with open(path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False)
    
    def validate(self) -> List[str]:
        """Validate configuration parameters.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        if not (1e9 <= self.center_frequency_hz <= 100e9):
            errors.append("center_frequency_hz must be 1-100 GHz")
        
        if not (0 < self.max_range_m <= 100000):
            errors.append("max_range_m must be 0-100 km")
        
        if not (0 < self.tx_power_w <= 10000):
            errors.append("tx_power_w must be 0-10 kW")
        
        if not (0 < self.bandwidth_hz <= 1e9):
            errors.append("bandwidth_hz must be 0-1 GHz")
        
        if not (0 < self.horizontal_beamwidth_deg <= 180):
            errors.append("horizontal_beamwidth_deg must be 0-180°")
        
        if not (0 < self.vertical_beamwidth_deg <= 180):
            errors.append("vertical_beamwidth_deg must be 0-180°")
        
        return errors
