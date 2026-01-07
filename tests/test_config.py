"""Tests for configuration module."""
import pytest
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import RadarConfig


class TestRadarConfig:
    """Tests for RadarConfig dataclass."""
    
    def test_default_values(self):
        """Default config should have Furuno DRS4D-NXT values."""
        config = RadarConfig()
        assert config.center_frequency_hz == 9.41e9
        assert config.name == "Furuno DRS4D-NXT"
        assert config.horizontal_beamwidth_deg == 3.9
        assert config.vertical_beamwidth_deg == 25.0
    
    def test_wavelength_calculation(self):
        """Wavelength at 9.41 GHz should be ~31.9 mm."""
        config = RadarConfig()
        expected = 299792458.0 / 9.41e9
        assert abs(config.wavelength_m - expected) < 1e-10
        assert abs(config.wavelength_m - 0.0319) < 0.001
    
    def test_wavenumber_calculation(self):
        """Wavenumber k = 2π/λ."""
        config = RadarConfig()
        import math
        expected = 2 * math.pi / config.wavelength_m
        assert abs(config.wavenumber - expected) < 1e-10
    
    def test_yaml_roundtrip(self):
        """Config should survive YAML serialize/deserialize."""
        config = RadarConfig(tx_power_w=100.0, n_rays=5000)
        
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            config.to_yaml(f.name)
            loaded = RadarConfig.from_yaml(f.name)
        
        assert loaded.tx_power_w == 100.0
        assert loaded.n_rays == 5000
        assert loaded.center_frequency_hz == config.center_frequency_hz
    
    def test_validation_valid(self):
        """Valid config should return no errors."""
        config = RadarConfig()
        errors = config.validate()
        assert len(errors) == 0
    
    def test_validation_invalid_frequency(self):
        """Invalid frequency should trigger error."""
        config = RadarConfig(center_frequency_hz=1e6)  # 1 MHz - too low
        errors = config.validate()
        assert any("frequency" in e.lower() for e in errors)
    
    def test_validation_invalid_power(self):
        """Invalid power should trigger error."""
        config = RadarConfig(tx_power_w=-10)  # Negative power
        errors = config.validate()
        assert any("power" in e.lower() for e in errors)
