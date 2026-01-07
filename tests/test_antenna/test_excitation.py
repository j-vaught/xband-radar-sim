"""Tests for excitation module."""
import pytest
from src.antenna.excitation import ExcitationConfig


class TestExcitationConfig:
    """Tests for ExcitationConfig dataclass."""
    
    def test_default_frequency(self):
        """Default frequency should be 9.41 GHz."""
        config = ExcitationConfig()
        assert config.center_freq_hz == pytest.approx(9.41e9)
    
    def test_default_bandwidth(self):
        """Default bandwidth should be 2 GHz."""
        config = ExcitationConfig()
        assert config.bandwidth_hz == pytest.approx(2e9)
    
    def test_default_impedance(self):
        """Default port impedance should be 50 ohms."""
        config = ExcitationConfig()
        assert config.port_impedance == pytest.approx(50.0)
    
    def test_custom_config(self):
        """Custom config should accept parameters."""
        config = ExcitationConfig(
            center_freq_hz=10e9,
            bandwidth_hz=1e9,
            port_impedance=75.0
        )
        assert config.center_freq_hz == 10e9
        assert config.bandwidth_hz == 1e9
        assert config.port_impedance == 75.0
