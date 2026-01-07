"""Tests for waveguide geometry module."""
import pytest
import math
from src.antenna.waveguide import (
    WaveguideConfig,
    create_waveguide,
    get_cutoff_frequency,
    get_guide_wavelength,
    get_wave_impedance,
    validate_operating_frequency,
    WR90_WIDTH_MM,
    WR90_HEIGHT_MM,
)


class TestWaveguideConfig:
    """Tests for WaveguideConfig dataclass."""
    
    def test_default_wr90(self):
        """Default config should be WR-90 dimensions."""
        config = WaveguideConfig()
        assert config.width_mm == WR90_WIDTH_MM
        assert config.height_mm == WR90_HEIGHT_MM
        assert config.width_mm == pytest.approx(22.86, rel=0.01)
        assert config.height_mm == pytest.approx(10.16, rel=0.01)
    
    def test_custom_config(self):
        """Custom configuration should accept parameters."""
        config = WaveguideConfig(
            width_mm=25.0,
            height_mm=12.0,
            length_mm=100.0,
            wall_thickness_mm=3.0
        )
        assert config.width_mm == 25.0
        assert config.height_mm == 12.0


class TestCutoffFrequency:
    """Tests for cutoff frequency calculations."""
    
    def test_wr90_te10_cutoff(self):
        """WR-90 TE10 cutoff should be ~6.56 GHz."""
        config = WaveguideConfig()
        fc = get_cutoff_frequency(config, "TE10")
        # Expected: c / (2 * a) = 299792458 / (2 * 0.02286) = 6.557 GHz
        assert 6.5e9 < fc < 6.6e9
    
    def test_wr90_te20_cutoff(self):
        """WR-90 TE20 cutoff should be ~13.1 GHz."""
        config = WaveguideConfig()
        fc = get_cutoff_frequency(config, "TE20")
        assert 13.0e9 < fc < 13.2e9
    
    def test_above_cutoff_at_operating_freq(self):
        """9.41 GHz should be above TE10 cutoff."""
        config = WaveguideConfig()
        fc = get_cutoff_frequency(config)
        assert 9.41e9 > fc
    
    def test_invalid_mode_raises(self):
        """Invalid mode string should raise ValueError."""
        config = WaveguideConfig()
        with pytest.raises(ValueError):
            get_cutoff_frequency(config, "TM10")  # Wrong format


class TestGuideWavelength:
    """Tests for guide wavelength calculation."""
    
    def test_guide_wavelength_9_4ghz(self):
        """Guide wavelength at 9.4 GHz should be ~40-45mm."""
        config = WaveguideConfig()
        lambda_g = get_guide_wavelength(config, 9.41e9)
        # λg = λ0 / sqrt(1 - (fc/f)^2)
        # λ0 = 31.9mm, fc = 6.56 GHz, f = 9.41 GHz
        assert 0.040 < lambda_g < 0.050  # 40-50 mm
    
    def test_below_cutoff_raises(self):
        """Frequency below cutoff should raise ValueError."""
        config = WaveguideConfig()
        with pytest.raises(ValueError):
            get_guide_wavelength(config, 5e9)  # Below 6.56 GHz cutoff
    
    def test_guide_wavelength_longer_than_free_space(self):
        """Guide wavelength should be longer than free-space wavelength."""
        config = WaveguideConfig()
        freq = 9.41e9
        lambda_0 = 299792458.0 / freq
        lambda_g = get_guide_wavelength(config, freq)
        assert lambda_g > lambda_0


class TestWaveImpedance:
    """Tests for wave impedance calculation."""
    
    def test_wave_impedance_above_free_space(self):
        """TE mode wave impedance should be higher than 377Ω."""
        config = WaveguideConfig()
        Zg = get_wave_impedance(config, 9.41e9)
        assert Zg > 377  # Free-space impedance
        assert Zg < 600  # Reasonable upper bound


class TestOperatingFrequencyValidation:
    """Tests for frequency validation."""
    
    def test_valid_operating_frequency(self):
        """9.41 GHz should be valid for WR-90."""
        config = WaveguideConfig()
        result = validate_operating_frequency(config, 9.41e9)
        assert result['above_cutoff'] is True
        assert result['single_mode'] is True
    
    def test_below_cutoff_invalid(self):
        """5 GHz should be below cutoff."""
        config = WaveguideConfig()
        result = validate_operating_frequency(config, 5e9)
        assert result['above_cutoff'] is False
        assert result['single_mode'] is False
