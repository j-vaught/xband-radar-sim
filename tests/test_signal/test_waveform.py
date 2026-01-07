"""Tests for waveform generation module."""
import pytest
import numpy as np
from src.signal.waveform import (
    generate_lfm_chirp,
    apply_window,
    WaveformConfig,
)


class TestWaveformConfig:
    """Tests for WaveformConfig dataclass."""
    
    def test_default_values(self):
        """Default config should have reasonable values."""
        config = WaveformConfig()
        assert config.pulse_width_s > 0
        assert config.bandwidth_hz > 0
        assert config.sample_rate_hz > 0
    
    def test_time_bandwidth_product(self):
        """BT product should be B × T."""
        config = WaveformConfig(
            bandwidth_hz=50e6,
            pulse_width_s=1e-6
        )
        assert config.time_bandwidth_product == pytest.approx(50.0)


class TestLFMChirp:
    """Tests for LFM chirp generation."""
    
    def test_chirp_returns_tuple(self):
        """Chirp should return (time, signal) tuple."""
        config = WaveformConfig()
        result = generate_lfm_chirp(config)
        assert len(result) == 2
        time, signal = result
        assert len(time) == len(signal)
    
    def test_chirp_length(self):
        """Chirp should have correct number of samples."""
        config = WaveformConfig(
            pulse_width_s=1e-6,
            sample_rate_hz=100e6
        )
        time, signal = generate_lfm_chirp(config)
        expected_samples = int(config.pulse_width_s * config.sample_rate_hz)
        assert len(signal) == expected_samples
    
    def test_chirp_is_complex(self):
        """Chirp should be complex-valued."""
        config = WaveformConfig()
        _, signal = generate_lfm_chirp(config)
        assert np.iscomplexobj(signal)
    
    def test_chirp_frequency_increases(self):
        """LFM chirp instantaneous frequency should increase."""
        config = WaveformConfig(bandwidth_hz=10e6)
        _, signal = generate_lfm_chirp(config)
        
        # Phase should have positive second derivative
        phase = np.unwrap(np.angle(signal))
        inst_freq = np.diff(phase)
        # Should generally increase
        assert inst_freq[-1] > inst_freq[0]


class TestWindow:
    """Tests for windowing functions."""
    
    def test_hamming_window(self):
        """Hamming window should taper edges."""
        signal = np.ones(100)
        windowed = apply_window(signal, 'hamming')
        assert windowed[0] < windowed[50]
        assert windowed[-1] < windowed[50]
    
    def test_rectangular_window(self):
        """Rectangular window should not change signal."""
        signal = np.ones(100)
        windowed = apply_window(signal, 'rectangular')
        assert np.allclose(signal, windowed)
    
    def test_hanning_window(self):
        """Hann window should taper edges."""
        signal = np.ones(100)
        windowed = apply_window(signal, 'hanning')
        assert windowed[0] < windowed[50]
