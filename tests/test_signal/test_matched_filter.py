"""Tests for matched filter module."""
import pytest
import numpy as np
from src.signal.matched_filter import (
    matched_filter,
    matched_filter_fft,
    compute_compression_gain,
    measure_resolution,
)


class TestMatchedFilter:
    """Tests for matched filter application."""
    
    def test_matched_filter_output_exists(self):
        """Should return compressed signal."""
        signal = np.random.randn(100) + 1j * np.random.randn(100)
        reference = np.random.randn(50) + 1j * np.random.randn(50)
        
        output = matched_filter(signal, reference)
        assert len(output) > 0
    
    def test_matched_filter_peak_at_target(self):
        """Peak should occur at target location."""
        reference = np.exp(1j * np.linspace(0, 10, 50))
        signal = np.zeros(200, dtype=complex)
        delay = 75
        signal[delay:delay+50] = reference
        
        output = matched_filter(signal, reference)
        peak_idx = np.argmax(np.abs(output))
        
        # Peak should be near center of the matched output
        assert abs(peak_idx - 100) < 30  # Within 30 samples of center


class TestMatchedFilterFFT:
    """Tests for FFT-based matched filter."""
    
    def test_fft_returns_same_length(self):
        """FFT method should return same length as input."""
        signal = np.random.randn(100) + 1j * np.random.randn(100)
        reference = np.random.randn(30) + 1j * np.random.randn(30)
        
        output_fft = matched_filter_fft(signal, reference)
        assert len(output_fft) == len(signal)


class TestCompressionGain:
    """Tests for compression gain measurement."""
    
    def test_compression_gain_positive(self):
        """Compression gain should be positive for typical signals."""
        input_signal = np.random.randn(100) + 1j * np.random.randn(100)
        # Simulate compression - peak should be higher
        compressed = np.abs(input_signal) * 10
        
        gain = compute_compression_gain(input_signal, compressed)
        assert gain > 0


class TestMeasureResolution:
    """Tests for range resolution measurement."""
    
    def test_resolution_positive(self):
        """Range resolution should be positive."""
        compressed = np.zeros(100, dtype=complex)
        compressed[50] = 1.0
        compressed[49] = 0.8
        compressed[51] = 0.8
        
        sample_rate = 100e6
        resolution, sidelobe = measure_resolution(compressed, sample_rate)
        assert resolution > 0
    
    def test_resolution_in_meters(self):
        """Resolution should be in reasonable range for radar."""
        compressed = np.exp(-np.linspace(-3, 3, 100)**2) + 0j
        sample_rate = 100e6
        
        resolution, _ = measure_resolution(compressed, sample_rate)
        assert 0.1 < resolution < 1000
