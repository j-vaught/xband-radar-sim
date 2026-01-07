"""Tests for detection module."""
import pytest
import numpy as np
from src.signal.detection import (
    cfar_1d,
    cfar_os,
    find_peaks,
    peaks_to_ranges,
    CFARConfig,
)


class TestCFARCA:
    """Tests for Cell-Averaging CFAR."""
    
    def test_cfar_detects_strong_target(self):
        """CFAR should detect target well above noise."""
        np.random.seed(42)
        signal = np.random.randn(500) + 1j * np.random.randn(500)
        signal[250] = 50 + 0j  # Strong target
        
        config = CFARConfig(guard_cells=4, training_cells=16, pfa=1e-4)
        detections, threshold = cfar_1d(signal, config)
        
        assert detections[250] or any(detections[245:255])
    
    def test_cfar_threshold_shape(self):
        """Threshold should match signal length."""
        signal = np.random.randn(100) + 1j * np.random.randn(100)
        config = CFARConfig(guard_cells=2, training_cells=8)
        detections, threshold = cfar_1d(signal, config)
        assert len(threshold) == len(signal)
    
    def test_cfar_returns_boolean_detections(self):
        """Detections should be boolean array."""
        signal = np.random.randn(100) + 0j
        config = CFARConfig()
        detections, _ = cfar_1d(signal, config)
        assert detections.dtype == bool


class TestCFAROS:
    """Tests for Order-Statistics CFAR."""
    
    def test_cfar_os_detects_target(self):
        """OS-CFAR should detect strong target."""
        np.random.seed(42)
        signal = np.random.randn(500) + 1j * np.random.randn(500)
        signal[250] = 40 + 0j
        
        config = CFARConfig(guard_cells=4, training_cells=24, pfa=1e-4)
        detections, threshold = cfar_os(signal, config, k=18)
        
        assert any(detections[245:255])


class TestPeakFinding:
    """Tests for peak finding."""
    
    def test_find_single_peak(self):
        """Should find single peak."""
        signal = np.zeros(100)
        signal[50] = 1.0
        detections = np.zeros(100, dtype=bool)
        detections[50] = True
        
        peaks = find_peaks(signal, detections)
        assert 50 in peaks
    
    def test_find_multiple_peaks(self):
        """Should find multiple separated peaks."""
        signal = np.zeros(200)
        signal[50] = 1.0
        signal[150] = 0.8
        
        detections = np.zeros(200, dtype=bool)
        detections[50] = True
        detections[150] = True
        
        peaks = find_peaks(signal, detections, min_separation=10)
        assert len(peaks) == 2


class TestRangeConversion:
    """Tests for range-index conversion."""
    
    def test_peaks_to_ranges(self):
        """Should convert peak indices to range."""
        sample_rate = 100e6
        peaks = [1000]
        
        ranges = peaks_to_ranges(peaks, sample_rate)
        # Time = 1000 / 100e6 = 10us
        # Range = c * t / 2 = 3e8 * 10e-6 / 2 = 1500m
        assert ranges[0] == pytest.approx(1500, rel=0.01)
    
    def test_empty_peaks(self):
        """Should handle empty peak list."""
        ranges = peaks_to_ranges([], 100e6)
        assert len(ranges) == 0
