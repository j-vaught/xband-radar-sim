"""Tests for interface dataclasses."""
import pytest
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from interfaces import AntennaPattern, RayBundle, ScatterResult, RangeProfile


class TestAntennaPattern:
    """Tests for AntennaPattern dataclass."""
    
    def test_creation(self):
        """Should create pattern with correct attributes."""
        pattern = AntennaPattern(
            frequency_hz=9.41e9,
            theta_deg=np.array([0, 45, 90]),
            phi_deg=np.array([0, 90, 180]),
            gain_linear=np.ones((3, 3)),
            phase_rad=np.zeros((3, 3))
        )
        assert pattern.frequency_hz == 9.41e9
        assert pattern.gain_linear.shape == (3, 3)
    
    def test_gain_interpolation(self):
        """Gain interpolation should return correct values at corners."""
        theta = np.array([0.0, 90.0])
        phi = np.array([0.0, 90.0])
        gain = np.array([[1.0, 0.5], [0.5, 0.25]])
        
        pattern = AntennaPattern(
            frequency_hz=9.41e9,
            theta_deg=theta,
            phi_deg=phi,
            gain_linear=gain,
            phase_rad=np.zeros((2, 2))
        )
        
        # Corner values
        assert abs(pattern.gain_at(0, 0) - 1.0) < 0.01
        assert abs(pattern.gain_at(90, 90) - 0.25) < 0.01
    
    def test_gain_dbi(self):
        """Peak gain should be computed correctly."""
        pattern = AntennaPattern(
            frequency_hz=9.41e9,
            theta_deg=np.array([0, 90]),
            phi_deg=np.array([0, 90]),
            gain_linear=np.array([[158.5, 1.0], [1.0, 1.0]]),  # ~22 dBi peak
            phase_rad=np.zeros((2, 2))
        )
        assert abs(pattern.gain_dbi() - 22.0) < 0.1


class TestRayBundle:
    """Tests for RayBundle dataclass."""
    
    def test_creation(self):
        """Should create bundle with correct attributes."""
        n = 100
        bundle = RayBundle(
            n_rays=n,
            origins=np.zeros((n, 3)),
            directions=np.tile([0, 0, 1], (n, 1)),
            hit_points=np.zeros((n, 3)),
            hit_normals=np.tile([0, 0, -1], (n, 1)),
            path_lengths_m=np.ones(n) * 1000,
            incident_powers_w=np.ones(n),
            hit_mask=np.ones(n, dtype=bool)
        )
        assert bundle.n_rays == 100
        assert bundle.origins.shape == (100, 3)


class TestScatterResult:
    """Tests for ScatterResult dataclass."""
    
    def test_creation(self):
        """Should create result with correct attributes."""
        result = ScatterResult(
            n_scatterers=1,
            scatter_points=np.array([[2000, 0, 0]]),
            rcs_m2=np.array([0.082]),
            scattered_power_w=np.array([1e-10]),
            path_to_rx_m=np.array([2000.0]),
            phase_rad=np.array([0.0])
        )
        assert result.n_scatterers == 1
        assert result.rcs_m2[0] == 0.082


class TestRangeProfile:
    """Tests for RangeProfile dataclass."""
    
    def test_creation(self):
        """Should create profile with correct attributes."""
        profile = RangeProfile(
            range_bins_m=np.linspace(0, 3000, 100),
            amplitude_db=np.zeros(100),
            snr_db=15.0,
            detected_ranges_m=np.array([2000.0])
        )
        assert profile.snr_db == 15.0
        assert len(profile.range_bins_m) == 100
