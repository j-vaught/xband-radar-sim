"""Tests for path loss module."""
import pytest
import numpy as np
from src.propagation.pathloss import (
    free_space_path_loss_db,
    two_ray_path_loss,
    compute_radar_range_equation,
)


class TestFreeSpacePathLoss:
    """Tests for FSPL calculation."""
    
    def test_fspl_1km_9ghz(self):
        """FSPL at 1km, 9.4 GHz should be ~111 dB."""
        fspl = free_space_path_loss_db(1000, 9.41e9)
        assert 110 < fspl < 114
    
    def test_fspl_increases_with_range(self):
        """FSPL should increase with range."""
        fspl_1km = free_space_path_loss_db(1000, 9.41e9)
        fspl_2km = free_space_path_loss_db(2000, 9.41e9)
        assert fspl_2km > fspl_1km
        # Should increase by ~6 dB for double distance
        assert fspl_2km - fspl_1km == pytest.approx(6.0, abs=0.5)
    
    def test_fspl_increases_with_frequency(self):
        """FSPL should increase with frequency."""
        fspl_5ghz = free_space_path_loss_db(1000, 5e9)
        fspl_10ghz = free_space_path_loss_db(1000, 10e9)
        assert fspl_10ghz > fspl_5ghz


class TestTwoRayModel:
    """Tests for two-ray ground reflection model."""
    
    def test_two_ray_basic(self):
        """Two-ray model should give reasonable path loss."""
        pl = two_ray_path_loss(
            distance_m=1000,
            tx_height_m=5,
            rx_height_m=1,
            frequency_hz=9.41e9
        )
        # Should return linear path loss > 1
        assert pl > 1


class TestRadarRangeEquation:
    """Tests for radar range equation."""
    
    def test_rre_basic(self):
        """Radar range equation should compute received power."""
        pr = compute_radar_range_equation(
            tx_power_w=25,
            antenna_gain_linear=158,  # ~22 dBi
            wavelength_m=0.0319,
            target_rcs_m2=0.082,
            range_m=1000
        )
        # Result should be very small (weak signal)
        assert pr > 0
        assert pr < 1e-6
    
    def test_rre_closer_stronger(self):
        """Closer targets should have stronger returns."""
        pr_1km = compute_radar_range_equation(25, 158, 0.0319, 0.1, 1000)
        pr_2km = compute_radar_range_equation(25, 158, 0.0319, 0.1, 2000)
        assert pr_1km > pr_2km
        # Should be 16× difference (R^4)
        assert pr_1km / pr_2km == pytest.approx(16.0, rel=0.1)
    
    def test_rre_larger_rcs_stronger(self):
        """Larger RCS should give stronger return."""
        pr_small = compute_radar_range_equation(25, 158, 0.0319, 0.01, 1000)
        pr_large = compute_radar_range_equation(25, 158, 0.0319, 1.0, 1000)
        assert pr_large > pr_small
