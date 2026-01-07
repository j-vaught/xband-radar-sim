"""Tests for analytical RCS module."""
import pytest
import numpy as np
from src.scattering.analytical import (
    sphere_rcs_mie,
    flat_plate_rcs,
    corner_reflector_rcs,
    dihedral_rcs,
    cylinder_rcs_broadside,
)


class TestSphereRCS:
    """Tests for Mie series sphere RCS."""
    
    def test_sphere_rcs_positive(self):
        """Sphere RCS should be positive."""
        rcs = sphere_rcs_mie(radius_m=0.1, wavelength_m=0.0319)
        assert rcs > 0
    
    def test_sphere_rcs_optical_limit(self):
        """Large sphere RCS should approach πr² (optical limit)."""
        r = 1.0
        lambda_m = 0.0319
        rcs = sphere_rcs_mie(r, lambda_m)
        optical_rcs = np.pi * r**2
        # Should be within factor of 2 for large sphere
        assert rcs < 2 * optical_rcs


class TestFlatPlateRCS:
    """Tests for flat plate RCS."""
    
    def test_plate_rcs_normal(self):
        """Plate RCS at normal incidence should be maximum."""
        rcs_0 = flat_plate_rcs(
            width_m=1.0,
            height_m=1.0,
            wavelength_m=0.0319,
            theta_inc=0
        )
        rcs_45 = flat_plate_rcs(
            width_m=1.0,
            height_m=1.0,
            wavelength_m=0.0319,
            theta_inc=np.pi/4
        )
        assert rcs_0 > rcs_45
    
    def test_plate_rcs_positive(self):
        """Plate RCS should be positive."""
        rcs = flat_plate_rcs(1.0, 1.0, 0.0319, 0)
        assert rcs > 0


class TestCornerReflectorRCS:
    """Tests for trihedral corner reflector RCS."""
    
    def test_3inch_corner_rcs(self):
        """3-inch corner reflector RCS should be ~0.08 m² at 9.41 GHz."""
        rcs = corner_reflector_rcs(
            edge_length_m=0.0762,  # 3 inches
            wavelength_m=0.0319   # 9.41 GHz
        )
        # Expected: ~0.08-0.14 m² (-10.9 to -8.5 dBsm)
        assert 0.05 < rcs < 0.2
    
    def test_corner_rcs_scales(self):
        """Larger corner reflector should have larger RCS."""
        rcs_small = corner_reflector_rcs(0.05, 0.0319)
        rcs_large = corner_reflector_rcs(0.10, 0.0319)
        assert rcs_large > rcs_small


class TestDihedralRCS:
    """Tests for dihedral corner reflector RCS."""
    
    def test_dihedral_rcs(self):
        """Dihedral RCS should be positive."""
        rcs = dihedral_rcs(
            width_m=0.1,
            height_m=0.1,
            wavelength_m=0.0319
        )
        assert rcs > 0


class TestCylinderRCS:
    """Tests for cylinder RCS."""
    
    def test_cylinder_rcs_broadside(self):
        """Cylinder RCS at broadside should be positive."""
        rcs = cylinder_rcs_broadside(
            radius_m=0.05,
            length_m=0.5,
            wavelength_m=0.0319
        )
        assert rcs > 0
