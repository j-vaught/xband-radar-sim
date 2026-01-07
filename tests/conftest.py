"""Shared pytest fixtures for all test modules."""
import pytest
import numpy as np
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import RadarConfig
from interfaces import AntennaPattern, RayBundle, ScatterResult


# === Configuration Fixtures ===

@pytest.fixture
def default_config():
    """Standard Furuno DRS4D-NXT configuration."""
    return RadarConfig()


@pytest.fixture
def test_config():
    """Reduced config for faster tests."""
    return RadarConfig(n_rays=1000, enable_gpu=False)


# === Antenna Fixtures ===

@pytest.fixture
def isotropic_pattern():
    """Isotropic antenna pattern for testing."""
    theta = np.linspace(-90, 90, 181)
    phi = np.linspace(-180, 180, 361)
    return AntennaPattern(
        frequency_hz=9.41e9,
        theta_deg=theta,
        phi_deg=phi,
        gain_linear=np.ones((181, 361)),
        phase_rad=np.zeros((181, 361))
    )


@pytest.fixture
def dipole_pattern():
    """Half-wave dipole pattern for validation."""
    theta = np.linspace(0, 180, 181)
    phi = np.array([0.0])
    theta_rad = np.deg2rad(theta)
    
    # Analytical dipole pattern: |cos(π/2 cos θ) / sin θ|
    with np.errstate(divide='ignore', invalid='ignore'):
        pattern = np.abs(
            np.cos(np.pi / 2 * np.cos(theta_rad)) /
            (np.sin(theta_rad) + 1e-10)
        )
    pattern = np.nan_to_num(pattern, nan=0.0, posinf=0.0)
    
    return AntennaPattern(
        frequency_hz=9.41e9,
        theta_deg=theta,
        phi_deg=phi,
        gain_linear=pattern.reshape(-1, 1),
        phase_rad=np.zeros((181, 1))
    )


# === Geometry Fixtures ===

@pytest.fixture
def fixtures_dir():
    """Path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def corner_reflector_3in(fixtures_dir):
    """Path to 3-inch corner reflector STL file."""
    return fixtures_dir / "corner_3in.stl"


# === Ray Fixtures ===

@pytest.fixture
def simple_ray_bundle():
    """Single ray pointing down z-axis."""
    return RayBundle(
        n_rays=1,
        origins=np.array([[0, 0, 0]], dtype=np.float64),
        directions=np.array([[0, 0, 1]], dtype=np.float64),
        hit_points=np.array([[0, 0, 1000]], dtype=np.float64),
        hit_normals=np.array([[0, 0, -1]], dtype=np.float64),
        path_lengths_m=np.array([1000.0]),
        incident_powers_w=np.array([1.0]),
        hit_mask=np.array([True])
    )


# === Scatter Fixtures ===

@pytest.fixture
def corner_scatter_2km():
    """Scatter result for corner reflector at 2 km."""
    return ScatterResult(
        n_scatterers=1,
        scatter_points=np.array([[2000, 0, 0]], dtype=np.float64),
        rcs_m2=np.array([0.082]),
        scattered_power_w=np.array([1e-10]),
        path_to_rx_m=np.array([2000.0]),
        phase_rad=np.array([0.0])
    )
