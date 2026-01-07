"""Analytical RCS formulas for validation."""
import numpy as np
from typing import Tuple
from scipy.special import spherical_jn, spherical_yn


def sphere_rcs_optical(radius_m: float) -> float:
    """
    Optical limit RCS for PEC sphere.
    
    Valid for ka >> 1 (electrically large sphere).
    σ = π × a²
    
    Args:
        radius_m: Sphere radius in meters
        
    Returns:
        RCS in m²
    """
    return np.pi * radius_m ** 2


def sphere_rcs_mie(
    radius_m: float,
    wavelength_m: float,
    n_terms: int = 50
) -> float:
    """
    Mie series RCS for PEC sphere.
    
    Valid for all ka (including resonant regime).
    
    Args:
        radius_m: Sphere radius
        wavelength_m: Wavelength
        n_terms: Number of series terms
        
    Returns:
        Backscatter RCS in m²
    """
    k = 2 * np.pi / wavelength_m
    ka = k * radius_m
    
    sigma_sum = 0.0 + 0.0j
    
    for n in range(1, n_terms + 1):
        # Spherical Bessel functions
        jn = spherical_jn(n, ka)
        jn_prime = spherical_jn(n, ka, derivative=True)
        yn = spherical_yn(n, ka)
        yn_prime = spherical_yn(n, ka, derivative=True)
        
        # Hankel function (second kind) h_n = j_n - i*y_n
        hn = jn - 1j * yn
        hn_prime = jn_prime - 1j * yn_prime
        
        # Scattering coefficients for PEC
        # a_n = j_n'(ka) / h_n'(ka)
        # b_n = j_n(ka) / h_n(ka)
        if np.abs(hn_prime) > 1e-20:
            an = jn_prime / hn_prime
        else:
            an = 0
            
        if np.abs(hn) > 1e-20:
            bn = jn / hn
        else:
            bn = 0
        
        # Contribution to backscatter RCS
        sigma_sum += ((-1) ** n) * (2 * n + 1) * (bn - an)
    
    rcs = (wavelength_m ** 2 / np.pi) * np.abs(sigma_sum) ** 2
    return rcs


def flat_plate_rcs(
    width_m: float,
    height_m: float,
    wavelength_m: float,
    theta_inc: float = 0.0
) -> float:
    """
    Physical Optics RCS for flat rectangular plate.
    
    At normal incidence: σ = 4π A² / λ²
    
    Args:
        width_m: Plate width
        height_m: Plate height
        wavelength_m: Wavelength
        theta_inc: Incidence angle from normal (radians)
        
    Returns:
        RCS in m²
    """
    A = width_m * height_m
    
    if abs(theta_inc) < 0.01:  # Normal incidence
        return (4 * np.pi * A ** 2) / wavelength_m ** 2
    
    # Off-normal (simplified sinc pattern in one dimension)
    k = 2 * np.pi / wavelength_m
    u = k * width_m * np.sin(theta_inc)
    
    if abs(u) < 1e-10:
        sinc_term = 1.0
    else:
        sinc_term = (np.sin(u) / u) ** 2
    
    return (4 * np.pi * A ** 2 / wavelength_m ** 2) * sinc_term


def corner_reflector_rcs(
    edge_length_m: float,
    wavelength_m: float
) -> float:
    """
    RCS for trihedral corner reflector at normal incidence.
    
    σ = 4π L⁴ / (3λ²)
    
    Valid for L/λ > ~1 (resonant and optical regimes).
    
    Args:
        edge_length_m: Corner edge length
        wavelength_m: Wavelength
        
    Returns:
        RCS in m²
    """
    return (4 * np.pi * edge_length_m ** 4) / (3 * wavelength_m ** 2)


def corner_reflector_rcs_3in_9_4ghz() -> Tuple[float, float]:
    """
    Reference RCS for 3-inch corner at 9.41 GHz.
    
    Returns:
        (rcs_m2, rcs_dbsm)
    """
    L = 0.0762       # 3 inches in meters
    wavelength = 0.0319  # 9.41 GHz
    
    rcs_m2 = corner_reflector_rcs(L, wavelength)
    rcs_dbsm = 10 * np.log10(rcs_m2)
    
    return rcs_m2, rcs_dbsm


def dihedral_rcs(
    width_m: float,
    height_m: float,
    wavelength_m: float
) -> float:
    """
    RCS for dihedral corner reflector.
    
    σ = 8π w² h² / λ²
    
    Args:
        width_m: Dihedral width
        height_m: Dihedral height
        wavelength_m: Wavelength
        
    Returns:
        RCS in m²
    """
    return (8 * np.pi * width_m ** 2 * height_m ** 2) / wavelength_m ** 2


def cylinder_rcs_broadside(
    radius_m: float,
    length_m: float,
    wavelength_m: float
) -> float:
    """
    RCS for cylinder at broadside incidence (optical limit).
    
    σ = 2π r L² / λ
    
    Args:
        radius_m: Cylinder radius
        length_m: Cylinder length
        wavelength_m: Wavelength
        
    Returns:
        RCS in m²
    """
    return (2 * np.pi * radius_m * length_m ** 2) / wavelength_m
