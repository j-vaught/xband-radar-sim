"""Path loss computation and utilities."""
import numpy as np
from typing import Tuple

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from interfaces import RayBundle, AntennaPattern


def free_space_path_loss(
    distance_m: np.ndarray,
    frequency_hz: float
) -> np.ndarray:
    """
    Compute free-space path loss.
    
    FSPL = (4πR/λ)² = (4πRf/c)²
    
    Args:
        distance_m: Distance(s) in meters
        frequency_hz: Frequency in Hz
        
    Returns:
        Path loss (linear, not dB)
    """
    c = 299792458.0
    wavelength_m = c / frequency_hz
    
    return (4 * np.pi * distance_m / wavelength_m) ** 2


def free_space_path_loss_db(
    distance_m: np.ndarray,
    frequency_hz: float
) -> np.ndarray:
    """
    Compute free-space path loss in dB.
    
    FSPL_dB = 20*log10(4πR/λ) = 20*log10(R) + 20*log10(f) - 147.55
    
    Args:
        distance_m: Distance(s) in meters
        frequency_hz: Frequency in Hz
        
    Returns:
        Path loss in dB
    """
    loss_linear = free_space_path_loss(distance_m, frequency_hz)
    return 10 * np.log10(loss_linear)


def two_ray_path_loss(
    distance_m: np.ndarray,
    tx_height_m: float,
    rx_height_m: float,
    frequency_hz: float,
    ground_reflection_coeff: float = -1.0
) -> np.ndarray:
    """
    Two-ray ground reflection model.
    
    For marine radar over water, ground reflection can be significant.
    
    Args:
        distance_m: Horizontal distance
        tx_height_m: Transmitter height
        rx_height_m: Receiver height (target height for monostatic)
        frequency_hz: Frequency
        ground_reflection_coeff: Reflection coefficient (default -1 for PEC)
        
    Returns:
        Path loss (linear)
    """
    c = 299792458.0
    wavelength_m = c / frequency_hz
    k = 2 * np.pi / wavelength_m
    
    # Direct path
    R_direct = np.sqrt(distance_m**2 + (tx_height_m - rx_height_m)**2)
    
    # Reflected path
    R_reflected = np.sqrt(distance_m**2 + (tx_height_m + rx_height_m)**2)
    
    # Phase difference
    delta_phi = k * (R_reflected - R_direct)
    
    # Combined field (with reflection coefficient)
    field_ratio = 1 + ground_reflection_coeff * np.exp(-1j * delta_phi)
    
    # FSPL of direct path
    fspl = free_space_path_loss(R_direct, frequency_hz)
    
    # Two-ray loss
    return fspl / (np.abs(field_ratio) ** 2 + 1e-20)


def apply_antenna_weighting(
    bundle: RayBundle,
    pattern: AntennaPattern
) -> RayBundle:
    """
    Apply antenna pattern weighting to ray powers.
    
    Args:
        bundle: RayBundle from ray tracing
        pattern: Antenna pattern
        
    Returns:
        RayBundle with weighted powers
    """
    n_rays = bundle.n_rays
    weights = np.ones(n_rays)
    
    for i in range(n_rays):
        if bundle.hit_mask[i]:
            direction = bundle.directions[i]
            
            # Convert direction to spherical angles
            theta = np.rad2deg(np.arccos(np.clip(direction[2], -1, 1)))
            phi = np.rad2deg(np.arctan2(direction[1], direction[0]))
            
            # Get gain at this angle
            weights[i] = pattern.gain_at(theta, phi)
    
    return RayBundle(
        n_rays=bundle.n_rays,
        origins=bundle.origins,
        directions=bundle.directions,
        hit_points=bundle.hit_points,
        hit_normals=bundle.hit_normals,
        path_lengths_m=bundle.path_lengths_m,
        incident_powers_w=bundle.incident_powers_w * weights,
        hit_mask=bundle.hit_mask
    )


def apply_path_loss(
    bundle: RayBundle,
    frequency_hz: float
) -> RayBundle:
    """
    Apply free-space path loss to ray powers.
    
    Args:
        bundle: RayBundle from ray tracing
        frequency_hz: Operating frequency
        
    Returns:
        RayBundle with attenuated powers
    """
    # Compute FSPL for each path
    fspl = free_space_path_loss(bundle.path_lengths_m, frequency_hz)
    
    # Avoid division by zero
    fspl = np.maximum(fspl, 1.0)
    
    return RayBundle(
        n_rays=bundle.n_rays,
        origins=bundle.origins,
        directions=bundle.directions,
        hit_points=bundle.hit_points,
        hit_normals=bundle.hit_normals,
        path_lengths_m=bundle.path_lengths_m,
        incident_powers_w=bundle.incident_powers_w / fspl,
        hit_mask=bundle.hit_mask
    )


def compute_radar_range_equation(
    tx_power_w: float,
    antenna_gain_linear: float,
    wavelength_m: float,
    target_rcs_m2: float,
    range_m: float
) -> float:
    """
    Compute received power using radar range equation.
    
    Pr = (Pt × G² × λ² × σ) / ((4π)³ × R⁴)
    
    Args:
        tx_power_w: Transmit power
        antenna_gain_linear: Antenna gain (linear, not dB)
        wavelength_m: Wavelength
        target_rcs_m2: Target RCS
        range_m: Range to target
        
    Returns:
        Received power in Watts
    """
    numerator = tx_power_w * antenna_gain_linear**2 * wavelength_m**2 * target_rcs_m2
    denominator = (4 * np.pi)**3 * range_m**4
    
    return numerator / denominator


def compute_max_detection_range(
    tx_power_w: float,
    antenna_gain_linear: float,
    wavelength_m: float,
    target_rcs_m2: float,
    min_detectable_power_w: float
) -> float:
    """
    Compute maximum detection range.
    
    Rmax = ((Pt × G² × λ² × σ) / ((4π)³ × Pmin))^(1/4)
    
    Args:
        tx_power_w: Transmit power
        antenna_gain_linear: Antenna gain
        wavelength_m: Wavelength
        target_rcs_m2: Target RCS
        min_detectable_power_w: Minimum detectable power
        
    Returns:
        Maximum range in meters
    """
    numerator = tx_power_w * antenna_gain_linear**2 * wavelength_m**2 * target_rcs_m2
    denominator = (4 * np.pi)**3 * min_detectable_power_w
    
    return (numerator / denominator) ** 0.25
