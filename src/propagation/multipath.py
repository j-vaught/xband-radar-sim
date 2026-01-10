"""Multipath propagation modeling for radar simulation.

Implements:
- Two-ray ground reflection model
- Specular surface reflections
- Diffuse scattering from rough surfaces
- Multiple bounce paths

References:
- Skolnik, M.I., Introduction to Radar Systems
- Long, M.W., Radar Reflectivity of Land and Sea
"""
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class SurfaceProperties:
    """Properties of a reflecting surface.

    Attributes:
        dielectric_constant: Relative permittivity (complex)
        conductivity: Conductivity in S/m
        roughness_m: RMS surface roughness in meters
        reflection_coefficient: Override reflection coefficient (if set)
    """
    dielectric_constant: complex = 80 + 0j  # Water default
    conductivity: float = 4.0  # S/m (seawater)
    roughness_m: float = 0.1  # 10cm RMS roughness
    reflection_coefficient: Optional[complex] = None


# Preset surface types
SURFACE_SEAWATER = SurfaceProperties(
    dielectric_constant=80 + 70j,
    conductivity=4.0,
    roughness_m=0.3  # Moderate sea state
)

SURFACE_FRESHWATER = SurfaceProperties(
    dielectric_constant=80 + 4j,
    conductivity=0.01,
    roughness_m=0.1
)

SURFACE_WET_GROUND = SurfaceProperties(
    dielectric_constant=25 + 5j,
    conductivity=0.02,
    roughness_m=0.05
)

SURFACE_DRY_GROUND = SurfaceProperties(
    dielectric_constant=4 + 0.1j,
    conductivity=0.001,
    roughness_m=0.02
)

SURFACE_CONCRETE = SurfaceProperties(
    dielectric_constant=6 + 0.3j,
    conductivity=0.01,
    roughness_m=0.01
)

SURFACE_METAL = SurfaceProperties(
    dielectric_constant=1 + 1e7j,  # Effectively PEC
    conductivity=1e7,
    roughness_m=0.001,
    reflection_coefficient=-1.0 + 0j
)


def fresnel_reflection_coefficient(grazing_angle_rad: float,
                                    frequency_hz: float,
                                    surface: SurfaceProperties,
                                    polarization: str = 'H') -> complex:
    """
    Calculate Fresnel reflection coefficient.

    Args:
        grazing_angle_rad: Grazing angle in radians (from surface)
        frequency_hz: Frequency in Hz
        surface: Surface properties
        polarization: 'H' (horizontal) or 'V' (vertical)

    Returns:
        Complex reflection coefficient
    """
    if surface.reflection_coefficient is not None:
        return surface.reflection_coefficient

    # Complex permittivity including conductivity
    eps_0 = 8.854e-12
    omega = 2 * np.pi * frequency_hz
    eps_r = surface.dielectric_constant + surface.conductivity / (1j * omega * eps_0)

    sin_psi = np.sin(grazing_angle_rad)
    cos_psi = np.cos(grazing_angle_rad)

    # Avoid numerical issues at very small angles
    if grazing_angle_rad < 1e-6:
        return -1.0 + 0j

    sqrt_term = np.sqrt(eps_r - cos_psi**2)

    if polarization.upper() == 'H':
        # Horizontal polarization (TE)
        rho = (sin_psi - sqrt_term) / (sin_psi + sqrt_term)
    else:
        # Vertical polarization (TM)
        rho = (eps_r * sin_psi - sqrt_term) / (eps_r * sin_psi + sqrt_term)

    return rho


def roughness_factor(grazing_angle_rad: float,
                     frequency_hz: float,
                     surface: SurfaceProperties) -> float:
    """
    Calculate roughness reduction factor using Ament model.

    Rough surfaces reduce specular reflection.

    Args:
        grazing_angle_rad: Grazing angle from surface
        frequency_hz: Frequency in Hz
        surface: Surface properties

    Returns:
        Roughness factor (0-1), multiply with reflection coefficient
    """
    c = 299792458.0
    wavelength = c / frequency_hz

    # Rayleigh roughness parameter
    g = (4 * np.pi * surface.roughness_m * np.sin(grazing_angle_rad) / wavelength) ** 2

    # Ament roughness factor
    rho_s = np.exp(-g / 2)

    # Also accounts for some diffuse scatter
    if g > 0.01:
        rho_s *= (1 + 0.5 * g * np.exp(-g))

    return np.clip(rho_s, 0.0, 1.0)


def divergence_factor(grazing_angle_rad: float,
                      range_m: float,
                      earth_radius_m: float = 8.5e6) -> float:
    """
    Calculate spherical Earth divergence factor.

    Accounts for beam spreading due to Earth curvature.

    Args:
        grazing_angle_rad: Grazing angle
        range_m: Range to reflection point
        earth_radius_m: Effective Earth radius

    Returns:
        Divergence factor (0-1)
    """
    if range_m < 100:
        return 1.0

    sin_psi = np.sin(grazing_angle_rad)

    # Divergence factor
    D = 1.0 / np.sqrt(1 + 2 * range_m / (earth_radius_m * sin_psi))

    return np.clip(D, 0.0, 1.0)


def two_ray_multipath(radar_height_m: float,
                      target_height_m: float,
                      horizontal_range_m: float,
                      frequency_hz: float,
                      surface: SurfaceProperties = SURFACE_SEAWATER,
                      polarization: str = 'H') -> Tuple[complex, float, float]:
    """
    Two-ray ground reflection model.

    Computes the interference between direct and surface-reflected paths.

    Args:
        radar_height_m: Radar antenna height
        target_height_m: Target height
        horizontal_range_m: Horizontal distance to target
        frequency_hz: Frequency in Hz
        surface: Surface properties
        polarization: 'H' or 'V'

    Returns:
        multipath_factor: Complex field factor (multiply with signal)
        direct_path_m: Direct path length
        reflected_path_m: Reflected path length
    """
    c = 299792458.0
    wavelength = c / frequency_hz
    k = 2 * np.pi / wavelength

    h1 = radar_height_m
    h2 = target_height_m
    d = horizontal_range_m

    # Prevent division by zero
    if d < 1.0:
        return 1.0 + 0j, h2 - h1, h2 - h1

    # Direct path
    R_direct = np.sqrt(d**2 + (h2 - h1)**2)

    # Reflected path (image method)
    R_reflected = np.sqrt(d**2 + (h2 + h1)**2)

    # Grazing angle at reflection point
    grazing_angle = np.arctan((h1 + h2) / d)

    # Fresnel reflection coefficient
    rho = fresnel_reflection_coefficient(grazing_angle, frequency_hz,
                                         surface, polarization)

    # Roughness reduction
    rho_s = roughness_factor(grazing_angle, frequency_hz, surface)

    # Divergence factor
    D = divergence_factor(grazing_angle, d/2)

    # Effective reflection coefficient
    Gamma = rho * rho_s * D

    # Path difference
    delta_R = R_reflected - R_direct

    # Phase difference
    delta_phi = k * delta_R

    # Amplitude ratio (approximately 1 for short ranges)
    amp_ratio = R_direct / R_reflected

    # Total field (direct + reflected)
    # F = 1 + Γ * exp(-jkΔR) * (Rd/Rr)
    multipath_factor = 1.0 + Gamma * np.exp(-1j * delta_phi) * amp_ratio

    return multipath_factor, R_direct, R_reflected


def compute_multipath_rays(radar_pos: np.ndarray,
                           target_pos: np.ndarray,
                           ground_height_m: float,
                           frequency_hz: float,
                           surface: SurfaceProperties = SURFACE_SEAWATER,
                           max_bounces: int = 1) -> List[dict]:
    """
    Compute all multipath ray paths between radar and target.

    Args:
        radar_pos: Radar position (x, y, z)
        target_pos: Target position (x, y, z)
        ground_height_m: Ground/water surface height
        frequency_hz: Frequency in Hz
        surface: Surface properties
        max_bounces: Maximum number of surface bounces (1 = two-ray)

    Returns:
        List of path dictionaries with:
        - path_length: Total path length in meters
        - amplitude: Complex amplitude factor
        - delay_s: Propagation delay
        - path_type: 'direct' or 'reflected_N'
    """
    c = 299792458.0
    paths = []

    # Direct path
    direct_vector = target_pos - radar_pos
    direct_length = np.linalg.norm(direct_vector)

    paths.append({
        'path_length': direct_length,
        'amplitude': 1.0 + 0j,
        'delay_s': direct_length / c,
        'path_type': 'direct'
    })

    # Single-bounce reflected path
    if max_bounces >= 1:
        h_radar = radar_pos[2] - ground_height_m
        h_target = target_pos[2] - ground_height_m

        if h_radar > 0 and h_target > 0:
            # Horizontal distance
            dx = target_pos[0] - radar_pos[0]
            dy = target_pos[1] - radar_pos[1]
            horizontal_range = np.sqrt(dx**2 + dy**2)

            # Image method for reflection point
            # Reflection point divides horizontal range proportionally
            reflection_fraction = h_radar / (h_radar + h_target)
            reflection_x = radar_pos[0] + dx * reflection_fraction
            reflection_y = radar_pos[1] + dy * reflection_fraction
            reflection_point = np.array([reflection_x, reflection_y, ground_height_m])

            # Path segments
            path_radar_to_ground = np.linalg.norm(reflection_point - radar_pos)
            path_ground_to_target = np.linalg.norm(target_pos - reflection_point)
            reflected_length = path_radar_to_ground + path_ground_to_target

            # Grazing angle
            grazing_angle = np.arctan(h_radar / (horizontal_range * reflection_fraction + 1e-6))

            # Reflection coefficient
            rho = fresnel_reflection_coefficient(grazing_angle, frequency_hz, surface, 'H')
            rho_rough = roughness_factor(grazing_angle, frequency_hz, surface)
            D = divergence_factor(grazing_angle, horizontal_range)

            Gamma = rho * rho_rough * D

            # Phase from path difference
            k = 2 * np.pi * frequency_hz / c
            phase = k * (reflected_length - direct_length)

            # Amplitude includes spreading loss difference
            amp = Gamma * (direct_length / reflected_length) * np.exp(-1j * phase)

            paths.append({
                'path_length': reflected_length,
                'amplitude': amp,
                'delay_s': reflected_length / c,
                'path_type': 'reflected_1',
                'reflection_point': reflection_point
            })

    # Double-bounce (radar-ground-target-ground-radar)
    if max_bounces >= 2:
        # This creates additional paths with more attenuation
        h_radar = radar_pos[2] - ground_height_m
        h_target = target_pos[2] - ground_height_m

        if h_radar > 0 and h_target > 0:
            dx = target_pos[0] - radar_pos[0]
            dy = target_pos[1] - radar_pos[1]
            horizontal_range = np.sqrt(dx**2 + dy**2)

            # Approximate double-bounce path length
            # Ground - Target (image) - Ground
            h_target_image = -h_target  # Image below ground
            double_bounce_length = 2 * np.sqrt(horizontal_range**2 + (h_radar + h_target)**2)

            grazing_angle = np.arctan((h_radar + h_target) / (horizontal_range + 1e-6))

            # Two reflections
            rho = fresnel_reflection_coefficient(grazing_angle, frequency_hz, surface, 'H')
            rho_rough = roughness_factor(grazing_angle, frequency_hz, surface)

            Gamma = (rho * rho_rough) ** 2  # Squared for double bounce

            k = 2 * np.pi * frequency_hz / c
            phase = k * (double_bounce_length - direct_length)

            amp = Gamma * (direct_length / double_bounce_length) * np.exp(-1j * phase)

            paths.append({
                'path_length': double_bounce_length,
                'amplitude': amp,
                'delay_s': double_bounce_length / c,
                'path_type': 'reflected_2'
            })

    return paths


def apply_multipath_to_signal(rx_signal: np.ndarray,
                              sample_rate_hz: float,
                              paths: List[dict],
                              reference_delay_s: float) -> np.ndarray:
    """
    Apply multipath effects to received signal.

    Adds delayed and scaled copies of the signal for each path.

    Args:
        rx_signal: Original received signal (direct path)
        sample_rate_hz: Sample rate in Hz
        paths: List of path dictionaries from compute_multipath_rays
        reference_delay_s: Reference delay (direct path)

    Returns:
        Signal with multipath components added
    """
    output = np.zeros_like(rx_signal, dtype=complex)
    n_samples = len(rx_signal)

    for path in paths:
        # Relative delay
        delay_s = path['delay_s'] - reference_delay_s
        delay_samples = int(delay_s * sample_rate_hz)

        # Apply delay and amplitude
        amp = path['amplitude']

        if delay_samples >= 0 and delay_samples < n_samples:
            # Shift and add
            valid_len = n_samples - delay_samples
            output[delay_samples:delay_samples + valid_len] += amp * rx_signal[:valid_len]
        elif delay_samples < 0 and abs(delay_samples) < n_samples:
            # Negative delay (path shorter than reference, shouldn't happen for multipath)
            start = abs(delay_samples)
            valid_len = n_samples - start
            output[:valid_len] += amp * rx_signal[start:start + valid_len]

    return output


def multipath_propagation_factor(radar_height_m: float,
                                  target_height_m: float,
                                  range_m: float,
                                  frequency_hz: float,
                                  surface: SurfaceProperties = SURFACE_SEAWATER) -> float:
    """
    Calculate one-way multipath propagation factor F.

    The received power is proportional to |F|⁴ for monostatic radar.

    Args:
        radar_height_m: Radar height above surface
        target_height_m: Target height above surface
        range_m: Horizontal range
        frequency_hz: Frequency

    Returns:
        Propagation factor magnitude |F|
    """
    F, _, _ = two_ray_multipath(radar_height_m, target_height_m, range_m,
                                 frequency_hz, surface)
    return np.abs(F)


def multipath_lobing_pattern(radar_height_m: float,
                              frequency_hz: float,
                              elevation_angles_deg: np.ndarray,
                              surface: SurfaceProperties = SURFACE_SEAWATER) -> np.ndarray:
    """
    Compute multipath lobing pattern vs elevation angle.

    Shows interference fringes due to surface reflection.

    Args:
        radar_height_m: Radar height
        frequency_hz: Frequency
        elevation_angles_deg: Array of elevation angles
        surface: Surface properties

    Returns:
        Propagation factor |F|² for each angle
    """
    c = 299792458.0
    wavelength = c / frequency_hz
    k = 2 * np.pi / wavelength

    el_rad = np.deg2rad(elevation_angles_deg)

    # Path difference for given elevation angle
    delta_R = 2 * radar_height_m * np.sin(el_rad)

    # Fresnel coefficient (use average grazing angle)
    rho_mag = np.abs(fresnel_reflection_coefficient(
        np.mean(el_rad), frequency_hz, surface, 'H'))

    # Propagation factor
    F_squared = 1 + rho_mag**2 + 2 * rho_mag * np.cos(k * delta_R)

    return F_squared
