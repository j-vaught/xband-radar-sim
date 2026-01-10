"""Atmospheric propagation effects for radar simulation.

Implements:
- Atmospheric refraction (ray bending due to refractive index gradient)
- Atmospheric attenuation (absorption by oxygen, water vapor)
- Standard atmosphere model (pressure, temperature, humidity vs altitude)

References:
- ITU-R P.676-12: Attenuation by atmospheric gases
- ITU-R P.453-14: The radio refractive index
- Bean & Dutton (1966): Radio Meteorology
"""
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class AtmosphereConfig:
    """Configuration for atmospheric model.

    Attributes:
        temperature_k: Surface temperature in Kelvin (default 288.15 K = 15°C)
        pressure_hpa: Surface pressure in hPa (default 1013.25 hPa)
        relative_humidity: Relative humidity 0-1 (default 0.6 = 60%)
        enable_refraction: Enable ray bending due to atmosphere
        enable_attenuation: Enable atmospheric absorption
    """
    temperature_k: float = 288.15  # 15°C standard
    pressure_hpa: float = 1013.25  # Standard sea level
    relative_humidity: float = 0.6  # 60%
    enable_refraction: bool = True
    enable_attenuation: bool = True


def standard_atmosphere(altitude_m: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    US Standard Atmosphere 1976 model.

    Args:
        altitude_m: Altitude(s) in meters

    Returns:
        temperature_k: Temperature in Kelvin
        pressure_hpa: Pressure in hPa
        density_kg_m3: Air density in kg/m³
    """
    altitude_m = np.atleast_1d(altitude_m)

    # Constants
    T0 = 288.15  # Sea level temperature (K)
    P0 = 1013.25  # Sea level pressure (hPa)
    L = 0.0065   # Temperature lapse rate (K/m) for troposphere
    g = 9.80665  # Gravity (m/s²)
    M = 0.0289644  # Molar mass of air (kg/mol)
    R = 8.31447   # Gas constant (J/(mol·K))

    # Troposphere (0-11 km)
    temperature_k = T0 - L * altitude_m

    # Pressure from barometric formula
    exponent = g * M / (R * L)
    pressure_hpa = P0 * (temperature_k / T0) ** exponent

    # Density from ideal gas law
    density_kg_m3 = pressure_hpa * 100 * M / (R * temperature_k)

    # Clip to reasonable values for troposphere
    temperature_k = np.clip(temperature_k, 200, 320)
    pressure_hpa = np.clip(pressure_hpa, 100, 1100)

    return temperature_k, pressure_hpa, density_kg_m3


def water_vapor_density(temperature_k: float, pressure_hpa: float,
                        relative_humidity: float) -> float:
    """
    Calculate water vapor density using Magnus formula.

    Args:
        temperature_k: Temperature in Kelvin
        pressure_hpa: Pressure in hPa
        relative_humidity: Relative humidity (0-1)

    Returns:
        Water vapor density in g/m³
    """
    T_c = temperature_k - 273.15  # Convert to Celsius

    # Saturation vapor pressure (Magnus formula)
    es = 6.112 * np.exp(17.67 * T_c / (T_c + 243.5))  # hPa

    # Actual vapor pressure
    e = relative_humidity * es

    # Water vapor density (g/m³)
    rho_w = 216.7 * e / temperature_k

    return rho_w


def radio_refractive_index(temperature_k: float, pressure_hpa: float,
                           water_vapor_density_g_m3: float) -> float:
    """
    Calculate radio refractive index N = (n - 1) × 10⁶.

    Uses ITU-R P.453 formula:
    N = 77.6 * P/T + 3.73e5 * e/T²

    Args:
        temperature_k: Temperature in Kelvin
        pressure_hpa: Total pressure in hPa
        water_vapor_density_g_m3: Water vapor density

    Returns:
        N-units (refractive index - 1) × 10⁶
    """
    T = temperature_k
    P = pressure_hpa

    # Water vapor partial pressure from density
    e = water_vapor_density_g_m3 * T / 216.7

    # Dry term + wet term
    N = 77.6 * P / T - 5.6 * e / T + 3.75e5 * e / (T * T)

    return N


def refractive_index_gradient(config: AtmosphereConfig,
                              altitude_m: float = 0.0) -> float:
    """
    Calculate vertical gradient of refractive index dN/dh.

    Standard atmosphere: dN/dh ≈ -40 N-units/km

    Args:
        config: Atmosphere configuration
        altitude_m: Altitude in meters

    Returns:
        dN/dh in N-units per meter
    """
    # Get atmosphere at two heights
    h1 = altitude_m
    h2 = altitude_m + 100.0  # 100m increment

    T1, P1, _ = standard_atmosphere(h1)
    T2, P2, _ = standard_atmosphere(h2)

    rho_w1 = water_vapor_density(float(T1), float(P1), config.relative_humidity)
    rho_w2 = water_vapor_density(float(T2), float(P2), config.relative_humidity)

    N1 = radio_refractive_index(float(T1), float(P1), rho_w1)
    N2 = radio_refractive_index(float(T2), float(P2), rho_w2)

    dN_dh = (N2 - N1) / 100.0  # N-units per meter

    return dN_dh


def effective_earth_radius(config: AtmosphereConfig) -> float:
    """
    Calculate effective Earth radius accounting for atmospheric refraction.

    k = 1 / (1 + a * dN/dh)

    where a = 6371 km (Earth radius), dN/dh in N-units/km

    Standard atmosphere: k ≈ 4/3 (effective radius = 8500 km)

    Args:
        config: Atmosphere configuration

    Returns:
        Effective Earth radius in meters
    """
    EARTH_RADIUS_M = 6.371e6

    if not config.enable_refraction:
        return EARTH_RADIUS_M

    dN_dh = refractive_index_gradient(config)  # N-units/m
    dN_dh_km = dN_dh * 1000  # N-units/km

    # k factor
    k = 1.0 / (1.0 + EARTH_RADIUS_M * dN_dh_km * 1e-6 / 1000)

    # Clip to reasonable range
    k = np.clip(k, 0.5, 2.5)

    return k * EARTH_RADIUS_M


def atmospheric_attenuation_db_per_km(frequency_hz: float,
                                       config: AtmosphereConfig,
                                       altitude_m: float = 0.0) -> float:
    """
    Calculate atmospheric attenuation using simplified ITU-R P.676 model.

    Includes oxygen and water vapor absorption.

    Args:
        frequency_hz: Frequency in Hz
        config: Atmosphere configuration
        altitude_m: Altitude in meters

    Returns:
        Attenuation in dB/km
    """
    if not config.enable_attenuation:
        return 0.0

    f_ghz = frequency_hz / 1e9

    # Get atmospheric parameters
    T, P, _ = standard_atmosphere(altitude_m)
    T = float(T)
    P = float(P)

    rho_w = water_vapor_density(T, P, config.relative_humidity)

    # Simplified model for X-band (8-12 GHz)
    # Oxygen absorption (relatively constant in X-band)
    gamma_o = 0.008 * (P / 1013.25) * (288.15 / T) ** 0.5

    # Water vapor absorption
    # Resonance at 22 GHz, but tail affects X-band
    f_r = 22.235  # GHz, water vapor resonance
    delta_f = 3.0  # GHz, line width (approximate)

    # Line shape factor
    F = f_ghz / f_r * ((delta_f ** 2) / ((f_ghz - f_r) ** 2 + delta_f ** 2) +
                       (delta_f ** 2) / ((f_ghz + f_r) ** 2 + delta_f ** 2))

    # Water vapor attenuation
    gamma_w = 0.05 * rho_w * F * (300 / T) ** 2.5

    # Total attenuation
    gamma_total = gamma_o + gamma_w

    return gamma_total


def apply_atmospheric_attenuation(power_w: np.ndarray,
                                   range_m: np.ndarray,
                                   frequency_hz: float,
                                   config: AtmosphereConfig) -> np.ndarray:
    """
    Apply atmospheric attenuation to signal power.

    Args:
        power_w: Power values in Watts
        range_m: One-way range in meters
        frequency_hz: Operating frequency
        config: Atmosphere configuration

    Returns:
        Attenuated power values
    """
    if not config.enable_attenuation:
        return power_w

    # Get attenuation coefficient (average over path)
    gamma_db_km = atmospheric_attenuation_db_per_km(frequency_hz, config)

    # Two-way path
    total_path_km = 2 * range_m / 1000.0

    # Total attenuation in dB
    attenuation_db = gamma_db_km * total_path_km

    # Convert to linear
    attenuation_linear = 10 ** (-attenuation_db / 10)

    return power_w * attenuation_linear


def refract_ray_direction(direction: np.ndarray,
                          origin_altitude_m: float,
                          config: AtmosphereConfig) -> np.ndarray:
    """
    Apply atmospheric refraction to ray direction.

    Rays bend downward in standard atmosphere due to decreasing refractive index.

    Args:
        direction: Unit direction vector (3,)
        origin_altitude_m: Ray origin altitude
        config: Atmosphere configuration

    Returns:
        Refracted direction vector
    """
    if not config.enable_refraction:
        return direction

    # Effective Earth radius accounts for refraction
    # For ray tracing, we can approximate refraction by curving the Earth
    # or by adjusting ray elevation angle

    # Get refractive index gradient
    dN_dh = refractive_index_gradient(config, origin_altitude_m)

    # For near-horizontal rays, angular deviation per km
    # δθ/δr ≈ -dN/dh × 10⁻⁶
    angular_rate = -dN_dh * 1e-6  # rad/m

    # Extract elevation angle
    el = np.arcsin(np.clip(direction[2], -1, 1))
    az = np.arctan2(direction[1], direction[0])

    # For simulation, apply a small downward tilt based on typical path
    # This is a simplification - full ray tracing would integrate along path
    typical_range = 1000.0  # meters, for initial correction
    delta_el = angular_rate * typical_range * np.cos(el)

    # Apply correction (limited magnitude)
    delta_el = np.clip(delta_el, -0.01, 0.01)  # ~0.5 degrees max
    el_new = el + delta_el

    # Reconstruct direction
    direction_refracted = np.array([
        np.cos(el_new) * np.cos(az),
        np.cos(el_new) * np.sin(az),
        np.sin(el_new)
    ])

    return direction_refracted / np.linalg.norm(direction_refracted)


def compute_radar_horizon(radar_height_m: float,
                          target_height_m: float,
                          config: AtmosphereConfig) -> float:
    """
    Compute radar horizon distance accounting for refraction.

    d = sqrt(2 * k * Re * h1) + sqrt(2 * k * Re * h2)

    where k is the effective Earth radius factor.

    Args:
        radar_height_m: Radar antenna height
        target_height_m: Target height
        config: Atmosphere configuration

    Returns:
        Horizon distance in meters
    """
    Re = effective_earth_radius(config)

    d1 = np.sqrt(2 * Re * radar_height_m)
    d2 = np.sqrt(2 * Re * target_height_m)

    return d1 + d2


def is_below_horizon(radar_pos: np.ndarray,
                     target_pos: np.ndarray,
                     config: AtmosphereConfig) -> bool:
    """
    Check if target is below radar horizon.

    Args:
        radar_pos: Radar position (x, y, z) in meters
        target_pos: Target position (x, y, z) in meters
        config: Atmosphere configuration

    Returns:
        True if target is below horizon
    """
    # Horizontal distance
    dx = target_pos[0] - radar_pos[0]
    dy = target_pos[1] - radar_pos[1]
    horizontal_range = np.sqrt(dx**2 + dy**2)

    # Horizon distance
    horizon = compute_radar_horizon(radar_pos[2], target_pos[2], config)

    return horizontal_range > horizon
