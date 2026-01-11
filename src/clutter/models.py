"""
Clutter Models for Radar Simulation

Land and sea surface clutter using normalized RCS (sigma0) models.
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class LandClutterParams:
    """Parameters for land clutter model."""
    # Terrain type affects sigma0
    terrain_type: str = "mixed"  # "forest", "urban", "grass", "rock", "mixed"

    # Base sigma0 at reference grazing angle (dB)
    sigma0_ref_db: float = -15.0

    # Reference grazing angle (degrees)
    ref_grazing_deg: float = 10.0

    # Grazing angle exponent (gamma in sigma0 ~ sin^gamma(psi))
    grazing_exponent: float = 1.5

    # Variation (adds texture)
    variation_db: float = 3.0


@dataclass
class SeaClutterParams:
    """Parameters for sea clutter model (simplified Georgia Tech model)."""
    # Sea state (Douglas scale 0-9)
    sea_state: int = 3

    # Wind speed (m/s)
    wind_speed_mps: float = 10.0

    # Wind direction relative to radar look (degrees)
    # 0 = upwind, 90 = crosswind, 180 = downwind
    wind_direction_deg: float = 45.0


# Typical sigma0 values for different terrain types at X-band (dB)
TERRAIN_SIGMA0_DB = {
    "urban": -8.0,      # Buildings, structures
    "forest": -12.0,    # Trees, vegetation
    "grass": -18.0,     # Low vegetation
    "rock": -15.0,      # Bare rock, mountains
    "mixed": -15.0,     # Mixed terrain
    "water": -25.0,     # Calm water (use sea model for rough)
}


def compute_land_sigma0(
    grazing_angle_rad: float,
    params: LandClutterParams,
    frequency_hz: float = 9.5e9
) -> float:
    """Compute land clutter sigma0 at given grazing angle.

    Uses empirical model: sigma0 = sigma0_ref * sin^gamma(psi)

    Args:
        grazing_angle_rad: Grazing angle in radians
        params: Land clutter parameters
        frequency_hz: Radar frequency (for future frequency scaling)

    Returns:
        sigma0 in linear units (m²/m²)
    """
    # Get base sigma0 for terrain type
    sigma0_ref_db = TERRAIN_SIGMA0_DB.get(params.terrain_type, params.sigma0_ref_db)

    # Convert reference grazing angle
    ref_grazing_rad = np.radians(params.ref_grazing_deg)

    # Grazing angle dependence
    sin_psi = np.sin(np.maximum(grazing_angle_rad, 0.01))
    sin_ref = np.sin(ref_grazing_rad)

    # sigma0 ~ sin^gamma(psi)
    grazing_factor = (sin_psi / sin_ref) ** params.grazing_exponent

    # Convert to linear
    sigma0_ref_linear = 10 ** (sigma0_ref_db / 10)
    sigma0 = sigma0_ref_linear * grazing_factor

    return sigma0


def compute_sea_sigma0(
    grazing_angle_rad: float,
    params: SeaClutterParams,
    frequency_hz: float = 9.5e9
) -> float:
    """Compute sea clutter sigma0.

    Returns:
        sigma0 in linear units (m²/m²)
    """
    # No water reflections - perfectly calm/flat water returns nothing
    return 0.0


def compute_clutter_rcs(
    sigma0: float,
    cell_area_m2: float,
    visibility: bool = True
) -> float:
    """Convert sigma0 to clutter RCS for a resolution cell.

    RCS = sigma0 * illuminated_area

    Args:
        sigma0: Normalized RCS (m²/m²)
        cell_area_m2: Area of resolution cell
        visibility: Whether cell is visible (False = shadowed)

    Returns:
        Clutter RCS in m²
    """
    if not visibility:
        return 0.0

    return sigma0 * cell_area_m2


def generate_clutter_map(
    heightmap: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    radar_position: np.ndarray,
    water_mask: np.ndarray,
    visibility_mask: np.ndarray,
    land_params: LandClutterParams,
    sea_params: SeaClutterParams,
    frequency_hz: float = 9.5e9
) -> np.ndarray:
    """Generate clutter sigma0 map for entire scene.

    Args:
        heightmap: Terrain elevation array [ny, nx]
        x_coords, y_coords: Coordinate arrays
        radar_position: [x, y, z] radar position
        water_mask: Boolean array, True where water
        visibility_mask: Boolean array, True where visible
        land_params: Land clutter parameters
        sea_params: Sea clutter parameters
        frequency_hz: Radar frequency

    Returns:
        sigma0_map: Array of sigma0 values [ny, nx]
    """
    ny, nx = heightmap.shape
    sigma0_map = np.zeros((ny, nx))

    X, Y = np.meshgrid(x_coords, y_coords)

    # Compute grazing angles
    dx = radar_position[0] - X
    dy = radar_position[1] - Y
    dz = radar_position[2] - heightmap

    r_horiz = np.sqrt(dx**2 + dy**2)
    grazing = np.arctan2(dz, r_horiz)
    grazing = np.maximum(grazing, 0.001)  # Ensure positive

    # Compute sigma0 for each cell
    for j in range(ny):
        for i in range(nx):
            if not visibility_mask[j, i]:
                sigma0_map[j, i] = 0.0
                continue

            psi = grazing[j, i]

            if water_mask[j, i]:
                sigma0_map[j, i] = compute_sea_sigma0(psi, sea_params, frequency_hz)
            else:
                sigma0_map[j, i] = compute_land_sigma0(psi, land_params, frequency_hz)

    return sigma0_map


def add_clutter_variation(
    sigma0_map: np.ndarray,
    variation_db: float = 3.0,
    seed: int = None
) -> np.ndarray:
    """Add random variation to clutter map for realistic texture.

    Args:
        sigma0_map: Base sigma0 map
        variation_db: Standard deviation of variation in dB
        seed: Random seed

    Returns:
        Modified sigma0 map with variation
    """
    rng = np.random.default_rng(seed)

    # Log-normal variation
    sigma0_db = 10 * np.log10(sigma0_map + 1e-30)
    variation = rng.normal(0, variation_db, sigma0_map.shape)
    sigma0_db += variation

    return 10 ** (sigma0_db / 10)
