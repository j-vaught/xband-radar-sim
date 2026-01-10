"""
Maritime Target Models

RCS models for boats, buoys, and shore towers.
"""
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class BoatTarget:
    """Small boat target model."""
    position: Tuple[float, float, float]  # (x, y, z) meters
    length_m: float = 8.0
    width_m: float = 2.5
    height_m: float = 2.0
    heading_deg: float = 0.0  # 0 = bow pointing North

    # Hull material affects RCS
    is_fiberglass: bool = True  # Fiberglass has lower RCS than metal

    def get_rcs(
        self,
        aspect_angle_deg: float,
        frequency_hz: float = 9.5e9
    ) -> float:
        """Compute RCS based on aspect angle.

        Boat RCS varies significantly with aspect:
        - Broadside: Maximum RCS (flat sides)
        - Bow/stern: Lower RCS (pointed)

        Args:
            aspect_angle_deg: Angle from bow (0 = bow-on, 90 = broadside)
            frequency_hz: Radar frequency

        Returns:
            RCS in m²
        """
        return compute_boat_rcs(
            self.length_m, self.width_m, self.height_m,
            aspect_angle_deg, self.is_fiberglass, frequency_hz
        )

    @property
    def name(self) -> str:
        return f"Boat_{self.length_m:.0f}m"


@dataclass
class BuoyTarget:
    """Navigation buoy target."""
    position: Tuple[float, float, float]  # (x, y, z) meters
    diameter_m: float = 0.6
    height_m: float = 1.5
    has_radar_reflector: bool = True  # Radar reflector for visibility

    def get_rcs(self, frequency_hz: float = 9.5e9) -> float:
        """Compute buoy RCS.

        Buoys with radar reflectors have much higher RCS.

        Args:
            frequency_hz: Radar frequency

        Returns:
            RCS in m²
        """
        return compute_buoy_rcs(
            self.diameter_m, self.height_m,
            self.has_radar_reflector, frequency_hz
        )

    @property
    def name(self) -> str:
        return f"Buoy_{self.diameter_m:.1f}m"


@dataclass
class TowerTarget:
    """Shore tower/mast target."""
    position: Tuple[float, float, float]  # (x, y, z) - center of tower
    height_m: float = 30.0
    width_m: float = 3.0

    # Tower type affects scattering
    tower_type: str = "lattice"  # "lattice", "cylinder", "building"

    def get_rcs(
        self,
        elevation_angle_deg: float,
        frequency_hz: float = 9.5e9
    ) -> float:
        """Compute tower RCS.

        Tower RCS dominated by:
        - Dihedral from base (ground-tower corner)
        - Structural returns

        Args:
            elevation_angle_deg: Elevation angle from radar
            frequency_hz: Radar frequency

        Returns:
            RCS in m²
        """
        return compute_tower_rcs(
            self.height_m, self.width_m,
            self.tower_type, elevation_angle_deg, frequency_hz
        )

    @property
    def name(self) -> str:
        return f"Tower_{self.height_m:.0f}m"


def compute_boat_rcs(
    length_m: float,
    width_m: float,
    height_m: float,
    aspect_angle_deg: float,
    is_fiberglass: bool = True,
    frequency_hz: float = 9.5e9
) -> float:
    """Compute boat RCS using empirical model.

    Based on measurements of small craft, RCS varies with aspect:
    - Broadside (90°): ~L * H (flat plate approximation)
    - Bow/stern (0°/180°): Much smaller, ~0.1-0.3 of broadside

    Args:
        length_m: Boat length
        width_m: Boat width (beam)
        height_m: Boat height above water
        aspect_angle_deg: Aspect angle (0 = bow-on)
        is_fiberglass: True for fiberglass (lower RCS)
        frequency_hz: Radar frequency

    Returns:
        RCS in m²
    """
    wavelength = 3e8 / frequency_hz

    # Normalize aspect to 0-90 (use symmetry)
    aspect = np.abs(aspect_angle_deg) % 180
    if aspect > 90:
        aspect = 180 - aspect

    aspect_rad = np.radians(aspect)

    # Base areas for different aspects
    broadside_area = length_m * height_m  # Side view
    bow_area = width_m * height_m * 0.3    # Bow view (reduced due to shape)

    # Interpolate between bow and broadside
    # Using sin^2 for smooth transition
    broadside_factor = np.sin(aspect_rad) ** 2
    effective_area = bow_area + (broadside_area - bow_area) * broadside_factor

    # Convert area to RCS (flat plate RCS = 4*pi*A^2/lambda^2)
    # But boats are not flat plates, use empirical factor
    rcs = effective_area * 0.5  # Empirical factor

    # Material factor (fiberglass reflects less than metal)
    if is_fiberglass:
        rcs *= 0.3

    # Frequency dependence (larger targets less frequency dependent)
    # Small boats: RCS roughly constant in X-band

    return max(rcs, 0.1)  # Minimum RCS


def compute_buoy_rcs(
    diameter_m: float,
    height_m: float,
    has_radar_reflector: bool = True,
    frequency_hz: float = 9.5e9
) -> float:
    """Compute buoy RCS.

    Buoys with radar reflectors (corner reflectors) have very high RCS
    for their size - typically 5-20 m² for navigation buoys.

    Args:
        diameter_m: Buoy diameter
        height_m: Buoy height above water
        has_radar_reflector: Whether buoy has radar reflector
        frequency_hz: Radar frequency

    Returns:
        RCS in m²
    """
    wavelength = 3e8 / frequency_hz

    if has_radar_reflector:
        # Typical marine radar reflector: ~10 m² RCS
        # Scales with reflector size (usually matched to buoy)
        reflector_size = max(diameter_m * 0.5, 0.2)  # meters
        # Corner reflector RCS = 4*pi*a^4/(3*lambda^2)
        rcs = 4 * np.pi * reflector_size**4 / (3 * wavelength**2)
        rcs = min(rcs, 20.0)  # Cap at reasonable value
    else:
        # Cylinder RCS (approximate)
        # RCS ~ 2*pi*r*L^2/lambda for broadside cylinder
        radius = diameter_m / 2
        rcs = 2 * np.pi * radius * height_m**2 / wavelength
        rcs = min(rcs, 1.0)  # Small without reflector

    return max(rcs, 0.01)


def compute_tower_rcs(
    height_m: float,
    width_m: float,
    tower_type: str = "lattice",
    elevation_angle_deg: float = 5.0,
    frequency_hz: float = 9.5e9
) -> float:
    """Compute tower RCS.

    Tower RCS comes from:
    1. Direct returns from structure
    2. Dihedral returns (ground-structure corner reflection)

    Args:
        height_m: Tower height
        width_m: Tower width
        tower_type: "lattice", "cylinder", or "building"
        elevation_angle_deg: Elevation angle from radar
        frequency_hz: Radar frequency

    Returns:
        RCS in m²
    """
    wavelength = 3e8 / frequency_hz
    elevation_rad = np.radians(np.abs(elevation_angle_deg))

    # Base RCS depends on type
    if tower_type == "lattice":
        # Lattice towers have many small scatterers
        # RCS ~ number of elements * element RCS
        # Typical: 10-50 m² for communication towers
        base_rcs = height_m * width_m * 0.5
    elif tower_type == "cylinder":
        # Cylindrical tower (e.g., smokestack)
        radius = width_m / 2
        base_rcs = 2 * np.pi * radius * height_m**2 / wavelength
    else:  # building
        # Building face
        base_rcs = width_m * height_m

    # Dihedral enhancement at low elevation angles
    # Ground-tower corner acts as dihedral reflector
    if elevation_angle_deg < 10:
        dihedral_factor = 1 + 2 * np.cos(elevation_rad)
    else:
        dihedral_factor = 1.0

    rcs = base_rcs * dihedral_factor

    return max(rcs, 1.0)


def compute_target_return_power(
    rcs_m2: float,
    range_m: float,
    peak_power_w: float,
    antenna_gain: float,
    wavelength_m: float,
    losses_linear: float = 2.0
) -> float:
    """Compute received power from target using radar equation.

    Args:
        rcs_m2: Target RCS
        range_m: Target range
        peak_power_w: Peak transmit power
        antenna_gain: Antenna gain (linear)
        wavelength_m: Wavelength
        losses_linear: System losses (linear)

    Returns:
        Received power in watts
    """
    # Handle both scalar and array inputs
    range_m = np.atleast_1d(np.asarray(range_m))
    rcs_m2 = np.atleast_1d(np.asarray(rcs_m2))

    # Clamp minimum range
    range_m = np.maximum(range_m, 1.0)

    numerator = peak_power_w * antenna_gain**2 * wavelength_m**2 * rcs_m2
    denominator = (4 * np.pi)**3 * range_m**4 * losses_linear

    result = numerator / denominator

    # Return scalar if inputs were scalar
    if result.size == 1:
        return float(result[0])
    return result


def aspect_angle_from_positions(
    radar_pos: Tuple[float, float, float],
    target_pos: Tuple[float, float, float],
    target_heading_deg: float
) -> float:
    """Compute aspect angle from radar to target.

    Args:
        radar_pos: (x, y, z) radar position
        target_pos: (x, y, z) target position
        target_heading_deg: Target heading (0 = North)

    Returns:
        Aspect angle in degrees (0 = bow-on, 90 = broadside)
    """
    # Vector from target to radar
    dx = radar_pos[0] - target_pos[0]
    dy = radar_pos[1] - target_pos[1]

    # Bearing from target to radar
    bearing_to_radar = np.degrees(np.arctan2(dx, dy))  # 0 = North

    # Aspect angle relative to target heading
    aspect = bearing_to_radar - target_heading_deg

    # Normalize to -180 to 180
    while aspect > 180:
        aspect -= 360
    while aspect < -180:
        aspect += 360

    return aspect
