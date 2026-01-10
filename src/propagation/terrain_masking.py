"""
Terrain Masking / Visibility Computation

Computes line-of-sight visibility from radar to all terrain points,
identifying shadow regions behind terrain features.
"""
import numpy as np
from typing import Tuple, Optional

# Try to import numba, fall back to no-op decorators
try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range


@jit(nopython=True, cache=True)
def _bilinear_interp(
    heightmap: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    x: float,
    y: float
) -> float:
    """Bilinear interpolation of heightmap at a point."""
    nx = len(x_coords)
    ny = len(y_coords)

    if nx < 2 or ny < 2:
        return 0.0

    dx = x_coords[1] - x_coords[0]
    dy = y_coords[1] - y_coords[0]

    # Normalized coordinates
    fi = (x - x_coords[0]) / dx
    fj = (y - y_coords[0]) / dy

    # Grid indices
    i0 = int(np.floor(fi))
    j0 = int(np.floor(fj))

    # Clamp to valid range
    i0 = max(0, min(i0, nx - 2))
    j0 = max(0, min(j0, ny - 2))
    i1 = i0 + 1
    j1 = j0 + 1

    # Fractional part
    fx = fi - i0
    fy = fj - j0
    fx = max(0.0, min(1.0, fx))
    fy = max(0.0, min(1.0, fy))

    # Bilinear interpolation
    h00 = heightmap[j0, i0]
    h10 = heightmap[j0, i1]
    h01 = heightmap[j1, i0]
    h11 = heightmap[j1, i1]

    h0 = h00 * (1 - fx) + h10 * fx
    h1 = h01 * (1 - fx) + h11 * fx

    return h0 * (1 - fy) + h1 * fy


@jit(nopython=True, cache=True)
def _check_los_blocked(
    radar_x: float,
    radar_y: float,
    radar_z: float,
    target_x: float,
    target_y: float,
    target_z: float,
    heightmap: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    n_steps: int
) -> bool:
    """Check if line-of-sight from radar to target is blocked by terrain.

    Uses ray marching through the heightmap.

    Returns:
        True if blocked, False if clear LOS
    """
    # Direction vector
    dx = target_x - radar_x
    dy = target_y - radar_y
    dz = target_z - radar_z

    dist = np.sqrt(dx*dx + dy*dy + dz*dz)
    if dist < 1.0:
        return False

    # March along ray
    for i in range(1, n_steps):
        t = i / n_steps

        # Position along ray
        x = radar_x + dx * t
        y = radar_y + dy * t
        z = radar_z + dz * t

        # Check bounds
        if x < x_coords[0] or x > x_coords[-1]:
            continue
        if y < y_coords[0] or y > y_coords[-1]:
            continue

        # Get terrain height at this point
        terrain_z = _bilinear_interp(heightmap, x_coords, y_coords, x, y)

        # Check if ray is below terrain
        if z < terrain_z:
            return True  # Blocked

    return False  # Clear LOS


@jit(nopython=True, parallel=True, cache=True)
def compute_visibility_mask(
    radar_position: np.ndarray,
    heightmap: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    n_steps: int = 100
) -> np.ndarray:
    """Compute visibility mask for entire terrain from radar viewpoint.

    Args:
        radar_position: [x, y, z] position of radar antenna
        heightmap: 2D elevation array [ny, nx]
        x_coords: X coordinate array
        y_coords: Y coordinate array
        n_steps: Number of steps for ray marching

    Returns:
        visibility_mask: Boolean array [ny, nx] - True = visible, False = shadowed
    """
    ny = len(y_coords)
    nx = len(x_coords)
    visibility = np.ones((ny, nx), dtype=np.bool_)

    radar_x = radar_position[0]
    radar_y = radar_position[1]
    radar_z = radar_position[2]

    for j in prange(ny):
        for i in range(nx):
            target_x = x_coords[i]
            target_y = y_coords[j]
            target_z = heightmap[j, i]

            # Check if terrain blocks LOS
            blocked = _check_los_blocked(
                radar_x, radar_y, radar_z,
                target_x, target_y, target_z,
                heightmap, x_coords, y_coords,
                n_steps
            )

            visibility[j, i] = not blocked

    return visibility


@jit(nopython=True, cache=True)
def _check_point_visible(
    radar_position: np.ndarray,
    point: np.ndarray,
    heightmap: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    n_steps: int
) -> bool:
    """Check if a single point is visible from radar."""
    blocked = _check_los_blocked(
        radar_position[0], radar_position[1], radar_position[2],
        point[0], point[1], point[2],
        heightmap, x_coords, y_coords,
        n_steps
    )
    return not blocked


def check_target_visible(
    radar_position: np.ndarray,
    target_position: Tuple[float, float, float],
    heightmap: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    n_steps: int = 100
) -> bool:
    """Check if a target is visible from radar.

    Args:
        radar_position: [x, y, z] radar position
        target_position: (x, y, z) target position
        heightmap: Terrain elevation array
        x_coords, y_coords: Coordinate arrays
        n_steps: Ray marching steps

    Returns:
        True if target is visible, False if blocked
    """
    point = np.array(target_position)
    return _check_point_visible(
        radar_position, point, heightmap, x_coords, y_coords, n_steps
    )


@jit(nopython=True, cache=True)
def compute_visibility_along_azimuth(
    radar_position: np.ndarray,
    azimuth_rad: float,
    ranges_m: np.ndarray,
    heightmap: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    n_steps: int = 50
) -> np.ndarray:
    """Compute visibility along a single azimuth direction.

    Args:
        radar_position: [x, y, z] radar position
        azimuth_rad: Azimuth angle in radians (0 = North, clockwise)
        ranges_m: Array of range values to check
        heightmap: Terrain elevation
        x_coords, y_coords: Coordinate arrays
        n_steps: Ray marching steps

    Returns:
        Boolean array - True where visible, False where shadowed
    """
    n_ranges = len(ranges_m)
    visibility = np.ones(n_ranges, dtype=np.bool_)

    # Direction unit vector (azimuth: 0=North=+Y, 90=East=+X)
    dir_x = np.sin(azimuth_rad)
    dir_y = np.cos(azimuth_rad)

    radar_x = radar_position[0]
    radar_y = radar_position[1]
    radar_z = radar_position[2]

    # Track maximum elevation angle seen so far (horizon tracking)
    max_elevation_angle = -np.inf

    for i in range(n_ranges):
        r = ranges_m[i]
        if r < 1.0:
            continue

        # Target position at this range (on ground)
        target_x = radar_x + dir_x * r
        target_y = radar_y + dir_y * r

        # Check bounds
        if target_x < x_coords[0] or target_x > x_coords[-1]:
            visibility[i] = True  # Out of terrain, assume visible
            continue
        if target_y < y_coords[0] or target_y > y_coords[-1]:
            visibility[i] = True
            continue

        # Get terrain height at target
        target_z = _bilinear_interp(heightmap, x_coords, y_coords, target_x, target_y)

        # Elevation angle to this point
        dz = target_z - radar_z
        elevation_angle = np.arctan2(dz, r)

        # If this point is below the horizon we've seen, it's shadowed
        if elevation_angle < max_elevation_angle:
            visibility[i] = False
        else:
            visibility[i] = True
            max_elevation_angle = elevation_angle

    return visibility


def compute_grazing_angle(
    radar_position: np.ndarray,
    heightmap: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray
) -> np.ndarray:
    """Compute grazing angle from radar to each terrain cell.

    Grazing angle is the angle between the radar beam and the local terrain surface.

    Args:
        radar_position: [x, y, z] radar position
        heightmap: Terrain elevation array [ny, nx]
        x_coords, y_coords: Coordinate arrays

    Returns:
        grazing_angle: Array of angles in radians [ny, nx]
    """
    ny, nx = heightmap.shape
    grazing = np.zeros((ny, nx))

    X, Y = np.meshgrid(x_coords, y_coords)

    # Vector from each point to radar
    dx = radar_position[0] - X
    dy = radar_position[1] - Y
    dz = radar_position[2] - heightmap

    # Distance to radar (horizontal)
    r_horiz = np.sqrt(dx**2 + dy**2)

    # Depression angle (angle below horizontal from radar to point)
    depression = np.arctan2(-dz, r_horiz)

    # For flat terrain, grazing angle = depression angle
    # For sloped terrain, need to account for surface normal
    # Simplified: assume grazing ≈ depression for now
    grazing = np.maximum(depression, 0.001)  # Clamp to positive

    return grazing


def compute_range_map(
    radar_position: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray
) -> np.ndarray:
    """Compute slant range from radar to each grid cell.

    Args:
        radar_position: [x, y, z] radar position
        x_coords, y_coords: Coordinate arrays

    Returns:
        range_map: Array of ranges in meters [ny, nx]
    """
    X, Y = np.meshgrid(x_coords, y_coords)

    dx = X - radar_position[0]
    dy = Y - radar_position[1]

    # Horizontal range (slant range approximately equals ground range for short ranges)
    range_map = np.sqrt(dx**2 + dy**2)

    return range_map


def compute_azimuth_map(
    radar_position: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray
) -> np.ndarray:
    """Compute azimuth angle from radar to each grid cell.

    Args:
        radar_position: [x, y, z] radar position
        x_coords, y_coords: Coordinate arrays

    Returns:
        azimuth_map: Array of azimuths in radians [ny, nx]
        (0 = North, increasing clockwise)
    """
    X, Y = np.meshgrid(x_coords, y_coords)

    dx = X - radar_position[0]
    dy = Y - radar_position[1]

    # atan2(x, y) gives azimuth from North
    azimuth_map = np.arctan2(dx, dy)

    # Convert to [0, 2pi]
    azimuth_map = np.mod(azimuth_map, 2 * np.pi)

    return azimuth_map
