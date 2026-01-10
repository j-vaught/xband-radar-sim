"""
Procedural Terrain Generation

Generates heightmaps for coastal/mountain scenarios using
multi-octave noise functions.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional

# Try to import numba, fall back to no-op decorators
try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # No-op decorators
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range


@dataclass
class TerrainParams:
    """Parameters for procedural terrain generation."""
    # Height range
    max_elevation_m: float = 150.0  # Peak mountain height
    min_elevation_m: float = 0.0    # Water level

    # Noise parameters
    noise_octaves: int = 5
    noise_persistence: float = 0.5  # Amplitude decay per octave
    noise_lacunarity: float = 2.0   # Frequency increase per octave
    base_scale: float = 400.0       # Meters per base noise period

    # Ridge features (for mountain ridges)
    ridge_weight: float = 0.4       # Blend factor for ridge noise
    ridge_sharpness: float = 2.0    # Higher = sharper ridges

    # Shoreline
    shoreline_distance_m: float = 600.0  # Distance from radar to shoreline
    transition_width_m: float = 50.0     # Width of shore transition

    # Random seed
    seed: int = 42


@jit(nopython=True, cache=True)
def _hash_2d(x: int, y: int, seed: int) -> float:
    """Simple hash function for pseudo-random values."""
    n = x + y * 57 + seed * 131
    n = (n << 13) ^ n
    return (1.0 - ((n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff) / 1073741824.0)


@jit(nopython=True, cache=True)
def _smoothstep(t: float) -> float:
    """Smooth interpolation curve."""
    return t * t * (3 - 2 * t)


@jit(nopython=True, cache=True)
def _lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation."""
    return a + t * (b - a)


@jit(nopython=True, cache=True)
def _value_noise_2d(x: float, y: float, seed: int) -> float:
    """2D value noise at a point."""
    # Grid cell coordinates
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = x0 + 1
    y1 = y0 + 1

    # Fractional position within cell
    fx = x - x0
    fy = y - y0

    # Smooth interpolation weights
    sx = _smoothstep(fx)
    sy = _smoothstep(fy)

    # Corner values
    n00 = _hash_2d(x0, y0, seed)
    n10 = _hash_2d(x1, y0, seed)
    n01 = _hash_2d(x0, y1, seed)
    n11 = _hash_2d(x1, y1, seed)

    # Bilinear interpolation
    nx0 = _lerp(n00, n10, sx)
    nx1 = _lerp(n01, n11, sx)
    return _lerp(nx0, nx1, sy)


@jit(nopython=True, cache=True)
def _fbm_noise(x: float, y: float, octaves: int, persistence: float,
               lacunarity: float, seed: int) -> float:
    """Fractal Brownian Motion (multi-octave) noise."""
    value = 0.0
    amplitude = 1.0
    frequency = 1.0
    max_value = 0.0

    for _ in range(octaves):
        value += amplitude * _value_noise_2d(x * frequency, y * frequency, seed)
        max_value += amplitude
        amplitude *= persistence
        frequency *= lacunarity

    return value / max_value


@jit(nopython=True, cache=True)
def _ridge_noise(x: float, y: float, octaves: int, persistence: float,
                 lacunarity: float, sharpness: float, seed: int) -> float:
    """Ridge noise for mountain ridges."""
    value = 0.0
    amplitude = 1.0
    frequency = 1.0
    max_value = 0.0

    for i in range(octaves):
        n = _value_noise_2d(x * frequency, y * frequency, seed + i * 17)
        # Create ridges by folding noise around 0
        n = 1.0 - abs(n)
        n = n ** sharpness
        value += amplitude * n
        max_value += amplitude
        amplitude *= persistence
        frequency *= lacunarity

    return value / max_value


@jit(nopython=True, parallel=True, cache=True)
def _generate_heightmap_core(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    base_scale: float,
    octaves: int,
    persistence: float,
    lacunarity: float,
    ridge_weight: float,
    ridge_sharpness: float,
    seed: int
) -> np.ndarray:
    """Core heightmap generation (Numba-accelerated)."""
    ny = len(y_coords)
    nx = len(x_coords)
    heightmap = np.zeros((ny, nx), dtype=np.float64)

    for j in prange(ny):
        for i in range(nx):
            x_scaled = x_coords[i] / base_scale
            y_scaled = y_coords[j] / base_scale

            # Base terrain from FBM
            fbm = _fbm_noise(x_scaled, y_scaled, octaves, persistence,
                            lacunarity, seed)

            # Ridge features for mountains
            ridge = _ridge_noise(x_scaled * 0.7, y_scaled * 0.7, octaves - 1,
                                persistence, lacunarity, ridge_sharpness, seed + 100)

            # Blend FBM and ridge noise
            heightmap[j, i] = (1 - ridge_weight) * fbm + ridge_weight * ridge

    return heightmap


def generate_heightmap(
    x_extent_m: float,
    y_extent_m: float,
    resolution_m: float,
    params: TerrainParams
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate procedural terrain heightmap.

    Args:
        x_extent_m: Total extent in X direction (east-west) in meters
        y_extent_m: Total extent in Y direction (north-south) in meters
        resolution_m: Grid cell size in meters
        params: Terrain generation parameters

    Returns:
        Tuple of (heightmap, x_coords, y_coords)
        - heightmap: 2D array of elevation values [ny, nx]
        - x_coords: 1D array of x coordinates
        - y_coords: 1D array of y coordinates
    """
    # Create coordinate arrays (centered on radar at origin)
    x_coords = np.arange(-x_extent_m/2, x_extent_m/2, resolution_m)
    y_coords = np.arange(-y_extent_m/2, y_extent_m/2, resolution_m)

    # Generate raw heightmap [0, 1]
    raw_heightmap = _generate_heightmap_core(
        x_coords, y_coords,
        params.base_scale,
        params.noise_octaves,
        params.noise_persistence,
        params.noise_lacunarity,
        params.ridge_weight,
        params.ridge_sharpness,
        params.seed
    )

    # Scale to elevation range
    heightmap = params.min_elevation_m + raw_heightmap * (
        params.max_elevation_m - params.min_elevation_m
    )

    return heightmap, x_coords, y_coords


def apply_shoreline_mask(
    heightmap: np.ndarray,
    y_coords: np.ndarray,
    shoreline_y: float,
    transition_width_m: float = 50.0,
    water_level_m: float = 0.0
) -> np.ndarray:
    """Apply shoreline mask to blend terrain into water.

    Everything south of shoreline (y < shoreline_y) becomes water.
    Creates smooth transition at the shoreline.

    Args:
        heightmap: Input elevation map
        y_coords: Y coordinate array
        shoreline_y: Y position of shoreline
        transition_width_m: Width of smooth transition
        water_level_m: Water surface elevation

    Returns:
        Modified heightmap with water areas
    """
    masked = heightmap.copy()

    for j, y in enumerate(y_coords):
        if y < shoreline_y - transition_width_m:
            # Fully in water
            masked[j, :] = water_level_m
        elif y < shoreline_y + transition_width_m:
            # Transition zone
            t = (y - (shoreline_y - transition_width_m)) / (2 * transition_width_m)
            t = np.clip(t, 0, 1)
            # Smooth step
            blend = t * t * (3 - 2 * t)
            masked[j, :] = water_level_m * (1 - blend) + heightmap[j, :] * blend

    return masked


def heightmap_to_mesh(
    heightmap: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert heightmap to triangle mesh.

    Args:
        heightmap: 2D elevation array [ny, nx]
        x_coords: X coordinate array
        y_coords: Y coordinate array

    Returns:
        Tuple of (vertices, faces)
        - vertices: Nx3 array of vertex positions
        - faces: Mx3 array of triangle face indices
    """
    ny, nx = heightmap.shape

    # Create vertex grid
    X, Y = np.meshgrid(x_coords, y_coords)
    vertices = np.stack([X.ravel(), Y.ravel(), heightmap.ravel()], axis=1)

    # Create triangle faces (two triangles per grid cell)
    faces = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            # Vertex indices for this cell
            v00 = j * nx + i
            v10 = j * nx + (i + 1)
            v01 = (j + 1) * nx + i
            v11 = (j + 1) * nx + (i + 1)

            # Two triangles
            faces.append([v00, v10, v01])
            faces.append([v10, v11, v01])

    return vertices.astype(np.float32), np.array(faces, dtype=np.int32)


def compute_terrain_normals(
    heightmap: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray
) -> np.ndarray:
    """Compute surface normals from heightmap.

    Args:
        heightmap: 2D elevation array
        x_coords: X coordinate array
        y_coords: Y coordinate array

    Returns:
        Normal vectors [ny, nx, 3]
    """
    dx = x_coords[1] - x_coords[0] if len(x_coords) > 1 else 1.0
    dy = y_coords[1] - y_coords[0] if len(y_coords) > 1 else 1.0

    # Compute gradients
    dz_dx = np.gradient(heightmap, dx, axis=1)
    dz_dy = np.gradient(heightmap, dy, axis=0)

    # Normal = (-dz/dx, -dz/dy, 1) normalized
    normals = np.stack([-dz_dx, -dz_dy, np.ones_like(heightmap)], axis=-1)

    # Normalize
    norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = normals / (norms + 1e-10)

    return normals
