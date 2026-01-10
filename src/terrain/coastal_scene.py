"""
Coastal Scene Builder

Combines terrain, water, and targets into a complete radar scenario.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from .generator import (
    TerrainParams,
    generate_heightmap,
    heightmap_to_mesh,
    apply_shoreline_mask,
    compute_terrain_normals,
)


@dataclass
class TargetInfo:
    """Information about a discrete target in the scene."""
    name: str
    position: Tuple[float, float, float]  # (x, y, z) in meters
    rcs_m2: float
    target_type: str  # "boat", "buoy", "tower"
    size_m: float = 5.0  # Approximate size for visualization


@dataclass
class CoastalSceneConfig:
    """Configuration for coastal radar scenario."""
    # Scene extents (radar at origin)
    scene_radius_m: float = 2000.0
    resolution_m: float = 5.0  # Grid resolution

    # Radar position
    radar_height_m: float = 10.0  # Antenna height above water

    # Shoreline position (y coordinate where water meets land)
    shoreline_y_m: float = 600.0

    # Target counts
    n_boats: int = 5
    n_buoys: int = 8
    n_towers: int = 2

    # Terrain parameters
    terrain_params: TerrainParams = field(default_factory=TerrainParams)

    # Random seed for reproducibility
    seed: int = 42

    def __post_init__(self):
        # Update terrain params with scene shoreline
        self.terrain_params.shoreline_distance_m = self.shoreline_y_m


@dataclass
class CoastalScene:
    """Complete coastal scene for radar simulation."""
    # Terrain data
    heightmap: np.ndarray          # [ny, nx] elevation values
    x_coords: np.ndarray           # [nx] x positions
    y_coords: np.ndarray           # [ny] y positions
    terrain_normals: np.ndarray    # [ny, nx, 3] surface normals

    # Masks
    water_mask: np.ndarray         # [ny, nx] True where water
    land_mask: np.ndarray          # [ny, nx] True where land

    # Targets
    targets: List[TargetInfo]

    # Configuration
    config: CoastalSceneConfig

    # Optional mesh (for ray tracing)
    terrain_vertices: Optional[np.ndarray] = None
    terrain_faces: Optional[np.ndarray] = None

    @property
    def radar_position(self) -> np.ndarray:
        """Radar antenna position."""
        return np.array([0.0, 0.0, self.config.radar_height_m])

    def get_height_at(self, x: float, y: float) -> float:
        """Get terrain height at a point (bilinear interpolation)."""
        # Find grid indices
        dx = self.x_coords[1] - self.x_coords[0]
        dy = self.y_coords[1] - self.y_coords[0]

        i = (x - self.x_coords[0]) / dx
        j = (y - self.y_coords[0]) / dy

        i0 = int(np.floor(i))
        j0 = int(np.floor(j))
        i1 = min(i0 + 1, len(self.x_coords) - 1)
        j1 = min(j0 + 1, len(self.y_coords) - 1)
        i0 = max(0, i0)
        j0 = max(0, j0)

        fi = i - i0
        fj = j - j0

        # Bilinear interpolation
        h00 = self.heightmap[j0, i0]
        h10 = self.heightmap[j0, i1]
        h01 = self.heightmap[j1, i0]
        h11 = self.heightmap[j1, i1]

        h0 = h00 * (1 - fi) + h10 * fi
        h1 = h01 * (1 - fi) + h11 * fi

        return h0 * (1 - fj) + h1 * fj

    def is_water(self, x: float, y: float) -> bool:
        """Check if point is over water."""
        return y < self.config.shoreline_y_m

    def get_targets_in_range(self, max_range_m: float) -> List[TargetInfo]:
        """Get all targets within specified range of radar."""
        result = []
        for target in self.targets:
            dist = np.sqrt(target.position[0]**2 + target.position[1]**2)
            if dist <= max_range_m:
                result.append(target)
        return result


def _generate_boat_positions(
    n_boats: int,
    shoreline_y: float,
    scene_radius: float,
    rng: np.random.Generator
) -> List[TargetInfo]:
    """Generate random boat positions on water."""
    boats = []

    for i in range(n_boats):
        # Boats are in water (y < shoreline)
        # But not too close to radar
        min_range = 100.0
        max_y = shoreline_y - 50.0  # Stay away from shore

        angle = rng.uniform(0, 2 * np.pi)
        distance = rng.uniform(min_range, min(scene_radius * 0.8, 1500))

        x = distance * np.sin(angle)
        y_offset = distance * np.cos(angle)

        # Ensure in water zone
        if y_offset > 0:
            # Boat would be north of radar, make it south
            y = -abs(y_offset)
        else:
            y = y_offset

        # Clamp to water area
        y = min(y, max_y)

        # Boat RCS varies with size
        size = rng.uniform(5.0, 15.0)
        rcs = size * size * rng.uniform(0.5, 2.0)  # Rough scaling

        boats.append(TargetInfo(
            name=f"Boat_{i+1}",
            position=(x, y, 2.0),  # Boats float at ~2m above water
            rcs_m2=rcs,
            target_type="boat",
            size_m=size
        ))

    return boats


def _generate_buoy_positions(
    n_buoys: int,
    shoreline_y: float,
    scene_radius: float,
    rng: np.random.Generator
) -> List[TargetInfo]:
    """Generate buoy positions (navigation markers)."""
    buoys = []

    for i in range(n_buoys):
        # Buoys in water, often in lines or channels
        min_range = 50.0

        angle = rng.uniform(0, 2 * np.pi)
        distance = rng.uniform(min_range, min(scene_radius * 0.6, 1000))

        x = distance * np.sin(angle)
        y = -abs(distance * np.cos(angle))  # Keep in water (y < 0 area)

        # Clamp to water
        y = min(y, shoreline_y - 30)

        # Buoys with radar reflectors have high RCS for size
        has_reflector = rng.random() > 0.3
        rcs = 10.0 if has_reflector else 1.0

        buoys.append(TargetInfo(
            name=f"Buoy_{i+1}",
            position=(x, y, 1.0),  # Buoys at ~1m
            rcs_m2=rcs,
            target_type="buoy",
            size_m=1.0
        ))

    return buoys


def _generate_tower_positions(
    n_towers: int,
    heightmap: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    shoreline_y: float,
    scene_radius: float,
    rng: np.random.Generator
) -> List[TargetInfo]:
    """Generate tower positions on high ground."""
    towers = []

    # Find high points on land for towers
    land_mask = y_coords > shoreline_y + 50

    if not np.any(land_mask):
        return towers

    # Get land portion of heightmap
    land_y_indices = np.where(land_mask)[0]

    for i in range(n_towers):
        # Pick a random high point
        attempts = 0
        while attempts < 100:
            j_idx = rng.choice(land_y_indices)
            i_idx = rng.integers(0, len(x_coords))

            x = x_coords[i_idx]
            y = y_coords[j_idx]
            ground_height = heightmap[j_idx, i_idx]

            # Check it's elevated and in range
            dist = np.sqrt(x**2 + y**2)
            if dist < scene_radius and ground_height > 20:
                break
            attempts += 1

        if attempts >= 100:
            continue

        tower_height = rng.uniform(20, 50)
        z = ground_height + tower_height / 2  # Tower center

        # Tower RCS (mainly corner reflector effect from structure)
        rcs = tower_height * 5.0  # Rough estimate

        towers.append(TargetInfo(
            name=f"Tower_{i+1}",
            position=(x, y, z),
            rcs_m2=rcs,
            target_type="tower",
            size_m=tower_height
        ))

    return towers


def build_coastal_scene(config: CoastalSceneConfig) -> CoastalScene:
    """Build complete coastal scene with terrain and targets.

    Args:
        config: Scene configuration

    Returns:
        CoastalScene with all components
    """
    rng = np.random.default_rng(config.seed)

    # Generate base terrain
    extent = config.scene_radius_m * 2
    heightmap, x_coords, y_coords = generate_heightmap(
        x_extent_m=extent,
        y_extent_m=extent,
        resolution_m=config.resolution_m,
        params=config.terrain_params
    )

    # Apply shoreline mask
    heightmap = apply_shoreline_mask(
        heightmap,
        y_coords,
        shoreline_y=config.shoreline_y_m,
        transition_width_m=config.terrain_params.transition_width_m,
        water_level_m=0.0
    )

    # Compute surface normals
    normals = compute_terrain_normals(heightmap, x_coords, y_coords)

    # Create water/land masks
    Y_grid = np.broadcast_to(y_coords[:, np.newaxis], heightmap.shape)
    water_mask = Y_grid < config.shoreline_y_m
    land_mask = ~water_mask

    # Generate mesh for ray tracing (optional)
    vertices, faces = heightmap_to_mesh(heightmap, x_coords, y_coords)

    # Generate targets
    targets = []

    # Boats on water
    targets.extend(_generate_boat_positions(
        config.n_boats, config.shoreline_y_m, config.scene_radius_m, rng
    ))

    # Buoys on water
    targets.extend(_generate_buoy_positions(
        config.n_buoys, config.shoreline_y_m, config.scene_radius_m, rng
    ))

    # Towers on land
    targets.extend(_generate_tower_positions(
        config.n_towers, heightmap, x_coords, y_coords,
        config.shoreline_y_m, config.scene_radius_m, rng
    ))

    return CoastalScene(
        heightmap=heightmap,
        x_coords=x_coords,
        y_coords=y_coords,
        terrain_normals=normals,
        water_mask=water_mask,
        land_mask=land_mask,
        targets=targets,
        config=config,
        terrain_vertices=vertices,
        terrain_faces=faces
    )


def create_simple_coastal_scene(
    max_range_m: float = 2000.0,
    shoreline_distance_m: float = 600.0,
    n_boats: int = 5,
    n_buoys: int = 8,
    seed: int = 42
) -> CoastalScene:
    """Convenience function to create a standard coastal scene.

    Args:
        max_range_m: Maximum radar range
        shoreline_distance_m: Distance from radar to shoreline
        n_boats: Number of boats
        n_buoys: Number of buoys
        seed: Random seed

    Returns:
        CoastalScene ready for simulation
    """
    terrain_params = TerrainParams(
        max_elevation_m=150.0,
        min_elevation_m=0.0,
        noise_octaves=5,
        base_scale=300.0,
        ridge_weight=0.5,
        shoreline_distance_m=shoreline_distance_m,
        seed=seed
    )

    config = CoastalSceneConfig(
        scene_radius_m=max_range_m,
        resolution_m=5.0,
        radar_height_m=10.0,
        shoreline_y_m=shoreline_distance_m,
        n_boats=n_boats,
        n_buoys=n_buoys,
        n_towers=2,
        terrain_params=terrain_params,
        seed=seed
    )

    return build_coastal_scene(config)
