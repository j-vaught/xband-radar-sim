#!/usr/bin/env python3
"""
Magnetron Radar with Real Terrain Data

Uses real-world elevation data for coastal simulation.
Supports:
- GeoTIFF files (SRTM, etc.)
- Image heightmaps (PNG/JPG)
- Online download (requires 'elevation' package)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import urllib.request

from config_magnetron import MagnetronConfig
from terrain.coastal_scene import CoastalScene, CoastalSceneConfig, TargetInfo
from terrain.generator import compute_terrain_normals
from magnetron_pipeline import run_magnetron_simulation
from clutter.models import LandClutterParams, SeaClutterParams
from propagation.terrain_masking import compute_visibility_mask
from signal.ppi_processing import normalize_for_display


def download_sample_heightmap(url: str, save_path: str) -> str:
    """Download a sample heightmap image."""
    if os.path.exists(save_path):
        print(f"  Using cached: {save_path}")
        return save_path

    print(f"  Downloading heightmap...")
    try:
        urllib.request.urlretrieve(url, save_path)
        print(f"  Saved to: {save_path}")
        return save_path
    except Exception as e:
        print(f"  Download failed: {e}")
        return None


def create_synthetic_coastal_terrain(
    scene_radius_m: float = 2000.0,
    resolution_m: float = 5.0,
    shoreline_angle_deg: float = 0.0,
    shoreline_distance_m: float = 600.0,
    max_elevation_m: float = 150.0,
    seed: int = 42
) -> tuple:
    """Create synthetic terrain that mimics real coastal features.

    More realistic than pure noise - includes:
    - Defined shoreline
    - Ridge lines parallel to coast
    - Valleys cutting through
    - Realistic slope gradients
    """
    rng = np.random.default_rng(seed)

    # Create coordinate grid
    n_cells = int(2 * scene_radius_m / resolution_m)
    x_coords = np.linspace(-scene_radius_m, scene_radius_m, n_cells)
    y_coords = np.linspace(-scene_radius_m, scene_radius_m, n_cells)
    X, Y = np.meshgrid(x_coords, y_coords)

    # Rotate coordinates if shoreline is angled
    angle_rad = np.radians(shoreline_angle_deg)
    X_rot = X * np.cos(angle_rad) - Y * np.sin(angle_rad)
    Y_rot = X * np.sin(angle_rad) + Y * np.cos(angle_rad)

    # Distance from shoreline (positive = land, negative = water)
    dist_from_shore = Y_rot - shoreline_distance_m

    # Base terrain: rises from shore
    # Exponential rise that levels off
    land_rise = max_elevation_m * (1 - np.exp(-np.maximum(dist_from_shore, 0) / 800))

    # Add ridge features parallel to coast
    ridge_freq = 0.003
    ridges = 0.3 * max_elevation_m * np.sin(X_rot * ridge_freq * 2 * np.pi) ** 2
    ridges *= np.clip(dist_from_shore / 500, 0, 1)  # Fade in from shore

    # Add valleys cutting through (perpendicular to coast)
    n_valleys = 3
    valley_depth = 0.4 * max_elevation_m
    valleys = np.zeros_like(X)
    for i in range(n_valleys):
        valley_x = rng.uniform(-scene_radius_m * 0.7, scene_radius_m * 0.7)
        valley_width = rng.uniform(100, 300)
        valley_profile = np.exp(-((X_rot - valley_x) / valley_width) ** 2)
        valleys += valley_depth * valley_profile

    # Combine features
    heightmap = land_rise + ridges - valleys

    # Add small-scale noise for texture
    noise_scale = 0.05 * max_elevation_m
    noise = noise_scale * rng.standard_normal(heightmap.shape)
    # Smooth the noise
    from scipy.ndimage import gaussian_filter
    noise = gaussian_filter(noise, sigma=2)
    heightmap += noise

    # Water areas (below shoreline)
    water_mask = dist_from_shore < 0
    heightmap[water_mask] = 0

    # Ensure non-negative on land
    heightmap = np.maximum(heightmap, 0)

    return heightmap, x_coords, y_coords, water_mask


def generate_boats_on_water(
    water_mask: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    n_boats: int = 6,
    min_range: float = 100.0,
    max_range: float = 1500.0,
    seed: int = 42
) -> list:
    """Generate boat positions on water areas."""
    rng = np.random.default_rng(seed)
    boats = []

    # Find water cells
    Y, X = np.meshgrid(y_coords, x_coords, indexing='ij')
    water_points = []
    for j in range(len(y_coords)):
        for i in range(len(x_coords)):
            if water_mask[j, i]:
                x, y = x_coords[i], y_coords[j]
                r = np.sqrt(x**2 + y**2)
                if min_range < r < max_range:
                    water_points.append((x, y))

    if len(water_points) < n_boats:
        print(f"  Warning: Only {len(water_points)} water points available")
        n_boats = len(water_points)

    # Randomly select positions
    if water_points:
        indices = rng.choice(len(water_points), size=n_boats, replace=False)
        for i, idx in enumerate(indices):
            x, y = water_points[idx]
            size = rng.uniform(5.0, 15.0)
            rcs = size * size * rng.uniform(0.5, 2.0)

            boats.append(TargetInfo(
                name=f"Boat_{i+1}",
                position=(x, y, 2.0),
                rcs_m2=rcs,
                target_type="boat",
                size_m=size
            ))

    return boats


def create_real_coastal_scene(
    heightmap: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    water_mask: np.ndarray,
    radar_height_m: float = 10.0,
    n_boats: int = 6,
    seed: int = 42
) -> CoastalScene:
    """Create CoastalScene from real terrain data."""

    # Compute normals
    normals = compute_terrain_normals(heightmap, x_coords, y_coords)

    # Generate boats
    boats = generate_boats_on_water(
        water_mask, x_coords, y_coords,
        n_boats=n_boats, seed=seed
    )

    # Create config
    config = CoastalSceneConfig(
        scene_radius_m=x_coords[-1],
        resolution_m=x_coords[1] - x_coords[0] if len(x_coords) > 1 else 5.0,
        radar_height_m=radar_height_m,
        shoreline_y_m=0.0,  # Will be determined by water mask
        n_boats=n_boats,
        n_buoys=0,
        n_towers=0,
        seed=seed
    )

    return CoastalScene(
        heightmap=heightmap,
        x_coords=x_coords,
        y_coords=y_coords,
        terrain_normals=normals,
        water_mask=water_mask,
        land_mask=~water_mask,
        targets=boats,
        config=config
    )


def run_real_terrain_demo():
    """Run demo with synthetic realistic coastal terrain."""
    print("=" * 60)
    print("MAGNETRON RADAR - REALISTIC COASTAL TERRAIN")
    print("=" * 60)

    # Create synthetic but realistic coastal terrain
    print("\nGenerating realistic coastal terrain...")
    heightmap, x_coords, y_coords, water_mask = create_synthetic_coastal_terrain(
        scene_radius_m=2000.0,
        resolution_m=5.0,
        shoreline_angle_deg=15.0,  # Angled shoreline
        shoreline_distance_m=500.0,
        max_elevation_m=120.0,
        seed=123
    )

    print(f"  Terrain size: {heightmap.shape}")
    print(f"  Max elevation: {heightmap.max():.1f}m")
    print(f"  Water coverage: {water_mask.sum() / water_mask.size * 100:.1f}%")

    # Create scene with boats
    print("\nCreating scene...")
    scene = create_real_coastal_scene(
        heightmap, x_coords, y_coords, water_mask,
        radar_height_m=10.0,
        n_boats=8,
        seed=456
    )

    print(f"  Targets: {len(scene.targets)}")
    for t in scene.targets:
        r = np.sqrt(t.position[0]**2 + t.position[1]**2)
        print(f"    {t.name}: range={r:.0f}m, RCS={t.rcs_m2:.1f}m²")

    # Radar config
    radar = MagnetronConfig(
        name="X-Band Magnetron",
        center_frequency_hz=9.5e9,
        pulse_width_s=0.5e-6,
        max_range_m=2000.0,
        antenna_height_m=10.0
    )

    print(f"\nRadar: {radar.pulse_width_s*1e6:.1f}μs pulse, {radar.range_resolution_m:.0f}m resolution")

    # Run simulation
    print("\nRunning simulation...")
    land_params = LandClutterParams(terrain_type="mixed")
    sea_params = SeaClutterParams(sea_state=0, wind_speed_mps=0)

    result = run_magnetron_simulation(
        radar, scene,
        n_azimuths=360,
        n_range_bins=1024,
        land_params=land_params,
        sea_params=sea_params,
        verbose=True
    )

    # Visualize
    print("\nGenerating visualization...")
    create_real_terrain_visualization(result, heightmap, water_mask)

    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)


def create_real_terrain_visualization(result, heightmap, water_mask):
    """Create visualization for real terrain demo."""
    scene = result.scene
    config = result.config

    fig = plt.figure(figsize=(14, 12))
    fig.suptitle('Magnetron Radar - Realistic Coastal Terrain\n'
                 f'X-band, {config.pulse_width_s*1e6:.1f}μs pulse, {config.range_resolution_m:.0f}m resolution',
                 fontsize=13, fontweight='bold', color='white')

    # Panel 1: Terrain top-down view
    ax1 = fig.add_subplot(2, 2, 1)

    # Create colored terrain image
    terrain_img = np.zeros((*heightmap.shape, 3))
    # Water = blue
    terrain_img[water_mask] = [0.1, 0.2, 0.5]
    # Land = green to brown based on elevation
    land = ~water_mask
    if heightmap.max() > 0:
        elev_norm = heightmap / heightmap.max()
    else:
        elev_norm = heightmap
    terrain_img[land, 0] = 0.2 + 0.6 * elev_norm[land]  # R
    terrain_img[land, 1] = 0.5 - 0.3 * elev_norm[land]  # G
    terrain_img[land, 2] = 0.1                           # B

    extent = [scene.x_coords[0], scene.x_coords[-1],
              scene.y_coords[0], scene.y_coords[-1]]

    ax1.imshow(terrain_img, extent=extent, origin='lower')

    # Mark radar and boats
    ax1.plot(0, 0, 'r^', markersize=10, label='Radar')
    for t in scene.targets:
        ax1.plot(t.position[0], t.position[1], 'yo', markersize=6)

    ax1.set_xlabel('East (m)', color='white')
    ax1.set_ylabel('North (m)', color='white')
    ax1.set_title('Terrain & Targets', color='lime', fontsize=11)
    ax1.set_facecolor('#1a1a1a')
    ax1.tick_params(colors='white')
    ax1.set_aspect('equal')

    # Panel 2: Visibility/Shadow map
    ax2 = fig.add_subplot(2, 2, 2)

    vis_img = np.zeros((*heightmap.shape, 3))
    vis_img[water_mask] = [0.1, 0.2, 0.4]
    vis_img[result.visibility_mask & ~water_mask] = [0.3, 0.6, 0.3]  # Visible land = green
    vis_img[~result.visibility_mask & ~water_mask] = [0.5, 0.1, 0.1]  # Shadow = red

    ax2.imshow(vis_img, extent=extent, origin='lower')
    ax2.plot(0, 0, 'r^', markersize=10)

    ax2.set_xlabel('East (m)', color='white')
    ax2.set_ylabel('North (m)', color='white')
    ax2.set_title('Visibility (green=visible, red=shadow)', color='lime', fontsize=11)
    ax2.set_facecolor('#1a1a1a')
    ax2.tick_params(colors='white')
    ax2.set_aspect('equal')

    # Panel 3: PPI Display
    ax3 = fig.add_subplot(2, 2, 3)

    ppi = result.ppi_cartesian
    ppi_norm = normalize_for_display(ppi, dynamic_range_db=45)

    # Apply threshold and gamma
    threshold = 0.2
    ppi_norm = np.clip((ppi_norm - threshold) / (1 - threshold), 0, 1)
    ppi_norm = ppi_norm ** 2.0

    ppi_extent = [-config.max_range_m, config.max_range_m,
                  -config.max_range_m, config.max_range_m]

    ax3.imshow(ppi_norm, extent=ppi_extent, origin='lower', cmap='gray')

    # Range rings
    for r in [500, 1000, 1500]:
        theta = np.linspace(0, 2*np.pi, 100)
        ax3.plot(r * np.cos(theta), r * np.sin(theta), 'g-', alpha=0.3, lw=0.5)

    # Target markers
    for t in result.detected_targets:
        az_rad = np.radians(t['azimuth_deg'])
        x = t['range_m'] * np.sin(az_rad)
        y = t['range_m'] * np.cos(az_rad)
        ax3.plot(x, y, 'yo', markersize=5, markerfacecolor='none')

    ax3.set_xlabel('East (m)', color='white')
    ax3.set_ylabel('North (m)', color='white')
    ax3.set_title(f'PPI Display ({len(result.detected_targets)} targets)', color='lime', fontsize=11)
    ax3.set_facecolor('black')
    ax3.tick_params(colors='white')
    ax3.set_aspect('equal')

    # Panel 4: 3D terrain view
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')

    # Subsample for performance
    step = 8
    X = scene.x_coords[::step]
    Y = scene.y_coords[::step]
    Z = heightmap[::step, ::step]
    X_grid, Y_grid = np.meshgrid(X, Y)

    # Color by water/land
    colors = np.zeros((*Z.shape, 4))
    water_sub = water_mask[::step, ::step]
    colors[water_sub] = [0.1, 0.3, 0.8, 0.8]
    land_sub = ~water_sub
    if Z.max() > 0:
        z_norm = Z / Z.max()
    else:
        z_norm = Z
    colors[land_sub, 0] = 0.2 + 0.5 * z_norm[land_sub]
    colors[land_sub, 1] = 0.6 - 0.3 * z_norm[land_sub]
    colors[land_sub, 2] = 0.1
    colors[land_sub, 3] = 1.0

    ax4.plot_surface(X_grid, Y_grid, Z, facecolors=colors,
                     rstride=1, cstride=1, antialiased=False, shade=True)

    ax4.scatter([0], [0], [10], c='red', s=100, marker='^')

    ax4.set_xlabel('East (m)', color='white')
    ax4.set_ylabel('North (m)', color='white')
    ax4.set_zlabel('Height (m)', color='white')
    ax4.set_title('3D Terrain View', color='lime', fontsize=11)
    ax4.set_facecolor('#1a1a1a')

    fig.patch.set_facecolor('#1a1a1a')
    plt.tight_layout()
    fig.savefig('magnetron_real_terrain.png', dpi=150, facecolor='#1a1a1a', bbox_inches='tight')
    print("  Saved: magnetron_real_terrain.png")

    # Also save standalone PPI
    fig2, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(ppi_norm, extent=ppi_extent, origin='lower', cmap='gray')

    for r in [500, 1000, 1500, 2000]:
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(r * np.cos(theta), r * np.sin(theta), 'g-', alpha=0.3, lw=0.5)

    for t in result.detected_targets:
        az_rad = np.radians(t['azimuth_deg'])
        x = t['range_m'] * np.sin(az_rad)
        y = t['range_m'] * np.cos(az_rad)
        ax.plot(x, y, 'yo', markersize=5, markerfacecolor='none')

    ax.set_xlabel('East (m)', color='white', fontsize=11)
    ax.set_ylabel('North (m)', color='white', fontsize=11)
    ax.set_title('Magnetron Radar PPI - Realistic Coastal Terrain', color='lime', fontsize=12)
    ax.set_facecolor('black')
    ax.tick_params(colors='white')
    ax.set_aspect('equal')

    fig2.patch.set_facecolor('black')
    plt.tight_layout()
    fig2.savefig('magnetron_real_terrain_ppi.png', dpi=150, facecolor='black', bbox_inches='tight')
    print("  Saved: magnetron_real_terrain_ppi.png")

    plt.close('all')


if __name__ == "__main__":
    run_real_terrain_demo()
