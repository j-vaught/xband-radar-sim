#!/usr/bin/env python3
"""
Magnetron Radar - Charleston Harbor Demo

Uses real elevation data from online APIs for Charleston, SC area.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import urllib.request
import json

from config_magnetron import MagnetronConfig
from terrain.coastal_scene import CoastalScene, CoastalSceneConfig, TargetInfo
from terrain.generator import compute_terrain_normals
from magnetron_pipeline import run_magnetron_simulation
from clutter.models import LandClutterParams, SeaClutterParams
from signal.ppi_processing import normalize_for_display


# Coastal locations - positioned for interesting radar views
LOCATIONS = {
    # San Diego Bay - good irregular coastline with hills
    'san_diego': {'lat': 32.7050, 'lon': -117.2300, 'name': 'San Diego Bay, CA'},
    # Seattle - dramatic terrain with mountains
    'seattle': {'lat': 47.6000, 'lon': -122.4000, 'name': 'Elliott Bay, Seattle'},
    # San Francisco - Golden Gate area
    'sf': {'lat': 37.8100, 'lon': -122.4500, 'name': 'San Francisco Bay, CA'},
    # Charleston - harbor mouth area
    'charleston': {'lat': 32.7450, 'lon': -79.8600, 'name': 'Charleston Harbor, SC'},
    # Chesapeake - irregular coastline
    'chesapeake': {'lat': 37.0000, 'lon': -76.0000, 'name': 'Chesapeake Bay, VA'},
    # Honolulu - volcanic terrain
    'honolulu': {'lat': 21.3000, 'lon': -157.8600, 'name': 'Honolulu Harbor, HI'},
}


def fetch_elevation_grid(
    center_lat: float,
    center_lon: float,
    radius_m: float = 2000.0,
    resolution_m: float = 10.0
) -> tuple:
    """Fetch real elevation data from Open-Elevation API.

    Args:
        center_lat: Center latitude
        center_lon: Center longitude
        radius_m: Scene radius in meters
        resolution_m: Grid resolution in meters

    Returns:
        Tuple of (heightmap, x_coords, y_coords)
    """
    from scipy.interpolate import RegularGridInterpolator
    from scipy.ndimage import gaussian_filter

    # Calculate lat/lon per meter
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * np.cos(np.radians(center_lat))

    # Grid size for output
    n_cells = int(2 * radius_m / resolution_m)

    # Create coordinate arrays
    x_coords = np.linspace(-radius_m, radius_m, n_cells)
    y_coords = np.linspace(-radius_m, radius_m, n_cells)

    # Convert to lat/lon grid
    lat_range = radius_m / m_per_deg_lat
    lon_range = radius_m / m_per_deg_lon

    lats_out = np.linspace(center_lat - lat_range, center_lat + lat_range, n_cells)
    lons_out = np.linspace(center_lon - lon_range, center_lon + lon_range, n_cells)

    print(f"  Target grid: {n_cells}x{n_cells} = {n_cells*n_cells} cells at {resolution_m}m resolution")
    print(f"  Area: {lats_out[0]:.4f} to {lats_out[-1]:.4f} lat, {lons_out[0]:.4f} to {lons_out[-1]:.4f} lon")

    # Sample from API - reduce to avoid timeouts
    # Open-Elevation API is slow, use fewer points with better interpolation
    n_sample = 60  # 60x60 = 3,600 points, faster API response

    sample_lats = np.linspace(center_lat - lat_range, center_lat + lat_range, n_sample)
    sample_lons = np.linspace(center_lon - lon_range, center_lon + lon_range, n_sample)

    # Batch size for API
    batch_size = 1000
    all_locations = []
    for lat in sample_lats:
        for lon in sample_lons:
            all_locations.append({"latitude": float(lat), "longitude": float(lon)})

    print(f"  Fetching {len(all_locations)} elevation points in {len(all_locations)//batch_size + 1} batches...")

    try:
        import time as time_module
        all_elevations = []
        url = "https://api.open-elevation.com/api/v1/lookup"

        for i in range(0, len(all_locations), batch_size):
            batch = all_locations[i:i+batch_size]
            data = json.dumps({"locations": batch}).encode('utf-8')
            req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})

            # Retry logic for API timeouts
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(req, timeout=60) as response:
                        result = json.loads(response.read().decode('utf-8'))
                    break
                except Exception as e:
                    if attempt < 2:
                        print(f"    Batch {i//batch_size + 1} attempt {attempt+1} failed, retrying...")
                        time_module.sleep(2)
                    else:
                        raise e

            batch_elevations = [r['elevation'] for r in result['results']]
            all_elevations.extend(batch_elevations)
            print(f"    Batch {i//batch_size + 1}: {len(batch_elevations)} points received")
            time_module.sleep(0.5)  # Rate limiting

        # Reshape to grid
        elev_sample = np.array(all_elevations).reshape(n_sample, n_sample)
        print(f"  Raw elevations: min={elev_sample.min():.1f}m, max={elev_sample.max():.1f}m")

        # Use proper interpolation to upsample to target resolution
        # Create interpolator
        interp = RegularGridInterpolator(
            (sample_lats, sample_lons),
            elev_sample,
            method='cubic',
            bounds_error=False,
            fill_value=0
        )

        # Create output grid
        LAT_out, LON_out = np.meshgrid(lats_out, lons_out, indexing='ij')
        points = np.stack([LAT_out.ravel(), LON_out.ravel()], axis=-1)

        heightmap = interp(points).reshape(n_cells, n_cells)

        # Light smoothing to remove any interpolation artifacts
        heightmap = gaussian_filter(heightmap, sigma=0.5)

        print(f"  Interpolated to {n_cells}x{n_cells} grid")

        return heightmap, x_coords, y_coords

    except Exception as e:
        print(f"  API request failed: {e}")
        print("  Falling back to synthetic terrain for Charleston...")
        return create_charleston_synthetic(radius_m, resolution_m)


def create_charleston_synthetic(radius_m: float = 2000.0, resolution_m: float = 10.0) -> tuple:
    """Create synthetic terrain based on Charleston Harbor geography.

    Radar is positioned in the harbor (center), looking at coastlines around it.
    Charleston is very flat coastal terrain with:
    - Harbor/water in CENTER (where radar is)
    - Low-lying land around edges (0-15m)
    - Natural curved coastlines
    """
    from scipy.ndimage import gaussian_filter

    n_cells = int(2 * radius_m / resolution_m)
    x_coords = np.linspace(-radius_m, radius_m, n_cells)
    y_coords = np.linspace(-radius_m, radius_m, n_cells)
    X, Y = np.meshgrid(x_coords, y_coords)

    rng = np.random.default_rng(42)

    # Distance from center (radar position)
    R = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)

    # Create natural coastline using perturbed circle
    # Base coastline at ~800-1200m from center, varying with angle
    coastline_base = 900
    coastline_variation = 300

    # Smooth random coastline shape
    n_harmonics = 8
    coastline_r = np.ones_like(theta) * coastline_base
    for k in range(1, n_harmonics + 1):
        amp = coastline_variation / k * rng.uniform(0.3, 1.0)
        phase = rng.uniform(0, 2*np.pi)
        coastline_r += amp * np.sin(k * theta + phase)

    # Water is inside the coastline
    water_mask = R < coastline_r

    # Create land elevation - rises from coastline
    dist_from_coast = R - coastline_r
    dist_from_coast = np.maximum(dist_from_coast, 0)

    # Gentle rise inland (Charleston is flat, max ~10-15m)
    heightmap = 3.0 + 8.0 * (1 - np.exp(-dist_from_coast / 500))

    # Add gentle terrain variation on land
    terrain_var = 2.0 * np.sin(X * 0.003 + 0.5) * np.cos(Y * 0.004)
    terrain_var += 1.5 * np.sin(X * 0.002 - Y * 0.003)
    heightmap += terrain_var * (1 - water_mask.astype(float))

    # Add small-scale texture
    noise = rng.standard_normal(heightmap.shape) * 1.0
    noise = gaussian_filter(noise, sigma=2)
    heightmap += noise * (1 - water_mask.astype(float))

    # Add a couple of islands in the harbor
    # Small island 1
    island1_x, island1_y = 400, 300
    island1_r = 80
    island1 = ((X - island1_x)**2 + (Y - island1_y)**2) < island1_r**2
    water_mask[island1] = False
    heightmap[island1] = 4.0 + rng.random(island1.sum()) * 2

    # Small island 2
    island2_x, island2_y = -300, -400
    island2_r = 60
    island2 = ((X - island2_x)**2 + (Y - island2_y)**2) < island2_r**2
    water_mask[island2] = False
    heightmap[island2] = 3.0 + rng.random(island2.sum()) * 2

    # Set water to 0
    heightmap[water_mask] = 0

    # Smooth the coastline transition
    heightmap = gaussian_filter(heightmap, sigma=1)
    heightmap[water_mask] = 0  # Re-zero water after smoothing

    # Ensure non-negative
    heightmap = np.maximum(heightmap, 0)

    return heightmap, x_coords, y_coords


def generate_harbor_boats(
    water_mask: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    n_boats: int = 10,
    seed: int = 42
) -> list:
    """Generate boats in harbor areas."""
    rng = np.random.default_rng(seed)
    boats = []

    # Find water cells
    water_points = []
    for j in range(len(y_coords)):
        for i in range(len(x_coords)):
            if water_mask[j, i]:
                x, y = x_coords[i], y_coords[j]
                r = np.sqrt(x**2 + y**2)
                if 100 < r < 1800:  # Valid range
                    water_points.append((x, y))

    if len(water_points) < n_boats:
        n_boats = len(water_points)

    if water_points:
        indices = rng.choice(len(water_points), size=n_boats, replace=False)

        # Mix of boat types
        boat_types = [
            ('Fishing', 8, 30),      # Small fishing boat
            ('Sailboat', 12, 50),    # Sailboat
            ('Cargo', 50, 500),      # Cargo ship
            ('Tanker', 100, 1000),   # Tanker
            ('Yacht', 20, 100),      # Yacht
            ('Tug', 15, 80),         # Tugboat
        ]

        for i, idx in enumerate(indices):
            x, y = water_points[idx]
            boat_type, size, rcs = boat_types[i % len(boat_types)]

            # Add some variation
            size *= rng.uniform(0.7, 1.3)
            rcs *= rng.uniform(0.5, 2.0)

            boats.append(TargetInfo(
                name=f"{boat_type}_{i+1}",
                position=(x, y, 2.0),
                rcs_m2=rcs,
                target_type="boat",
                size_m=size
            ))

    return boats


def run_charleston_demo(location_key: str = 'charleston', use_api: bool = True):
    """Run demo with Charleston Harbor terrain."""

    loc = LOCATIONS.get(location_key, LOCATIONS['charleston'])

    print("=" * 60)
    print(f"MAGNETRON RADAR - {loc['name'].upper()}")
    print("=" * 60)

    # Get terrain data
    print(f"\nFetching terrain data for {loc['name']}...")
    print(f"  Center: {loc['lat']:.4f}°N, {loc['lon']:.4f}°W")

    if use_api:
        try:
            heightmap, x_coords, y_coords = fetch_elevation_grid(
                loc['lat'], loc['lon'],
                radius_m=2000.0,
                resolution_m=10.0  # 10m resolution with cubic interpolation
            )
        except Exception as e:
            print(f"  API failed: {e}")
            print("  Using synthetic Charleston terrain...")
            heightmap, x_coords, y_coords = create_charleston_synthetic()
    else:
        heightmap, x_coords, y_coords = create_charleston_synthetic()

    print(f"  Terrain size: {heightmap.shape}")
    print(f"  Elevation range: {heightmap.min():.1f}m to {heightmap.max():.1f}m")

    # Create water mask (areas at or below sea level)
    water_mask = heightmap <= 0.5
    print(f"  Water coverage: {water_mask.sum() / water_mask.size * 100:.1f}%")

    # Ensure water is at 0
    heightmap[water_mask] = 0

    # Generate boats
    print("\nPlacing vessels in harbor...")
    boats = generate_harbor_boats(water_mask, x_coords, y_coords, n_boats=12, seed=789)

    for t in boats[:6]:  # Show first 6
        r = np.sqrt(t.position[0]**2 + t.position[1]**2)
        print(f"  {t.name}: range={r:.0f}m, RCS={t.rcs_m2:.0f}m²")
    if len(boats) > 6:
        print(f"  ... and {len(boats)-6} more vessels")

    # Compute normals
    normals = compute_terrain_normals(heightmap, x_coords, y_coords)

    # Create scene
    config = CoastalSceneConfig(
        scene_radius_m=x_coords[-1],
        resolution_m=x_coords[1] - x_coords[0],
        radar_height_m=15.0,  # Radar on elevated position
        shoreline_y_m=0.0,
        n_boats=len(boats),
        n_buoys=0,
        n_towers=0,
        seed=42
    )

    scene = CoastalScene(
        heightmap=heightmap,
        x_coords=x_coords,
        y_coords=y_coords,
        terrain_normals=normals,
        water_mask=water_mask,
        land_mask=~water_mask,
        targets=boats,
        config=config
    )

    # Radar config
    radar = MagnetronConfig(
        name="Harbor Surveillance Radar",
        center_frequency_hz=9.5e9,
        pulse_width_s=0.5e-6,
        max_range_m=2000.0,
        antenna_height_m=15.0
    )

    print(f"\nRadar: {radar.pulse_width_s*1e6:.1f}μs pulse, {radar.range_resolution_m:.0f}m resolution")

    # Run simulation
    print("\nRunning simulation...")
    result = run_magnetron_simulation(
        radar, scene,
        n_azimuths=360,
        n_range_bins=1024,
        land_params=LandClutterParams(terrain_type="urban"),  # Urban area
        sea_params=SeaClutterParams(sea_state=1, wind_speed_mps=3),  # Light harbor chop
        verbose=True
    )

    # Visualize
    print("\nGenerating visualization...")
    create_charleston_visualization(result, loc['name'])

    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)


def create_charleston_visualization(result, location_name: str):
    """Create visualization for Charleston demo."""
    scene = result.scene
    config = result.config
    heightmap = scene.heightmap
    water_mask = scene.water_mask

    fig = plt.figure(figsize=(14, 12))
    fig.suptitle(f'Magnetron Radar - {location_name}\n'
                 f'X-band, {config.pulse_width_s*1e6:.1f}μs pulse, {config.range_resolution_m:.0f}m resolution',
                 fontsize=13, fontweight='bold', color='white')

    extent = [scene.x_coords[0], scene.x_coords[-1],
              scene.y_coords[0], scene.y_coords[-1]]

    # Panel 1: Terrain view
    ax1 = fig.add_subplot(2, 2, 1)

    terrain_img = np.zeros((*heightmap.shape, 3))
    terrain_img[water_mask] = [0.1, 0.2, 0.5]  # Water = blue
    land = ~water_mask
    if heightmap.max() > 0:
        elev_norm = np.clip(heightmap / 15.0, 0, 1)  # Normalize to ~15m max
    else:
        elev_norm = np.zeros_like(heightmap)
    terrain_img[land, 0] = 0.3 + 0.4 * elev_norm[land]
    terrain_img[land, 1] = 0.5 - 0.2 * elev_norm[land]
    terrain_img[land, 2] = 0.2

    ax1.imshow(terrain_img, extent=extent, origin='lower')
    ax1.plot(0, 0, 'r^', markersize=10, label='Radar')
    for t in scene.targets:
        ax1.plot(t.position[0], t.position[1], 'yo', markersize=5)

    ax1.set_xlabel('East (m)', color='white')
    ax1.set_ylabel('North (m)', color='white')
    ax1.set_title('Harbor & Vessels', color='lime', fontsize=11)
    ax1.set_facecolor('#1a1a1a')
    ax1.tick_params(colors='white')
    ax1.set_aspect('equal')

    # Panel 2: Visibility
    ax2 = fig.add_subplot(2, 2, 2)

    vis_img = np.zeros((*heightmap.shape, 3))
    vis_img[water_mask] = [0.1, 0.2, 0.4]
    vis_img[result.visibility_mask & ~water_mask] = [0.3, 0.6, 0.3]
    vis_img[~result.visibility_mask & ~water_mask] = [0.5, 0.1, 0.1]

    ax2.imshow(vis_img, extent=extent, origin='lower')
    ax2.plot(0, 0, 'r^', markersize=10)

    ax2.set_xlabel('East (m)', color='white')
    ax2.set_ylabel('North (m)', color='white')
    ax2.set_title('Visibility (green=visible, red=shadow)', color='lime', fontsize=11)
    ax2.set_facecolor('#1a1a1a')
    ax2.tick_params(colors='white')
    ax2.set_aspect('equal')

    # Panel 3: PPI
    ax3 = fig.add_subplot(2, 2, 3)

    ppi = result.ppi_cartesian
    ppi_norm = normalize_for_display(ppi, dynamic_range_db=45)
    ppi_norm = np.clip((ppi_norm - 0.2) / 0.8, 0, 1) ** 2.0

    ppi_extent = [-config.max_range_m, config.max_range_m,
                  -config.max_range_m, config.max_range_m]

    ax3.imshow(ppi_norm, extent=ppi_extent, origin='lower', cmap='gray')

    for r in [500, 1000, 1500]:
        theta = np.linspace(0, 2*np.pi, 100)
        ax3.plot(r * np.cos(theta), r * np.sin(theta), 'g-', alpha=0.3, lw=0.5)

    for t in result.detected_targets:
        az_rad = np.radians(t['azimuth_deg'])
        x = t['range_m'] * np.sin(az_rad)
        y = t['range_m'] * np.cos(az_rad)
        ax3.plot(x, y, 'yo', markersize=4, markerfacecolor='none')

    ax3.set_xlabel('East (m)', color='white')
    ax3.set_ylabel('North (m)', color='white')
    ax3.set_title(f'PPI Display ({len(result.detected_targets)} targets)', color='lime', fontsize=11)
    ax3.set_facecolor('black')
    ax3.tick_params(colors='white')
    ax3.set_aspect('equal')

    # Panel 4: Elevation profile
    ax4 = fig.add_subplot(2, 2, 4)

    # Plot elevation along a few radials
    for az in [0, 45, 90, 135, 180, 225, 270, 315]:
        az_rad = np.radians(az)
        ranges = np.linspace(0, 2000, 200)
        elevations = []
        for r in ranges:
            x = r * np.sin(az_rad)
            y = r * np.cos(az_rad)
            # Find nearest cell
            i = np.argmin(np.abs(scene.x_coords - x))
            j = np.argmin(np.abs(scene.y_coords - y))
            if 0 <= i < len(scene.x_coords) and 0 <= j < len(scene.y_coords):
                elevations.append(heightmap[j, i])
            else:
                elevations.append(0)
        ax4.plot(ranges, elevations, alpha=0.5, lw=1, label=f'{az}°')

    ax4.set_xlabel('Range (m)', color='white')
    ax4.set_ylabel('Elevation (m)', color='white')
    ax4.set_title('Elevation Profiles (8 directions)', color='lime', fontsize=11)
    ax4.set_facecolor('#1a1a1a')
    ax4.tick_params(colors='white')
    ax4.set_xlim(0, 2000)
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=7, ncol=4, loc='upper right')

    fig.patch.set_facecolor('#1a1a1a')
    plt.tight_layout()

    filename = 'magnetron_charleston.png'
    fig.savefig(filename, dpi=150, facecolor='#1a1a1a', bbox_inches='tight')
    print(f"  Saved: {filename}")

    # Standalone PPI
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
    ax.set_title(f'Harbor Surveillance Radar - {location_name}', color='lime', fontsize=12)
    ax.set_facecolor('black')
    ax.tick_params(colors='white')
    ax.set_aspect('equal')

    fig2.patch.set_facecolor('black')
    plt.tight_layout()
    fig2.savefig('magnetron_charleston_ppi.png', dpi=150, facecolor='black', bbox_inches='tight')
    print("  Saved: magnetron_charleston_ppi.png")

    plt.close('all')


if __name__ == "__main__":
    # Try with API first, fall back to synthetic
    # San Diego has hills + irregular coastline - more interesting than flat Charleston
    run_charleston_demo('san_diego', use_api=True)
