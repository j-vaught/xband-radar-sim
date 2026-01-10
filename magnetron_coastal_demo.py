#!/usr/bin/env python3
"""
Magnetron Radar Coastal/Lake Simulation Demo

Demonstrates magnetron radar simulation with:
- Water area with boats and buoys
- Shoreline transition
- Mountains/valleys behind shore
- Terrain masking (radar shadows)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D

from config_magnetron import MagnetronConfig
from terrain.generator import TerrainParams
from terrain.coastal_scene import CoastalSceneConfig, build_coastal_scene
from magnetron_pipeline import run_magnetron_simulation
from clutter.models import LandClutterParams, SeaClutterParams
from signal.ppi_processing import normalize_for_display


def create_demo_scene():
    """Create a coastal scene for demonstration."""
    print("Creating coastal scene...")

    terrain_params = TerrainParams(
        max_elevation_m=120.0,       # Mountain peaks
        min_elevation_m=0.0,
        noise_octaves=5,
        noise_persistence=0.5,
        base_scale=250.0,            # Terrain feature scale
        ridge_weight=0.5,            # Ridge features for mountains
        ridge_sharpness=2.0,
        shoreline_distance_m=600.0,
        transition_width_m=30.0,
        seed=42
    )

    config = CoastalSceneConfig(
        scene_radius_m=2000.0,
        resolution_m=5.0,
        radar_height_m=10.0,
        shoreline_y_m=600.0,         # Shoreline 600m north of radar
        n_boats=6,
        n_buoys=10,
        n_towers=2,
        terrain_params=terrain_params,
        seed=42
    )

    scene = build_coastal_scene(config)

    print(f"  Terrain: {scene.heightmap.shape}")
    print(f"  Targets: {len(scene.targets)}")
    for t in scene.targets:
        print(f"    {t.name}: range={np.sqrt(t.position[0]**2 + t.position[1]**2):.0f}m, RCS={t.rcs_m2:.1f}m²")

    return scene


def run_demo():
    """Run the complete demonstration."""
    print("=" * 60)
    print("MAGNETRON RADAR COASTAL SIMULATION")
    print("=" * 60)

    # Create scene
    scene = create_demo_scene()

    # Create radar config
    radar = MagnetronConfig(
        name="X-Band Magnetron",
        center_frequency_hz=9.5e9,
        pulse_width_s=0.5e-6,        # 75m resolution
        prf_hz=3000.0,
        peak_power_w=10000.0,        # 10 kW
        horizontal_beamwidth_deg=1.8,
        vertical_beamwidth_deg=22.0,
        antenna_gain_dbi=28.0,
        antenna_height_m=10.0,
        max_range_m=2000.0
    )

    print(f"\nRadar Configuration:")
    print(f"  Frequency: {radar.center_frequency_hz/1e9:.2f} GHz")
    print(f"  Pulse: {radar.pulse_width_s*1e6:.1f} μs")
    print(f"  Resolution: {radar.range_resolution_m:.1f} m")
    print(f"  Max Range: {radar.max_range_m:.0f} m")
    print(f"  Beamwidth: {radar.horizontal_beamwidth_deg:.1f}°")

    # Clutter parameters
    land_params = LandClutterParams(terrain_type="mixed")
    sea_params = SeaClutterParams(sea_state=2, wind_speed_mps=8.0)

    # Run simulation
    print("\nRunning simulation...")
    result = run_magnetron_simulation(
        radar, scene,
        n_azimuths=360,
        n_range_bins=1024,
        land_params=land_params,
        sea_params=sea_params,
        verbose=True
    )

    # Create visualization
    print("\nGenerating visualization...")
    create_visualization(result)

    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE!")
    print("=" * 60)


def create_visualization(result):
    """Create multi-panel visualization."""

    scene = result.scene
    config = result.config

    # Create figure with 4 panels
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle('Magnetron Radar Coastal Simulation\n'
                 f'X-band, {config.pulse_width_s*1e6:.1f}μs pulse, {config.range_resolution_m:.0f}m resolution',
                 fontsize=14, fontweight='bold', color='white')

    # Panel 1: 3D Terrain View
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    plot_3d_terrain(ax1, scene)

    # Panel 2: Top-down with Visibility
    ax2 = fig.add_subplot(2, 2, 2)
    plot_visibility_map(ax2, result)

    # Panel 3: PPI Display
    ax3 = fig.add_subplot(2, 2, 3)
    plot_ppi(ax3, result)

    # Panel 4: Range Profile
    ax4 = fig.add_subplot(2, 2, 4)
    plot_range_profile(ax4, result)

    fig.patch.set_facecolor('#1a1a1a')
    plt.tight_layout()
    fig.savefig('magnetron_coastal_demo.png', dpi=150, facecolor='#1a1a1a', bbox_inches='tight')
    print("  Saved: magnetron_coastal_demo.png")
    plt.close()

    # Additional PPI-only figure
    fig2, ax = plt.subplots(figsize=(10, 10))
    plot_ppi_detailed(ax, result)
    fig2.patch.set_facecolor('black')
    plt.tight_layout()
    fig2.savefig('magnetron_ppi.png', dpi=150, facecolor='black', bbox_inches='tight')
    print("  Saved: magnetron_ppi.png")
    plt.close()


def plot_3d_terrain(ax, scene):
    """Plot 3D terrain view."""
    # Subsample for performance
    step = 5
    X = scene.x_coords[::step]
    Y = scene.y_coords[::step]
    Z = scene.heightmap[::step, ::step]

    X_grid, Y_grid = np.meshgrid(X, Y)

    # Color by elevation and water/land
    colors = np.zeros((*Z.shape, 4))

    for j in range(Z.shape[0]):
        for i in range(Z.shape[1]):
            y = Y[j]
            z = Z[j, i]

            if y < scene.config.shoreline_y_m:
                # Water - blue
                colors[j, i] = [0.1, 0.3, 0.8, 0.8]
            else:
                # Land - green to brown based on elevation
                t = min(1, z / 100)
                colors[j, i] = [0.2 + 0.5*t, 0.6 - 0.3*t, 0.1, 1.0]

    ax.plot_surface(X_grid, Y_grid, Z, facecolors=colors,
                    rstride=1, cstride=1, shade=True, antialiased=False)

    # Mark radar position
    ax.scatter([0], [0], [scene.config.radar_height_m], c='red', s=100,
               marker='^', label='Radar')

    # Mark targets
    for target in scene.targets:
        color = 'yellow' if target.target_type == 'boat' else 'cyan' if target.target_type == 'buoy' else 'magenta'
        ax.scatter([target.position[0]], [target.position[1]], [target.position[2]],
                   c=color, s=50, marker='o')

    ax.set_xlabel('East (m)', color='white')
    ax.set_ylabel('North (m)', color='white')
    ax.set_zlabel('Height (m)', color='white')
    ax.set_title('3D Terrain View', color='lime', fontsize=11)

    ax.set_facecolor('#1a1a1a')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False


def plot_visibility_map(ax, result):
    """Plot top-down view with visibility overlay."""
    scene = result.scene

    # Show terrain height with visibility overlay
    extent = [scene.x_coords[0], scene.x_coords[-1],
              scene.y_coords[0], scene.y_coords[-1]]

    # Create composite image
    terrain_img = scene.heightmap / max(scene.heightmap.max(), 1)

    # Darken shadowed areas
    display_img = terrain_img.copy()
    display_img[~result.visibility_mask] *= 0.2  # Darken shadows

    # Water mask
    water_img = np.zeros((*terrain_img.shape, 4))
    water_img[scene.water_mask] = [0.1, 0.3, 0.8, 0.6]

    # Show terrain
    ax.imshow(display_img, extent=extent, origin='lower', cmap='terrain',
              vmin=0, vmax=1)

    # Overlay water
    ax.imshow(water_img, extent=extent, origin='lower')

    # Mark shadow regions (red outline)
    shadow_mask = ~result.visibility_mask & scene.land_mask
    if np.any(shadow_mask):
        ax.contour(scene.x_coords, scene.y_coords, shadow_mask.astype(float),
                   levels=[0.5], colors='red', linewidths=1, alpha=0.7)

    # Shoreline
    ax.axhline(scene.config.shoreline_y_m, color='cyan', linestyle='--',
               alpha=0.5, label='Shoreline')

    # Radar and targets
    ax.plot(0, 0, 'r^', markersize=12, label='Radar')

    for target in scene.targets:
        color = 'yellow' if target.target_type == 'boat' else 'cyan' if target.target_type == 'buoy' else 'magenta'
        ax.plot(target.position[0], target.position[1], 'o',
                color=color, markersize=6)

    # Range rings
    for r in [500, 1000, 1500, 2000]:
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(r * np.cos(theta), r * np.sin(theta), 'w-', alpha=0.2, lw=0.5)

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel('East (m)', color='white')
    ax.set_ylabel('North (m)', color='white')
    ax.set_title('Visibility Map (red = shadowed)', color='lime', fontsize=11)
    ax.set_facecolor('#1a1a1a')
    ax.tick_params(colors='white')
    ax.set_aspect('equal')


def plot_ppi(ax, result):
    """Plot PPI display."""
    ppi = result.ppi_cartesian
    config = result.config

    # Normalize for display
    ppi_norm = normalize_for_display(ppi, dynamic_range_db=40)

    extent = [-config.max_range_m, config.max_range_m,
              -config.max_range_m, config.max_range_m]

    ax.imshow(ppi_norm, extent=extent, origin='lower', cmap='viridis',
              vmin=0, vmax=1)

    # Range rings
    for r in [500, 1000, 1500]:
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(r * np.cos(theta), r * np.sin(theta), 'g-', alpha=0.3, lw=0.5)

    # Mark detected targets
    for t in result.detected_targets:
        az_rad = np.radians(t['azimuth_deg'])
        x = t['range_m'] * np.sin(az_rad)
        y = t['range_m'] * np.cos(az_rad)
        ax.plot(x, y, 'c+', markersize=8, markeredgewidth=1)

    # Radar position
    ax.plot(0, 0, 'r^', markersize=8)

    ax.set_xlabel('East (m)', color='white')
    ax.set_ylabel('North (m)', color='white')
    ax.set_title(f'PPI Display ({len(result.detected_targets)} targets detected)',
                 color='lime', fontsize=11)
    ax.set_facecolor('black')
    ax.tick_params(colors='white')
    ax.set_aspect('equal')


def plot_ppi_detailed(ax, result):
    """Detailed PPI display."""
    ppi = result.ppi_cartesian
    config = result.config
    scene = result.scene

    ppi_norm = normalize_for_display(ppi, dynamic_range_db=45)

    extent = [-config.max_range_m, config.max_range_m,
              -config.max_range_m, config.max_range_m]

    im = ax.imshow(ppi_norm, extent=extent, origin='lower', cmap='viridis')

    # Range rings with labels
    for r in [500, 1000, 1500, 2000]:
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(r * np.cos(theta), r * np.sin(theta), 'g-', alpha=0.3, lw=0.5)
        ax.text(r * 0.7, r * 0.7, f'{r}m', color='green', fontsize=8, alpha=0.5)

    # Cardinal directions
    ax.text(0, config.max_range_m - 100, 'N', color='white', fontsize=12,
            ha='center', fontweight='bold')
    ax.text(config.max_range_m - 100, 0, 'E', color='white', fontsize=12,
            va='center', fontweight='bold')

    # Mark targets
    for t in result.detected_targets:
        az_rad = np.radians(t['azimuth_deg'])
        x = t['range_m'] * np.sin(az_rad)
        y = t['range_m'] * np.cos(az_rad)

        color = 'yellow' if t['type'] == 'boat' else 'cyan' if t['type'] == 'buoy' else 'magenta'
        ax.plot(x, y, 'o', color=color, markersize=4, markerfacecolor='none')

    # Shoreline indicator
    theta_shore = np.linspace(-np.pi/3, np.pi/3, 50)
    r_shore = scene.config.shoreline_y_m / np.cos(theta_shore)
    valid = r_shore < config.max_range_m
    ax.plot(r_shore[valid] * np.sin(theta_shore[valid]),
            r_shore[valid] * np.cos(theta_shore[valid]),
            'c--', alpha=0.4, lw=1, label='Shoreline')

    ax.set_xlabel('East (m)', color='white', fontsize=11)
    ax.set_ylabel('North (m)', color='white', fontsize=11)
    ax.set_title(f'Magnetron Radar PPI\n{config.pulse_width_s*1e6:.1f}μs pulse, '
                 f'{config.range_resolution_m:.0f}m resolution',
                 color='lime', fontsize=12)
    ax.set_facecolor('black')
    ax.tick_params(colors='white')
    ax.set_aspect('equal')

    # Legend
    ax.plot([], [], 'yo', markersize=6, markerfacecolor='none', label='Boats')
    ax.plot([], [], 'co', markersize=6, markerfacecolor='none', label='Buoys')
    ax.plot([], [], 'mo', markersize=6, markerfacecolor='none', label='Towers')
    ax.legend(loc='lower right', facecolor='#333', labelcolor='white', fontsize=9)


def plot_range_profile(ax, result):
    """Plot range profile at selected azimuth."""
    # Get profile at azimuth pointing into land (north)
    az_idx = 0  # North

    profile = result.ppi_polar[az_idx, :]
    ranges = result.range_bins_m

    # Normalize and convert to dB
    profile_norm = profile / (profile.max() + 1e-30)
    profile_db = 10 * np.log10(profile_norm + 1e-10)

    ax.plot(ranges / 1000, profile_db, 'g-', linewidth=1.5)

    # Mark shoreline
    ax.axvline(result.scene.config.shoreline_y_m / 1000, color='cyan',
               linestyle='--', alpha=0.5, label='Shoreline')

    # Mark blind zone
    ax.axvspan(0, result.config.blind_range_m / 1000, alpha=0.3, color='red',
               label='Blind Zone')

    ax.set_xlabel('Range (km)', color='white')
    ax.set_ylabel('Power (dB)', color='white')
    ax.set_title('Range Profile (North azimuth)', color='lime', fontsize=11)
    ax.set_xlim(0, result.config.max_range_m / 1000)
    ax.set_ylim(-50, 5)
    ax.grid(True, alpha=0.3, color='white')
    ax.set_facecolor('#1a1a1a')
    ax.tick_params(colors='white')
    ax.legend(loc='upper right', facecolor='#333', labelcolor='white', fontsize=9)


if __name__ == "__main__":
    run_demo()
