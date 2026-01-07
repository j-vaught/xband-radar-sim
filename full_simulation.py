#!/usr/bin/env python3
"""
Multi-Pulse Ray Tracing Simulation.

Demonstrates how pulse length affects range resolution and shadow "smearing".
Longer pulses = wider shadows in range dimension.

Uses full Numba-accelerated ray tracing.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

from config import RadarConfig
from propagation.cpu_raytrace import trace_rays_cpu, NUMBA_AVAILABLE
from propagation.scene import Scene, TargetObject, create_corner_reflector, create_flat_plate, create_sphere


def create_scene():
    """Create scene with targets."""
    scene = Scene()
    
    # Large building at 200m NE
    cr_v, cr_f = create_corner_reflector(edge_length_m=50.0)
    scene.add_target(TargetObject(
        name="building_NE",
        position=(200 * np.cos(np.deg2rad(45)), 200 * np.sin(np.deg2rad(45)), 0),
        vertices=cr_v,
        faces=cr_f
    ))
    
    # Ship at 150m ESE
    p_v, p_f = create_flat_plate(width_m=40.0, height_m=20.0)
    scene.add_target(TargetObject(
        name="ship_ESE",
        position=(150 * np.cos(np.deg2rad(120)), 150 * np.sin(np.deg2rad(120)), 0),
        vertices=p_v,
        faces=p_f
    ))
    
    # Dome at 100m W
    s_v, s_f = create_sphere(radius_m=20.0, n_segments=16)
    scene.add_target(TargetObject(
        name="dome_W",
        position=(100 * np.cos(np.deg2rad(270)), 100 * np.sin(np.deg2rad(270)), 20),
        vertices=s_v,
        faces=s_f
    ))
    
    # Warehouse at 300m S
    cr_v2, cr_f2 = create_corner_reflector(edge_length_m=40.0)
    scene.add_target(TargetObject(
        name="warehouse_S",
        position=(300 * np.cos(np.deg2rad(180)), 300 * np.sin(np.deg2rad(180)), 0),
        vertices=cr_v2,
        faces=cr_f2
    ))
    
    return scene


def run_ray_tracing_ppi(
    scene: Scene,
    radar_pos: np.ndarray,
    n_rays_per_az: int,
    n_azimuths: int,
    n_ranges: int,
    max_range: float,
    pulse_smear_bins: int = 1  # Number of range bins to smear (simulates pulse width)
):
    """Run ray tracing with pulse-width smearing."""
    
    triangles = scene.get_all_triangles()
    
    azimuths = np.linspace(0, 360, n_azimuths, endpoint=False)
    ranges = np.linspace(0, max_range, n_ranges)
    range_res = max_range / n_ranges
    ppi = np.zeros((n_azimuths, n_ranges))
    
    total_hits = 0
    start_time = time.time()
    
    for az_idx, az_deg in enumerate(azimuths):
        az_rad = np.deg2rad(az_deg)
        beamwidth_rad = np.deg2rad(3.9)
        elevation_spread = np.deg2rad(25.0)
        
        ray_az = az_rad + np.random.uniform(-beamwidth_rad/2, beamwidth_rad/2, n_rays_per_az)
        ray_el = np.random.uniform(-elevation_spread/2, elevation_spread/2, n_rays_per_az)
        
        directions = np.column_stack([
            np.cos(ray_el) * np.cos(ray_az),
            np.cos(ray_el) * np.sin(ray_az),
            np.sin(ray_el)
        ])
        
        origins = np.tile(radar_pos, (n_rays_per_az, 1))
        
        bundle = trace_rays_cpu(origins, directions, triangles, max_range)
        
        for i in range(bundle.n_rays):
            if bundle.hit_mask[i]:
                range_m = bundle.path_lengths_m[i]
                range_idx = int(range_m / range_res)
                
                # Apply pulse smearing - spread hit across multiple bins
                for smear in range(pulse_smear_bins):
                    r_idx = range_idx + smear
                    if 0 <= r_idx < n_ranges:
                        # Power decreases with smear distance
                        smear_weight = 1.0 - (smear / pulse_smear_bins) * 0.5
                        ppi[az_idx, r_idx] += smear_weight / (range_m**4 + 1)
                        total_hits += 1
        
        if (az_idx + 1) % 36 == 0:
            elapsed = time.time() - start_time
            print(f"  {az_idx+1}/{n_azimuths} azimuths, {total_hits} hits, {elapsed:.1f}s")
    
    return azimuths, ranges, ppi


def plot_multi_pulse_comparison(results, output_path):
    """Create comparison plot of different pulse lengths."""
    n_pulses = len(results)
    
    fig, axes = plt.subplots(1, n_pulses, figsize=(6*n_pulses, 6), 
                             subplot_kw={'projection': 'polar'})
    if n_pulses == 1:
        axes = [axes]
    
    for idx, (pulse_name, azimuths, ranges, ppi) in enumerate(results):
        ax = axes[idx]
        
        az_rad = np.deg2rad(azimuths)
        R, AZ = np.meshgrid(ranges, az_rad)
        
        ppi_db = 10 * np.log10(ppi + 1e-30)
        vmax = np.max(ppi_db)
        vmin = vmax - 40
        ppi_norm = np.clip((ppi_db - vmin) / (vmax - vmin + 1e-10), 0, 1)
        
        ax.pcolormesh(AZ, R, ppi_norm, cmap='gray', shading='auto')
        
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_facecolor('black')
        
        for r in [100, 200, 300, 400]:
            ax.plot(np.linspace(0, 2*np.pi, 100), [r]*100, 'g-', alpha=0.3, lw=0.5)
        
        ax.set_title(f'{pulse_name}', color='green', fontsize=14, pad=10)
        ax.tick_params(colors='green')
        ax.grid(True, color='green', alpha=0.3, lw=0.5)
        ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
        ax.set_xticklabels(['N', 'E', 'S', 'W'], color='green')
    
    fig.patch.set_facecolor('black')
    fig.suptitle('Range Resolution vs Pulse Width - Full Ray Tracing\n'
                 'Longer pulse = more range smearing (shadow lengthening)',
                 color='green', fontsize=16, y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor='black', bbox_inches='tight')
    plt.close()


def main():
    print("=" * 70)
    print("  Multi-Pulse Ray Tracing Simulation - Shadow Effect Demonstration")
    print("=" * 70)
    print()
    
    print(f"Numba acceleration: {'ENABLED' if NUMBA_AVAILABLE else 'DISABLED'}")
    print()
    
    # Create scene
    scene = create_scene()
    radar_pos = np.array([0.0, 0.0, 5.0])
    
    print(f"Scene: {len(scene.get_all_triangles())} triangles, {len(scene.targets)} targets")
    print()
    
    # Define pulse configurations
    # Pulse smear in bins ~ c * pulse_width / (2 * range_resolution)
    # More bins = longer pulse = more shadow
    pulse_configs = [
        ("Short Pulse (1μs)\n1.5m resolution", 1),     # 1 bin smear
        ("Medium Pulse (10μs)\n15m resolution", 10),    # 10 bin smear  
        ("Long Pulse (50μs)\n75m resolution", 50),     # 50 bin smear
    ]
    
    results = []
    n_rays = 14000  # ~5M total across all configs
    n_azimuths = 360
    n_ranges = 868
    max_range = 500.0
    
    for pulse_name, smear_bins in pulse_configs:
        print(f"Running: {pulse_name.split(chr(10))[0]}...")
        print(f"  Smear bins: {smear_bins}")
        
        azimuths, ranges, ppi = run_ray_tracing_ppi(
            scene=scene,
            radar_pos=radar_pos,
            n_rays_per_az=n_rays,
            n_azimuths=n_azimuths,
            n_ranges=n_ranges,
            max_range=max_range,
            pulse_smear_bins=smear_bins
        )
        
        results.append((pulse_name, azimuths, ranges, ppi))
        print()
    
    # Plot comparison
    output_path = os.path.join(os.path.dirname(__file__), 'ppi_pulse_comparison.png')
    plot_multi_pulse_comparison(results, output_path)
    print(f"Saved: {output_path}")
    
    print()
    print("=" * 70)
    print("  SIMULATION COMPLETE - Compare shadow lengths!")
    print("=" * 70)


if __name__ == "__main__":
    main()
