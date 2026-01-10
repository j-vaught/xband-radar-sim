#!/usr/bin/env python3
"""
Realistic Radar Simulation with:
- Atmospheric effects (refraction, attenuation)
- Multipath propagation (ground reflections)
- Beam spreading (azimuth convolution)
- 2D PSF convolution
- Polar-to-Cartesian scan conversion with interpolation

720 azimuths (0.5 degree spacing) for high-quality output.
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
from propagation.scene import Scene, TargetObject, create_corner_reflector, create_sphere
from propagation.atmosphere import (
    AtmosphereConfig, apply_atmospheric_attenuation,
    effective_earth_radius, is_below_horizon
)
from propagation.multipath import (
    SurfaceProperties, SURFACE_SEAWATER, SURFACE_CONCRETE,
    two_ray_multipath, multipath_propagation_factor
)
from signal.waveform import WaveformConfig, generate_lfm_chirp
from signal.matched_filter import matched_filter_fft_windowed
from signal.ppi_processing import (
    PPIProcessingConfig, process_ppi, quick_beam_spread,
    normalize_for_display, polar_to_cartesian
)


def antenna_pattern_1d(angle_rad: float, beamwidth_rad: float) -> float:
    """Sinc-squared antenna pattern."""
    u = 2.783 * angle_rad / beamwidth_rad
    if abs(u) < 1e-6:
        return 1.0
    return max((np.sin(np.pi * u) / (np.pi * u)) ** 2, 1e-6)


def two_way_pattern(az_off: float, el_off: float,
                    az_bw: float, el_bw: float) -> float:
    """Two-way antenna pattern gain."""
    return (antenna_pattern_1d(az_off, az_bw) *
            antenna_pattern_1d(el_off, el_bw)) ** 2


def create_scene() -> Scene:
    """Create test scene with multiple targets."""
    scene = Scene()

    # Building NE - large corner reflector at 200m
    cr_v, cr_f = create_corner_reflector(edge_length_m=50.0)
    scene.add_target(TargetObject(
        name="building_NE",
        position=(200 * np.cos(np.deg2rad(45)), 200 * np.sin(np.deg2rad(45)), 0),
        vertices=cr_v, faces=cr_f))

    # Dome W - sphere at 100m, elevated
    s_v, s_f = create_sphere(radius_m=20.0, n_segments=16)
    scene.add_target(TargetObject(
        name="dome_W",
        position=(100 * np.cos(np.deg2rad(270)), 100 * np.sin(np.deg2rad(270)), 20),
        vertices=s_v, faces=s_f))

    # Warehouse S - corner reflector at 300m
    cr_v2, cr_f2 = create_corner_reflector(edge_length_m=40.0)
    scene.add_target(TargetObject(
        name="warehouse_S",
        position=(300 * np.cos(np.deg2rad(180)), 300 * np.sin(np.deg2rad(180)), 0),
        vertices=cr_v2, faces=cr_f2))

    # Additional target E - small sphere at 150m
    s_v2, s_f2 = create_sphere(radius_m=10.0, n_segments=12)
    scene.add_target(TargetObject(
        name="tank_E",
        position=(150 * np.cos(np.deg2rad(90)), 150 * np.sin(np.deg2rad(90)), 5),
        vertices=s_v2, faces=s_f2))

    # Target NW - corner at 250m
    cr_v3, cr_f3 = create_corner_reflector(edge_length_m=30.0)
    scene.add_target(TargetObject(
        name="structure_NW",
        position=(250 * np.cos(np.deg2rad(315)), 250 * np.sin(np.deg2rad(315)), 0),
        vertices=cr_v3, faces=cr_f3))

    return scene


def run_realistic_sim(scene: Scene,
                       radar_pos: np.ndarray,
                       wf_config: WaveformConfig,
                       radar_config: RadarConfig,
                       atm_config: AtmosphereConfig,
                       surface: SurfaceProperties,
                       n_rays_per_az: int,
                       n_azimuths: int,
                       max_range: float,
                       enable_multipath: bool = True,
                       enable_atmosphere: bool = True) -> tuple:
    """
    Run realistic radar simulation with atmospheric and multipath effects.

    Args:
        scene: Scene with targets
        radar_pos: Radar position (x, y, z)
        wf_config: Waveform configuration
        radar_config: Radar configuration
        atm_config: Atmosphere configuration
        surface: Ground/water surface properties
        n_rays_per_az: Rays per azimuth for ray tracing
        n_azimuths: Number of azimuth samples
        max_range: Maximum range in meters
        enable_multipath: Enable multipath propagation
        enable_atmosphere: Enable atmospheric effects

    Returns:
        azimuths, ranges, ppi (raw, before PPI processing)
    """
    c = 299792458.0
    triangles = scene.get_all_triangles()

    az_bw_rad = np.deg2rad(radar_config.horizontal_beamwidth_deg)
    el_bw_rad = np.deg2rad(radar_config.vertical_beamwidth_deg)

    _, tx_waveform = generate_lfm_chirp(wf_config)
    pulse_samples = len(tx_waveform)

    max_time = 2 * max_range / c
    n_samples = int(max_time * wf_config.sample_rate_hz) + pulse_samples * 2
    n_ranges = n_samples - pulse_samples + 1

    azimuths = np.linspace(0, 360, n_azimuths, endpoint=False)
    ppi = np.zeros((n_azimuths, n_ranges))

    sample_time = 1.0 / wf_config.sample_rate_hz
    ranges = np.arange(n_ranges) * sample_time * c / 2

    # Get effective Earth radius for refraction
    if enable_atmosphere:
        eff_earth_radius = effective_earth_radius(atm_config)
    else:
        eff_earth_radius = 6.371e6

    radar_height = radar_pos[2]

    for az_idx, az_deg in enumerate(azimuths):
        boresight_az = np.deg2rad(az_deg)

        # Generate ray bundle around boresight
        ray_az_offset = np.random.uniform(-1.5 * az_bw_rad, 1.5 * az_bw_rad, n_rays_per_az)
        ray_el_offset = np.random.uniform(-1.5 * el_bw_rad, 1.5 * el_bw_rad, n_rays_per_az)

        ray_az = boresight_az + ray_az_offset
        ray_el = ray_el_offset

        directions = np.column_stack([
            np.cos(ray_el) * np.cos(ray_az),
            np.cos(ray_el) * np.sin(ray_az),
            np.sin(ray_el)
        ])
        origins = np.tile(radar_pos, (n_rays_per_az, 1))

        # Trace rays
        bundle = trace_rays_cpu(origins, directions, triangles, max_range)

        # Build received signal
        rx_signal = np.zeros(n_samples, dtype=complex)

        for i in range(bundle.n_rays):
            if bundle.hit_mask[i]:
                range_m = bundle.path_lengths_m[i]
                hit_point = bundle.hit_points[i]
                target_height = hit_point[2]

                # Antenna pattern gain
                antenna_gain = two_way_pattern(ray_az_offset[i], ray_el_offset[i],
                                               az_bw_rad, el_bw_rad)

                # Base amplitude (radar equation simplified)
                amplitude = antenna_gain / (range_m**2 + 1)

                # Apply atmospheric attenuation
                if enable_atmosphere:
                    power = amplitude**2
                    power_atten = apply_atmospheric_attenuation(
                        np.array([power]), np.array([range_m]),
                        wf_config.center_frequency_hz, atm_config
                    )[0]
                    amplitude = np.sqrt(power_atten)

                # Apply multipath propagation factor
                if enable_multipath and target_height > 0:
                    # Horizontal range for multipath calculation
                    horiz_range = np.sqrt((hit_point[0] - radar_pos[0])**2 +
                                         (hit_point[1] - radar_pos[1])**2)

                    if horiz_range > 10:  # Avoid singularity at very short range
                        mp_factor, _, _ = two_ray_multipath(
                            radar_height, target_height, horiz_range,
                            wf_config.center_frequency_hz, surface, 'H'
                        )
                        # Apply multipath (two-way)
                        amplitude = amplitude * np.abs(mp_factor)**2

                # Delay
                delay_samples = int(2 * range_m / c * wf_config.sample_rate_hz)

                if delay_samples + pulse_samples < n_samples:
                    # Phase from range
                    phase = 4 * np.pi * wf_config.center_frequency_hz * range_m / c

                    echo = amplitude * np.exp(1j * phase) * tx_waveform
                    rx_signal[delay_samples:delay_samples + pulse_samples] += echo

        # Matched filter / pulse compression
        compressed = matched_filter_fft_windowed(rx_signal, tx_waveform, "hamming")
        valid_len = min(len(compressed), n_ranges)
        ppi[az_idx, :valid_len] = np.abs(compressed[:valid_len])**2

    # Trim to max range
    max_range_idx = int(max_range / (c / 2 / wf_config.sample_rate_hz))
    return azimuths, ranges[:max_range_idx], ppi[:, :max_range_idx]


def plot_ppi_polar(azimuths: np.ndarray,
                    ranges: np.ndarray,
                    ppi: np.ndarray,
                    title: str,
                    output_path: str,
                    dynamic_range_db: float = 50.0):
    """Plot PPI in polar coordinates."""
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})

    az_rad = np.deg2rad(azimuths)
    R, AZ = np.meshgrid(ranges, az_rad)

    # Normalize for display
    ppi_norm = normalize_for_display(ppi, dynamic_range_db)

    ax.pcolormesh(AZ, R, ppi_norm, cmap='viridis', shading='auto')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')

    # Range rings
    for r in [100, 200, 300, 400]:
        if r < ranges[-1]:
            ax.plot(np.linspace(0, 2*np.pi, 100), [r]*100, 'g-', alpha=0.3, lw=0.5)

    ax.set_title(title, color='green', fontsize=12, pad=15)
    ax.tick_params(colors='green')
    ax.grid(True, color='green', alpha=0.3, lw=0.5)
    ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
    ax.set_xticklabels(['N', 'E', 'S', 'W'], color='green')

    plt.savefig(output_path, dpi=150, facecolor='black', bbox_inches='tight')
    plt.close()


def plot_ppi_cartesian(cartesian: np.ndarray,
                        x: np.ndarray,
                        y: np.ndarray,
                        title: str,
                        output_path: str,
                        dynamic_range_db: float = 50.0):
    """Plot PPI in Cartesian coordinates."""
    fig, ax = plt.subplots(figsize=(8, 8))

    # Normalize for display
    cart_norm = normalize_for_display(cartesian, dynamic_range_db)

    extent = [x[0], x[-1], y[0], y[-1]]
    ax.imshow(cart_norm, cmap='viridis', origin='lower', extent=extent, aspect='equal')

    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')

    # Range rings
    for r in [100, 200, 300, 400]:
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(r * np.cos(theta), r * np.sin(theta), 'g-', alpha=0.3, lw=0.5)

    # Cardinal directions
    max_r = x[-1]
    ax.text(0, max_r * 0.95, 'N', color='green', ha='center', va='top', fontsize=10)
    ax.text(max_r * 0.95, 0, 'E', color='green', ha='left', va='center', fontsize=10)
    ax.text(0, -max_r * 0.95, 'S', color='green', ha='center', va='bottom', fontsize=10)
    ax.text(-max_r * 0.95, 0, 'W', color='green', ha='right', va='center', fontsize=10)

    ax.set_xlabel('East-West (m)', color='green')
    ax.set_ylabel('North-South (m)', color='green')
    ax.set_title(title, color='green', fontsize=12, pad=15)
    ax.tick_params(colors='green')

    plt.savefig(output_path, dpi=150, facecolor='black', bbox_inches='tight')
    plt.close()


def main():
    print("=" * 70)
    print("  REALISTIC RADAR SIMULATION")
    print("  - Atmospheric Effects (refraction, attenuation)")
    print("  - Multipath Propagation (ground reflections)")
    print("  - Beam Spreading & PSF Convolution")
    print("  - 720 Azimuths (0.5 degree resolution)")
    print("=" * 70)
    print(f"\nNumba acceleration: {'ENABLED' if NUMBA_AVAILABLE else 'DISABLED'}")

    # Configuration
    scene = create_scene()
    radar_pos = np.array([0.0, 0.0, 10.0])  # 10m height
    radar_config = RadarConfig()

    # Waveform: 50 MHz bandwidth, 10 us pulse
    wf_config = WaveformConfig(
        pulse_width_s=10e-6,
        bandwidth_hz=50e6,
        center_frequency_hz=9.41e9,
        sample_rate_hz=200e6
    )

    # Atmosphere: standard conditions
    atm_config = AtmosphereConfig(
        temperature_k=288.15,  # 15C
        pressure_hpa=1013.25,
        relative_humidity=0.6,
        enable_refraction=True,
        enable_attenuation=True
    )

    # Surface: concrete (urban environment)
    surface = SURFACE_CONCRETE

    # PPI processing configuration
    ppi_config = PPIProcessingConfig(
        apply_beam_spreading=True,
        apply_psf=True,
        apply_scan_conversion=True,
        beamwidth_deg=radar_config.horizontal_beamwidth_deg,
        range_resolution_m=3.0,  # From bandwidth
        output_size=512,
        dynamic_range_db=50.0
    )

    # Simulation parameters
    n_rays_per_az = 2000
    n_azimuths = 720  # 0.5 degree spacing
    max_range = 500.0

    c = 299792458.0
    range_res = c / (2 * wf_config.bandwidth_hz)

    print(f"\nSimulation Parameters:")
    print(f"  Frequency: {wf_config.center_frequency_hz/1e9:.2f} GHz")
    print(f"  Bandwidth: {wf_config.bandwidth_hz/1e6:.0f} MHz")
    print(f"  Range Resolution: {range_res:.2f} m")
    print(f"  Beamwidth: {radar_config.horizontal_beamwidth_deg:.1f} deg")
    print(f"  Azimuths: {n_azimuths} (spacing: {360/n_azimuths:.2f} deg)")
    print(f"  Rays per azimuth: {n_rays_per_az}")
    print(f"  Max range: {max_range:.0f} m")

    # Run simulation
    print(f"\n[1/4] Running ray tracing simulation...")
    start = time.time()
    azimuths, ranges, ppi_raw = run_realistic_sim(
        scene, radar_pos, wf_config, radar_config, atm_config, surface,
        n_rays_per_az, n_azimuths, max_range,
        enable_multipath=True, enable_atmosphere=True
    )
    sim_time = time.time() - start
    print(f"      Completed in {sim_time:.1f}s")

    # Save raw PPI (before processing)
    print(f"\n[2/4] Saving raw PPI (before beam spreading)...")
    plot_ppi_polar(azimuths, ranges, ppi_raw,
                   "Raw PPI (Before Processing)",
                   "ppi_raw.png")
    print(f"      Saved: ppi_raw.png")

    # Apply PPI processing (beam spreading, PSF, scan conversion)
    print(f"\n[3/4] Applying PPI processing...")
    print(f"      - Beam spreading (beamwidth: {ppi_config.beamwidth_deg:.1f} deg)")
    print(f"      - 2D PSF convolution")
    print(f"      - Polar-to-Cartesian interpolation")

    start = time.time()
    ppi_processed, x, y = process_ppi(ppi_raw, azimuths, ranges, ppi_config)
    proc_time = time.time() - start
    print(f"      Completed in {proc_time:.2f}s")

    # Save processed PPI (Cartesian)
    print(f"\n[4/4] Saving processed PPI outputs...")
    plot_ppi_cartesian(ppi_processed, x, y,
                       "Realistic PPI (Processed, Cartesian)",
                       "ppi_realistic_cartesian.png")
    print(f"      Saved: ppi_realistic_cartesian.png")

    # Also save polar version with beam spreading only
    ppi_spread = quick_beam_spread(ppi_raw, radar_config.horizontal_beamwidth_deg, n_azimuths)
    plot_ppi_polar(azimuths, ranges, ppi_spread,
                   "PPI with Beam Spreading (Polar)",
                   "ppi_beam_spread_polar.png")
    print(f"      Saved: ppi_beam_spread_polar.png")

    # Comparison plot
    print(f"\n[Bonus] Creating comparison plot...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Raw
    ax = axes[0]
    az_rad = np.deg2rad(azimuths)
    R, AZ = np.meshgrid(ranges, az_rad)
    ppi_norm = normalize_for_display(ppi_raw, 50.0)
    ax.set_title("1. Raw (Ray Artifacts)", color='white', fontsize=11)
    ax_polar = fig.add_subplot(1, 3, 1, projection='polar')
    ax_polar.pcolormesh(AZ, R, ppi_norm, cmap='viridis', shading='auto')
    ax_polar.set_theta_zero_location('N')
    ax_polar.set_theta_direction(-1)
    ax_polar.set_facecolor('black')
    ax_polar.set_title("1. Raw (Ray Artifacts)", color='green', fontsize=11)
    ax_polar.tick_params(colors='green')
    axes[0].remove()

    # Beam spread
    ax2_polar = fig.add_subplot(1, 3, 2, projection='polar')
    ppi_norm2 = normalize_for_display(ppi_spread, 50.0)
    ax2_polar.pcolormesh(AZ, R, ppi_norm2, cmap='viridis', shading='auto')
    ax2_polar.set_theta_zero_location('N')
    ax2_polar.set_theta_direction(-1)
    ax2_polar.set_facecolor('black')
    ax2_polar.set_title("2. Beam Spreading", color='green', fontsize=11)
    ax2_polar.tick_params(colors='green')
    axes[1].remove()

    # Cartesian
    ax3 = axes[2]
    cart_norm = normalize_for_display(ppi_processed, 50.0)
    extent = [x[0], x[-1], y[0], y[-1]]
    ax3.imshow(cart_norm, cmap='viridis', origin='lower', extent=extent, aspect='equal')
    ax3.set_facecolor('black')
    ax3.set_title("3. Full Processing (Cartesian)", color='green', fontsize=11)
    ax3.tick_params(colors='green')
    for r in [100, 200, 300, 400]:
        theta = np.linspace(0, 2*np.pi, 100)
        ax3.plot(r * np.cos(theta), r * np.sin(theta), 'g-', alpha=0.3, lw=0.5)

    fig.patch.set_facecolor('black')
    plt.tight_layout()
    plt.savefig("ppi_comparison.png", dpi=150, facecolor='black', bbox_inches='tight')
    plt.close()
    print(f"      Saved: ppi_comparison.png")

    # Summary
    print(f"\n" + "=" * 70)
    print(f"  SIMULATION COMPLETE")
    print(f"=" * 70)
    print(f"\nOutputs:")
    print(f"  ppi_raw.png              - Raw PPI (shows ray artifacts)")
    print(f"  ppi_beam_spread_polar.png - With beam spreading (polar)")
    print(f"  ppi_realistic_cartesian.png - Full processing (Cartesian)")
    print(f"  ppi_comparison.png        - Side-by-side comparison")
    print(f"\nPhysics included:")
    print(f"  [x] Atmospheric refraction (effective Earth radius)")
    print(f"  [x] Atmospheric attenuation (O2, H2O absorption)")
    print(f"  [x] Multipath (two-ray ground reflection)")
    print(f"  [x] Beam spreading (azimuth convolution)")
    print(f"  [x] 2D PSF (range + azimuth)")
    print(f"  [x] Scan conversion (bilinear interpolation)")
    print(f"\nTotal time: {sim_time + proc_time:.1f}s")


if __name__ == "__main__":
    main()
