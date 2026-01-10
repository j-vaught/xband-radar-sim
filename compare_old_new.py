#!/usr/bin/env python3
"""
Compare old vs new radar simulation.

Shows the difference between:
- Old: Raw ray tracing with no beam spreading
- New: Full physics (atmosphere, multipath, beam spreading, PSF, scan conversion)
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
from propagation.atmosphere import AtmosphereConfig, apply_atmospheric_attenuation
from propagation.multipath import SURFACE_CONCRETE, two_ray_multipath
from signal.waveform import WaveformConfig, generate_lfm_chirp
from signal.matched_filter import matched_filter_fft_windowed
from signal.ppi_processing import (
    PPIProcessingConfig, process_ppi, quick_beam_spread, normalize_for_display
)


def antenna_pattern_1d(angle_rad, beamwidth_rad):
    u = 2.783 * angle_rad / beamwidth_rad
    if abs(u) < 1e-6:
        return 1.0
    return max((np.sin(np.pi * u) / (np.pi * u)) ** 2, 1e-6)


def two_way_pattern(az_off, el_off, az_bw, el_bw):
    return (antenna_pattern_1d(az_off, az_bw) * antenna_pattern_1d(el_off, el_bw)) ** 2


def create_scene():
    """Create test scene with targets."""
    scene = Scene()

    # Same targets as before
    cr_v, cr_f = create_corner_reflector(edge_length_m=50.0)
    scene.add_target(TargetObject(
        name="building_NE",
        position=(200 * np.cos(np.deg2rad(45)), 200 * np.sin(np.deg2rad(45)), 0),
        vertices=cr_v, faces=cr_f))

    s_v, s_f = create_sphere(radius_m=20.0, n_segments=16)
    scene.add_target(TargetObject(
        name="dome_W",
        position=(100 * np.cos(np.deg2rad(270)), 100 * np.sin(np.deg2rad(270)), 20),
        vertices=s_v, faces=s_f))

    cr_v2, cr_f2 = create_corner_reflector(edge_length_m=40.0)
    scene.add_target(TargetObject(
        name="warehouse_S",
        position=(300 * np.cos(np.deg2rad(180)), 300 * np.sin(np.deg2rad(180)), 0),
        vertices=cr_v2, faces=cr_f2))

    return scene


def run_old_sim(scene, radar_pos, wf_config, radar_config, n_rays_per_az, n_azimuths, max_range):
    """OLD simulation - no atmosphere, no multipath, no beam spreading."""
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

    for az_idx, az_deg in enumerate(azimuths):
        boresight_az = np.deg2rad(az_deg)

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

        bundle = trace_rays_cpu(origins, directions, triangles, max_range)

        rx_signal = np.zeros(n_samples, dtype=complex)

        for i in range(bundle.n_rays):
            if bundle.hit_mask[i]:
                range_m = bundle.path_lengths_m[i]
                antenna_gain = two_way_pattern(ray_az_offset[i], ray_el_offset[i], az_bw_rad, el_bw_rad)

                delay_samples = int(2 * range_m / c * wf_config.sample_rate_hz)

                if delay_samples + pulse_samples < n_samples:
                    amplitude = antenna_gain / (range_m**2 + 1)
                    phase = 4 * np.pi * wf_config.center_frequency_hz * range_m / c

                    echo = amplitude * np.exp(1j * phase) * tx_waveform
                    rx_signal[delay_samples:delay_samples + pulse_samples] += echo

        compressed = matched_filter_fft_windowed(rx_signal, tx_waveform, "hamming")
        valid_len = min(len(compressed), n_ranges)
        ppi[az_idx, :valid_len] = np.abs(compressed[:valid_len])**2

    max_range_idx = int(max_range / (c / 2 / wf_config.sample_rate_hz))
    return azimuths, ranges[:max_range_idx], ppi[:, :max_range_idx]


def run_new_sim(scene, radar_pos, wf_config, radar_config, n_rays_per_az, n_azimuths, max_range):
    """NEW simulation - with atmosphere, multipath effects."""
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

    atm_config = AtmosphereConfig()
    surface = SURFACE_CONCRETE
    radar_height = radar_pos[2]

    for az_idx, az_deg in enumerate(azimuths):
        boresight_az = np.deg2rad(az_deg)

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

        bundle = trace_rays_cpu(origins, directions, triangles, max_range)

        rx_signal = np.zeros(n_samples, dtype=complex)

        for i in range(bundle.n_rays):
            if bundle.hit_mask[i]:
                range_m = bundle.path_lengths_m[i]
                hit_point = bundle.hit_points[i]
                target_height = hit_point[2]

                antenna_gain = two_way_pattern(ray_az_offset[i], ray_el_offset[i], az_bw_rad, el_bw_rad)
                amplitude = antenna_gain / (range_m**2 + 1)

                # Atmospheric attenuation
                power = amplitude**2
                power_atten = apply_atmospheric_attenuation(
                    np.array([power]), np.array([range_m]),
                    wf_config.center_frequency_hz, atm_config
                )[0]
                amplitude = np.sqrt(power_atten)

                # Multipath
                if target_height > 0:
                    horiz_range = np.sqrt((hit_point[0] - radar_pos[0])**2 +
                                         (hit_point[1] - radar_pos[1])**2)
                    if horiz_range > 10:
                        mp_factor, _, _ = two_ray_multipath(
                            radar_height, target_height, horiz_range,
                            wf_config.center_frequency_hz, surface, 'H'
                        )
                        amplitude = amplitude * np.abs(mp_factor)**2

                delay_samples = int(2 * range_m / c * wf_config.sample_rate_hz)

                if delay_samples + pulse_samples < n_samples:
                    phase = 4 * np.pi * wf_config.center_frequency_hz * range_m / c
                    echo = amplitude * np.exp(1j * phase) * tx_waveform
                    rx_signal[delay_samples:delay_samples + pulse_samples] += echo

        compressed = matched_filter_fft_windowed(rx_signal, tx_waveform, "hamming")
        valid_len = min(len(compressed), n_ranges)
        ppi[az_idx, :valid_len] = np.abs(compressed[:valid_len])**2

    max_range_idx = int(max_range / (c / 2 / wf_config.sample_rate_hz))
    return azimuths, ranges[:max_range_idx], ppi[:, :max_range_idx]


def main():
    print("=" * 70)
    print("  OLD vs NEW RADAR SIMULATION COMPARISON")
    print("=" * 70)
    print(f"\nNumba: {'ENABLED' if NUMBA_AVAILABLE else 'DISABLED'}")

    # Setup
    scene = create_scene()
    radar_pos = np.array([0.0, 0.0, 10.0])
    radar_config = RadarConfig()

    wf_config = WaveformConfig(
        pulse_width_s=10e-6,
        bandwidth_hz=50e6,
        center_frequency_hz=9.41e9,
        sample_rate_hz=200e6
    )

    n_rays_per_az = 2000
    n_azimuths = 360  # Use 360 for faster comparison
    max_range = 500.0

    # Run OLD simulation
    print("\n[1/4] Running OLD simulation (no physics enhancements)...")
    np.random.seed(42)  # For reproducibility
    start = time.time()
    az_old, rng_old, ppi_old = run_old_sim(
        scene, radar_pos, wf_config, radar_config, n_rays_per_az, n_azimuths, max_range
    )
    old_time = time.time() - start
    print(f"      Completed in {old_time:.1f}s")

    # Run NEW simulation
    print("\n[2/4] Running NEW simulation (with atmosphere + multipath)...")
    np.random.seed(42)  # Same seed for fair comparison
    start = time.time()
    az_new, rng_new, ppi_new = run_new_sim(
        scene, radar_pos, wf_config, radar_config, n_rays_per_az, n_azimuths, max_range
    )
    new_time = time.time() - start
    print(f"      Completed in {new_time:.1f}s")

    # Apply beam spreading to NEW
    print("\n[3/4] Applying beam spreading to NEW simulation...")
    ppi_new_spread = quick_beam_spread(ppi_new, radar_config.horizontal_beamwidth_deg, n_azimuths)

    # Full processing
    ppi_config = PPIProcessingConfig(
        apply_beam_spreading=True,
        apply_psf=True,
        apply_scan_conversion=True,
        beamwidth_deg=radar_config.horizontal_beamwidth_deg,
        range_resolution_m=3.0,
        output_size=512,
        dynamic_range_db=50.0
    )
    ppi_full, x, y = process_ppi(ppi_new, az_new, rng_new, ppi_config)

    # Create comparison figure
    print("\n[4/4] Creating comparison figure...")

    fig = plt.figure(figsize=(20, 10))
    fig.patch.set_facecolor('black')

    # Prepare meshgrid for polar plots
    az_rad_old = np.deg2rad(az_old)
    R_old, AZ_old = np.meshgrid(rng_old, az_rad_old)
    az_rad_new = np.deg2rad(az_new)
    R_new, AZ_new = np.meshgrid(rng_new, az_rad_new)

    # 1. OLD - Raw (no processing)
    ax1 = fig.add_subplot(2, 3, 1, projection='polar')
    ppi_norm = normalize_for_display(ppi_old, 50.0)
    ax1.pcolormesh(AZ_old, R_old, ppi_norm, cmap='viridis', shading='auto')
    ax1.set_theta_zero_location('N')
    ax1.set_theta_direction(-1)
    ax1.set_facecolor('black')
    ax1.set_title("OLD: Raw Simulation\n(No physics, no spreading)", color='red', fontsize=11, pad=10)
    ax1.tick_params(colors='green', labelsize=8)

    # 2. NEW - Raw (with atmosphere + multipath, no spreading)
    ax2 = fig.add_subplot(2, 3, 2, projection='polar')
    ppi_norm2 = normalize_for_display(ppi_new, 50.0)
    ax2.pcolormesh(AZ_new, R_new, ppi_norm2, cmap='viridis', shading='auto')
    ax2.set_theta_zero_location('N')
    ax2.set_theta_direction(-1)
    ax2.set_facecolor('black')
    ax2.set_title("NEW: With Atmosphere+Multipath\n(No beam spreading yet)", color='yellow', fontsize=11, pad=10)
    ax2.tick_params(colors='green', labelsize=8)

    # 3. NEW - With beam spreading
    ax3 = fig.add_subplot(2, 3, 3, projection='polar')
    ppi_norm3 = normalize_for_display(ppi_new_spread, 50.0)
    ax3.pcolormesh(AZ_new, R_new, ppi_norm3, cmap='viridis', shading='auto')
    ax3.set_theta_zero_location('N')
    ax3.set_theta_direction(-1)
    ax3.set_facecolor('black')
    ax3.set_title("NEW: With Beam Spreading\n(Azimuth convolution)", color='cyan', fontsize=11, pad=10)
    ax3.tick_params(colors='green', labelsize=8)

    # 4. OLD zoomed (to show ray artifacts)
    ax4 = fig.add_subplot(2, 3, 4, projection='polar')
    ax4.pcolormesh(AZ_old, R_old, ppi_norm, cmap='viridis', shading='auto')
    ax4.set_theta_zero_location('N')
    ax4.set_theta_direction(-1)
    ax4.set_facecolor('black')
    ax4.set_rlim(0, 250)  # Zoom in
    ax4.set_title("OLD: Zoomed (see ray artifacts)", color='red', fontsize=11, pad=10)
    ax4.tick_params(colors='green', labelsize=8)

    # 5. NEW zoomed (beam spread)
    ax5 = fig.add_subplot(2, 3, 5, projection='polar')
    ax5.pcolormesh(AZ_new, R_new, ppi_norm3, cmap='viridis', shading='auto')
    ax5.set_theta_zero_location('N')
    ax5.set_theta_direction(-1)
    ax5.set_facecolor('black')
    ax5.set_rlim(0, 250)  # Zoom in
    ax5.set_title("NEW: Zoomed (realistic blobs)", color='cyan', fontsize=11, pad=10)
    ax5.tick_params(colors='green', labelsize=8)

    # 6. Full Cartesian processing
    ax6 = fig.add_subplot(2, 3, 6)
    cart_norm = normalize_for_display(ppi_full, 50.0)
    extent = [x[0], x[-1], y[0], y[-1]]
    ax6.imshow(cart_norm, cmap='viridis', origin='lower', extent=extent, aspect='equal')
    ax6.set_facecolor('black')
    ax6.set_title("NEW: Full Processing\n(PSF + Scan Conversion)", color='lime', fontsize=11, pad=10)
    ax6.tick_params(colors='green', labelsize=8)
    for r in [100, 200, 300, 400]:
        theta = np.linspace(0, 2*np.pi, 100)
        ax6.plot(r * np.cos(theta), r * np.sin(theta), 'g-', alpha=0.3, lw=0.5)
    ax6.set_xlim(-max_range, max_range)
    ax6.set_ylim(-max_range, max_range)

    plt.suptitle("OLD vs NEW Radar Simulation Comparison", color='white', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    output_path = "old_vs_new_comparison.png"
    plt.savefig(output_path, dpi=150, facecolor='black', bbox_inches='tight')
    plt.close()

    print(f"\n" + "=" * 70)
    print(f"  COMPARISON COMPLETE")
    print(f"=" * 70)
    print(f"\nSaved: {output_path}")
    print(f"\nKey Differences:")
    print(f"  OLD: Ray artifacts visible, targets appear as thin radial lines")
    print(f"  NEW: Targets appear as realistic blobs spread across beamwidth")
    print(f"\nPhysics in NEW version:")
    print(f"  - Atmospheric attenuation (O2, H2O)")
    print(f"  - Multipath (ground reflection interference)")
    print(f"  - Beam spreading (3.9 deg beamwidth)")
    print(f"  - 2D PSF convolution")
    print(f"  - Bilinear scan conversion")


if __name__ == "__main__":
    main()
