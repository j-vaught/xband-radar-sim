#!/usr/bin/env python3
"""
Pulse Length Comparison

Shows how changing pulse length affects radar image:
- Longer pulse = more energy = better SNR
- Longer pulse = more samples in matched filter = different compression behavior
- Longer pulse = larger minimum range (blind zone)

We test: 1μs, 10μs, 100μs, 1000μs with CONSTANT bandwidth (50 MHz)
This isolates the pulse length effect from range resolution changes.
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
from signal.ppi_processing import quick_beam_spread, normalize_for_display


def antenna_pattern_1d(angle_rad, beamwidth_rad):
    u = 2.783 * angle_rad / beamwidth_rad
    if abs(u) < 1e-6:
        return 1.0
    return max((np.sin(np.pi * u) / (np.pi * u)) ** 2, 1e-6)


def two_way_pattern(az_off, el_off, az_bw, el_bw):
    return (antenna_pattern_1d(az_off, az_bw) * antenna_pattern_1d(el_off, el_bw)) ** 2


def create_scene():
    """Create scene with targets at various ranges."""
    scene = Scene()

    # Close target: 80m NE
    cr_v, cr_f = create_corner_reflector(edge_length_m=30.0)
    scene.add_target(TargetObject(
        name="close_NE",
        position=(80 * np.cos(np.deg2rad(45)), 80 * np.sin(np.deg2rad(45)), 5),
        vertices=cr_v, faces=cr_f))

    # Medium target: 200m E
    s_v, s_f = create_sphere(radius_m=15.0, n_segments=16)
    scene.add_target(TargetObject(
        name="medium_E",
        position=(200, 0, 10),
        vertices=s_v, faces=s_f))

    # Far target: 350m S
    cr_v2, cr_f2 = create_corner_reflector(edge_length_m=40.0)
    scene.add_target(TargetObject(
        name="far_S",
        position=(0, -350, 0),
        vertices=cr_v2, faces=cr_f2))

    # Another close: 100m W
    s_v2, s_f2 = create_sphere(radius_m=20.0, n_segments=12)
    scene.add_target(TargetObject(
        name="close_W",
        position=(-100, 0, 15),
        vertices=s_v2, faces=s_f2))

    # Medium: 250m NW
    cr_v3, cr_f3 = create_corner_reflector(edge_length_m=35.0)
    scene.add_target(TargetObject(
        name="medium_NW",
        position=(250 * np.cos(np.deg2rad(135)), 250 * np.sin(np.deg2rad(135)), 0),
        vertices=cr_v3, faces=cr_f3))

    return scene


def run_sim_with_pulse(scene, radar_pos, pulse_width_s, bandwidth_hz,
                       radar_config, n_rays_per_az, n_azimuths, max_range):
    """Run simulation with specific pulse width."""
    c = 299792458.0

    wf_config = WaveformConfig(
        pulse_width_s=pulse_width_s,
        bandwidth_hz=bandwidth_hz,
        center_frequency_hz=9.41e9,
        sample_rate_hz=200e6
    )

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

    # Calculate blind range (can't receive while transmitting)
    blind_range = c * pulse_width_s / 2

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

                # Skip targets in blind zone
                if range_m < blind_range:
                    continue

                hit_point = bundle.hit_points[i]
                target_height = hit_point[2]

                antenna_gain = two_way_pattern(ray_az_offset[i], ray_el_offset[i],
                                               az_bw_rad, el_bw_rad)
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

    return azimuths, ranges[:max_range_idx], ppi[:, :max_range_idx], wf_config, blind_range


def main():
    print("=" * 70)
    print("  PULSE LENGTH COMPARISON")
    print("  Testing: 1μs, 10μs, 100μs, 1000μs")
    print("  Bandwidth: 50 MHz (constant) → Range Res: 3.0m")
    print("=" * 70)
    print(f"\nNumba: {'ENABLED' if NUMBA_AVAILABLE else 'DISABLED'}")

    # Setup
    scene = create_scene()
    radar_pos = np.array([0.0, 0.0, 10.0])
    radar_config = RadarConfig()

    # Simulation parameters
    bandwidth_hz = 50e6  # CONSTANT bandwidth
    n_rays_per_az = 500  # Minimal for speed
    n_azimuths = 120  # 3 degree spacing
    max_range = 300.0

    c = 299792458.0
    range_res = c / (2 * bandwidth_hz)

    # Pulse widths to test
    pulse_widths = [1e-6, 10e-6, 100e-6, 1000e-6]
    labels = ['1 μs', '10 μs', '100 μs', '1000 μs']

    results = []

    for pw, label in zip(pulse_widths, labels):
        print(f"\n[{label}] Running simulation...")
        print(f"  Pulse samples: {int(pw * 200e6)}")
        print(f"  Time-BW product: {pw * bandwidth_hz:.0f}")
        print(f"  Blind range: {c * pw / 2:.1f} m")

        np.random.seed(42)  # Reproducibility
        start = time.time()
        az, rng, ppi, wf_config, blind_range = run_sim_with_pulse(
            scene, radar_pos, pw, bandwidth_hz,
            radar_config, n_rays_per_az, n_azimuths, max_range
        )
        elapsed = time.time() - start
        print(f"  Completed in {elapsed:.1f}s")

        # Apply beam spreading
        ppi_spread = quick_beam_spread(ppi, radar_config.horizontal_beamwidth_deg, n_azimuths)

        results.append({
            'pulse_width': pw,
            'label': label,
            'azimuths': az,
            'ranges': rng,
            'ppi_raw': ppi,
            'ppi_spread': ppi_spread,
            'blind_range': blind_range,
            'tb_product': pw * bandwidth_hz
        })

    # Create comparison figure
    print("\n" + "=" * 70)
    print("Creating comparison figures...")

    # Figure 1: 2x2 grid of PPIs
    fig, axes = plt.subplots(2, 2, figsize=(14, 14), subplot_kw={'projection': 'polar'})
    fig.patch.set_facecolor('black')

    for idx, (ax, res) in enumerate(zip(axes.flat, results)):
        az_rad = np.deg2rad(res['azimuths'])
        R, AZ = np.meshgrid(res['ranges'], az_rad)

        ppi_norm = normalize_for_display(res['ppi_spread'], 50.0)

        ax.pcolormesh(AZ, R, ppi_norm, cmap='viridis', shading='auto')
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_facecolor('black')

        # Range rings
        for r in [100, 200, 300, 400]:
            if r < res['ranges'][-1]:
                ax.plot(np.linspace(0, 2*np.pi, 100), [r]*100, 'g-', alpha=0.3, lw=0.5)

        # Blind zone
        if res['blind_range'] > 10:
            ax.fill_between(np.linspace(0, 2*np.pi, 100), 0, res['blind_range'],
                          color='red', alpha=0.3)

        ax.set_title(f"{res['label']}\nTB={res['tb_product']:.0f}, Blind={res['blind_range']:.0f}m",
                    color='green', fontsize=11, pad=10)
        ax.tick_params(colors='green', labelsize=8)
        ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
        ax.set_xticklabels(['N', 'E', 'S', 'W'], color='green', fontsize=9)

    plt.suptitle(f'Pulse Length Comparison (Bandwidth = 50 MHz, Range Res = {range_res:.1f}m)',
                color='white', fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('pulse_length_comparison_ppi.png', dpi=150, facecolor='black', bbox_inches='tight')
    plt.close()
    print("  Saved: pulse_length_comparison_ppi.png")

    # Figure 2: Zoomed comparison (one target, all pulse lengths)
    fig, axes = plt.subplots(2, 2, figsize=(14, 14), subplot_kw={'projection': 'polar'})
    fig.patch.set_facecolor('black')

    for idx, (ax, res) in enumerate(zip(axes.flat, results)):
        az_rad = np.deg2rad(res['azimuths'])
        R, AZ = np.meshgrid(res['ranges'], az_rad)

        ppi_norm = normalize_for_display(res['ppi_spread'], 50.0)

        ax.pcolormesh(AZ, R, ppi_norm, cmap='viridis', shading='auto')
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_facecolor('black')

        # Zoom to close target area (NE, around 80m)
        ax.set_rlim(0, 150)
        ax.set_thetamin(0)
        ax.set_thetamax(90)

        # Blind zone
        if res['blind_range'] > 5:
            theta_fill = np.linspace(0, np.pi/2, 50)
            ax.fill_between(theta_fill, 0, res['blind_range'], color='red', alpha=0.4)
            ax.text(np.deg2rad(45), res['blind_range']/2, 'BLIND',
                   color='red', fontsize=10, ha='center', va='center')

        ax.set_title(f"{res['label']} - Zoomed NE Quadrant", color='cyan', fontsize=11, pad=10)
        ax.tick_params(colors='green', labelsize=8)

    plt.suptitle('Zoomed View: Effect of Pulse Length on Close Targets',
                color='white', fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('pulse_length_comparison_zoomed.png', dpi=150, facecolor='black', bbox_inches='tight')
    plt.close()
    print("  Saved: pulse_length_comparison_zoomed.png")

    # Figure 3: Range profile comparison (single azimuth)
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    fig.patch.set_facecolor('black')

    target_azimuth_idx = 45  # ~45 degrees (NE target)

    for idx, (ax, res) in enumerate(zip(axes, results)):
        ranges = res['ranges']
        profile = res['ppi_spread'][target_azimuth_idx, :]

        # Convert to dB
        profile_db = 10 * np.log10(profile + 1e-30)
        profile_db = profile_db - np.max(profile_db)  # Normalize to peak

        ax.plot(ranges, profile_db, 'g-', linewidth=1)
        ax.fill_between(ranges, -60, profile_db, alpha=0.3, color='green')

        ax.set_facecolor('black')
        ax.set_xlim(0, max_range)
        ax.set_ylim(-60, 5)
        ax.set_ylabel('Power (dB)', color='green')
        ax.tick_params(colors='green')
        ax.grid(True, alpha=0.3, color='green')

        # Mark blind zone
        if res['blind_range'] > 5:
            ax.axvspan(0, res['blind_range'], color='red', alpha=0.3)
            ax.text(res['blind_range']/2, -10, 'BLIND', color='red',
                   fontsize=10, ha='center', va='center')

        # Annotations
        ax.set_title(f"{res['label']} | TB Product: {res['tb_product']:.0f} | "
                    f"Pulse Samples: {int(res['pulse_width']*200e6)}",
                    color='cyan', fontsize=10)

        # Mark expected target position
        ax.axvline(x=80, color='yellow', linestyle='--', alpha=0.5)
        ax.text(82, -5, 'Target\n~80m', color='yellow', fontsize=8)

    axes[-1].set_xlabel('Range (m)', color='green')
    plt.suptitle('Range Profile at 45° Azimuth (NE Target)',
                color='white', fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('pulse_length_range_profiles.png', dpi=150, facecolor='black', bbox_inches='tight')
    plt.close()
    print("  Saved: pulse_length_range_profiles.png")

    # Figure 4: Target extent analysis
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('black')

    # For each pulse width, measure the apparent target size
    for idx, (ax, res) in enumerate(zip(axes.flat, results)):
        # Find target peak at ~80m, ~45° (NE target)
        az_center = 45  # degrees
        range_center = 80  # meters

        # Get indices
        az_idx = int(az_center / (360 / len(res['azimuths'])))

        # Find range index
        range_idx = np.argmin(np.abs(res['ranges'] - range_center))

        # Extract local region
        az_window = 30  # ±15 bins
        rng_window = 50  # ±25 bins

        az_start = max(0, az_idx - az_window//2)
        az_end = min(n_azimuths, az_idx + az_window//2)
        rng_start = max(0, range_idx - rng_window//2)
        rng_end = min(len(res['ranges']), range_idx + rng_window//2)

        local_ppi = res['ppi_spread'][az_start:az_end, rng_start:rng_end]
        local_ranges = res['ranges'][rng_start:rng_end]
        local_az = res['azimuths'][az_start:az_end]

        # Normalize
        local_norm = normalize_for_display(local_ppi, 40.0)

        im = ax.imshow(local_norm, aspect='auto', cmap='viridis', origin='lower',
                      extent=[local_ranges[0], local_ranges[-1], local_az[0], local_az[-1]])

        ax.set_facecolor('black')
        ax.set_xlabel('Range (m)', color='green')
        ax.set_ylabel('Azimuth (°)', color='green')
        ax.tick_params(colors='green')

        # Measure target extent (FWHM)
        if local_ppi.max() > 0:
            # Range extent (at center azimuth)
            range_slice = local_ppi[az_window//2, :]
            if range_slice.max() > 0:
                half_max = range_slice.max() / 2
                above_half = range_slice > half_max
                if np.any(above_half):
                    range_extent = np.sum(above_half) * (local_ranges[1] - local_ranges[0])
                else:
                    range_extent = 0
            else:
                range_extent = 0

            # Azimuth extent (at peak range)
            peak_range_idx = np.argmax(range_slice)
            az_slice = local_ppi[:, peak_range_idx]
            if az_slice.max() > 0:
                half_max_az = az_slice.max() / 2
                above_half_az = az_slice > half_max_az
                if np.any(above_half_az):
                    az_extent = np.sum(above_half_az) * (local_az[1] - local_az[0])
                else:
                    az_extent = 0
            else:
                az_extent = 0
        else:
            range_extent = 0
            az_extent = 0

        ax.set_title(f"{res['label']}\nRange extent: {range_extent:.1f}m | Az extent: {az_extent:.1f}°",
                    color='cyan', fontsize=10)

    plt.suptitle('Target Extent Analysis (Close Target at ~80m, 45°)',
                color='white', fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('pulse_length_target_extent.png', dpi=150, facecolor='black', bbox_inches='tight')
    plt.close()
    print("  Saved: pulse_length_target_extent.png")

    # Summary
    print("\n" + "=" * 70)
    print("  PULSE LENGTH COMPARISON COMPLETE")
    print("=" * 70)
    print("\nOutputs:")
    print("  pulse_length_comparison_ppi.png    - 2x2 PPI grid")
    print("  pulse_length_comparison_zoomed.png - Zoomed NE quadrant")
    print("  pulse_length_range_profiles.png    - Range profiles at 45°")
    print("  pulse_length_target_extent.png     - Target extent analysis")
    print("\nKey Observations:")
    print("  - Longer pulse = larger blind zone (can't see close targets)")
    print("  - Time-bandwidth product affects compression gain")
    print("  - With constant bandwidth, range resolution stays ~3m")
    print("  - Target extent should be similar (set by bandwidth & beamwidth)")


if __name__ == "__main__":
    main()
