#!/usr/bin/env python3
"""
Target Shadow / Trailing Edge Demonstration

Shows realistic asymmetric target response:
- Sharp leading edge (front of target facing radar)
- Extended trailing edge / "shadow" (back of target)

Physical causes:
1. Target depth - back surfaces return later
2. Multiple scattering within target
3. Creeping/diffraction waves around edges
4. Exponential decay of energy penetration
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from dataclasses import dataclass
from typing import Tuple, List

from signal.waveform import WaveformConfig, generate_lfm_chirp
from signal.matched_filter import matched_filter_fft_windowed
from signal.ppi_processing import quick_beam_spread

C = 299792458.0


@dataclass
class ExtendedTarget:
    """Target with physical depth for realistic range response."""
    center: Tuple[float, float, float]  # (x, y, z)
    depth_m: float           # Physical depth in range direction
    width_m: float           # Cross-range width
    rcs_dbsm: float          # Peak RCS
    # Trailing edge parameters
    trailing_decay: float = 0.3    # Exponential decay rate for trailing edge
    trailing_extent: float = 2.0   # How many depths the trail extends
    n_scatterers: int = 10         # Number of scattering points along depth


def generate_target_response(
    target: ExtendedTarget,
    radar_pos: np.ndarray,
    ray_direction: np.ndarray,
    tx_waveform: np.ndarray,
    sample_rate: float,
    center_freq: float,
    blind_range: float = 0
) -> Tuple[np.ndarray, float]:
    """
    Generate realistic target response with leading/trailing edge asymmetry.

    Returns:
        compressed: Compressed pulse response
        base_range: Range to target center
    """
    # Vector to target center
    to_target = np.array(target.center) - radar_pos
    base_range = np.linalg.norm(to_target)

    if base_range < blind_range:
        return None, base_range

    # Check if ray points at target (simplified)
    target_dir = to_target / base_range
    dot = np.dot(ray_direction, target_dir)
    angle = np.arccos(np.clip(dot, -1, 1))
    target_angular_size = np.arctan(target.width_m / base_range)

    if angle > target_angular_size + np.radians(1.0):
        return None, base_range

    # Generate scatterers along target depth
    # Front edge is at base_range - depth/2, back edge at base_range + depth/2
    half_depth = target.depth_m / 2

    # Scatterer positions: more at front, exponentially decaying toward back
    scatterer_offsets = []
    scatterer_amplitudes = []

    # 1. Strong leading edge (front surface facing radar)
    scatterer_offsets.append(-half_depth)
    scatterer_amplitudes.append(1.0)  # Full strength

    # 2. Internal scatterers (decreasing strength going back)
    for i in range(1, target.n_scatterers):
        # Position within target
        frac = i / target.n_scatterers
        offset = -half_depth + frac * target.depth_m
        # Exponential decay going into target
        amp = np.exp(-target.trailing_decay * frac * 5)
        scatterer_offsets.append(offset)
        scatterer_amplitudes.append(amp * 0.5)  # Internal weaker than surface

    # 3. Back surface (weaker than front due to shadowing)
    scatterer_offsets.append(half_depth)
    scatterer_amplitudes.append(0.3)  # 30% of front surface

    # 4. Trailing edge / shadow (diffraction, creeping waves, multipath)
    # This extends BEHIND the target and decays exponentially
    trailing_length = target.depth_m * target.trailing_extent
    n_trailing = int(target.n_scatterers * target.trailing_extent)

    for i in range(1, n_trailing + 1):
        offset = half_depth + (i / n_trailing) * trailing_length
        # Exponential decay
        amp = 0.2 * np.exp(-target.trailing_decay * i)
        if amp > 0.01:  # Threshold
            scatterer_offsets.append(offset)
            scatterer_amplitudes.append(amp)

    # Build received signal
    max_offset = max(scatterer_offsets) + 50  # Extra margin
    max_delay_samples = int(2 * (base_range + max_offset) / C * sample_rate)
    n_samples = len(tx_waveform) + max_delay_samples + 100

    rx_signal = np.zeros(n_samples, dtype=complex)
    rcs_linear = 10 ** (target.rcs_dbsm / 10)

    for offset, amp in zip(scatterer_offsets, scatterer_amplitudes):
        scatter_range = base_range + offset
        if scatter_range < blind_range or scatter_range < 0:
            continue

        # Radar equation
        received_power = (rcs_linear * amp**2) / (scatter_range**4 + 1)

        delay_samples = int(2 * scatter_range / C * sample_rate)

        if delay_samples + len(tx_waveform) < n_samples:
            phase = 4 * np.pi * center_freq * scatter_range / C
            # Add some phase randomization for internal scatterers
            if abs(offset) < half_depth:
                phase += np.random.uniform(-np.pi/4, np.pi/4)

            echo = np.sqrt(received_power) * np.exp(1j * phase) * tx_waveform
            rx_signal[delay_samples:delay_samples + len(tx_waveform)] += echo

    # Apply matched filter
    compressed = matched_filter_fft_windowed(rx_signal, tx_waveform, "hamming")

    return compressed, base_range


def run_shadow_simulation(
    targets: List[ExtendedTarget],
    pulse_width_us: float,
    bandwidth_mhz: float,
    n_azimuths: int = 180,
    n_rays_per_az: int = 200,
    max_range: float = 500
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run simulation with extended targets showing shadow effect."""

    pulse_width_s = pulse_width_us * 1e-6
    bandwidth_hz = bandwidth_mhz * 1e6
    sample_rate = bandwidth_hz * 4
    blind_range = C * pulse_width_s / 2

    wf_config = WaveformConfig(
        pulse_width_s=pulse_width_s,
        bandwidth_hz=bandwidth_hz,
        center_frequency_hz=9.41e9,
        sample_rate_hz=sample_rate
    )

    _, tx_waveform = generate_lfm_chirp(wf_config)

    range_resolution = C / (2 * bandwidth_hz)
    n_range_bins = int(max_range / range_resolution) + 100

    azimuths = np.linspace(0, 360, n_azimuths, endpoint=False)
    ranges = np.arange(n_range_bins) * range_resolution

    ppi = np.zeros((n_azimuths, n_range_bins))
    radar_pos = np.array([0.0, 0.0, 10.0])
    beamwidth_deg = 3.9

    print(f"  Pulse: {pulse_width_us}μs, BW: {bandwidth_mhz}MHz, Res: {range_resolution:.1f}m")
    print(f"  Blind range: {blind_range:.0f}m")

    for az_idx, az_deg in enumerate(azimuths):
        if az_idx % 30 == 0:
            print(f"    Azimuth {az_idx}/{n_azimuths}")

        az_rad = np.radians(az_deg)

        for _ in range(n_rays_per_az):
            az_off = np.random.normal(0, beamwidth_deg / 3)
            el_off = np.random.normal(0, 10 / 3)

            ray_az = az_rad + np.radians(az_off)
            ray_el = np.radians(el_off)

            direction = np.array([
                np.cos(ray_el) * np.sin(ray_az),
                np.cos(ray_el) * np.cos(ray_az),
                np.sin(ray_el)
            ])
            direction = direction / np.linalg.norm(direction)

            for target in targets:
                compressed, _ = generate_target_response(
                    target, radar_pos, direction,
                    tx_waveform, sample_rate,
                    wf_config.center_frequency_hz,
                    blind_range
                )

                if compressed is not None:
                    # Map to range bins
                    for j, val in enumerate(np.abs(compressed)):
                        range_m = j * C / (2 * sample_rate)
                        range_bin = int(range_m / range_resolution)
                        if 0 <= range_bin < n_range_bins and val > 1e-12:
                            ppi[az_idx, range_bin] += val

    # Normalize
    if ppi.max() > 0:
        ppi = ppi / ppi.max()

    # Apply beam spreading
    ppi_spread = quick_beam_spread(ppi, beamwidth_deg, n_azimuths, range_sigma_bins=1.0)

    return azimuths, ranges[:n_range_bins], ppi_spread


def create_test_targets():
    """Create targets with varying depths to demonstrate shadow effect."""
    targets = []

    # Target 1: Large building at 200m NE (deep target = long shadow)
    r1, az1 = 200, np.radians(45)
    targets.append(ExtendedTarget(
        center=(r1 * np.sin(az1), r1 * np.cos(az1), 15),
        depth_m=30,          # 30m deep building
        width_m=25,
        rcs_dbsm=25,
        trailing_decay=0.25,  # Slower decay = longer shadow
        trailing_extent=3.0   # Trail extends 3x the depth
    ))

    # Target 2: Ship at 150m E (moderate depth)
    r2, az2 = 150, np.radians(90)
    targets.append(ExtendedTarget(
        center=(r2 * np.sin(az2), r2 * np.cos(az2), 5),
        depth_m=20,
        width_m=8,
        rcs_dbsm=22,
        trailing_decay=0.3,
        trailing_extent=2.5
    ))

    # Target 3: Small boat at 100m S (shallow = short shadow)
    r3, az3 = 100, np.radians(180)
    targets.append(ExtendedTarget(
        center=(r3 * np.sin(az3), r3 * np.cos(az3), 2),
        depth_m=8,
        width_m=4,
        rcs_dbsm=15,
        trailing_decay=0.5,   # Faster decay
        trailing_extent=1.5
    ))

    # Target 4: Container stack at 250m W (very deep = very long shadow)
    r4, az4 = 250, np.radians(270)
    targets.append(ExtendedTarget(
        center=(r4 * np.sin(az4), r4 * np.cos(az4), 10),
        depth_m=50,          # 50m deep container yard
        width_m=30,
        rcs_dbsm=28,
        trailing_decay=0.2,   # Very slow decay
        trailing_extent=4.0   # Trail extends 4x depth = 200m!
    ))

    # Target 5: Buoy at 120m NW (point-like, minimal shadow)
    r5, az5 = 120, np.radians(315)
    targets.append(ExtendedTarget(
        center=(r5 * np.sin(az5), r5 * np.cos(az5), 1),
        depth_m=2,           # Small depth
        width_m=2,
        rcs_dbsm=10,
        trailing_decay=0.8,   # Fast decay
        trailing_extent=1.0
    ))

    return targets


def plot_shadow_comparison(results_sym, results_asym, targets):
    """Compare symmetric vs asymmetric (shadow) responses."""

    az_sym, rng_sym, ppi_sym = results_sym
    az_asym, rng_asym, ppi_asym = results_asym

    # Figure 1: Side-by-side PPI comparison
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), subplot_kw={'projection': 'polar'})
    fig1.suptitle('Point Target vs Extended Target with Shadow',
                  fontsize=14, fontweight='bold', color='white')

    for ax, ppi, ranges, title in [
        (ax1, ppi_sym, rng_sym, 'Point Targets (Symmetric)'),
        (ax2, ppi_asym, rng_asym, 'Extended Targets (Asymmetric Shadow)')
    ]:
        az_rad = np.radians(az_sym)
        R, AZ = np.meshgrid(ranges, az_rad)

        ppi_plot = ppi.copy()
        ppi_plot[ppi_plot < 1e-6] = 1e-6

        ax.pcolormesh(AZ, R, ppi_plot, cmap='hot', shading='auto',
                     norm=LogNorm(vmin=1e-4, vmax=1))
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_facecolor('black')
        ax.set_title(title, color='lime', fontsize=11)
        ax.tick_params(colors='green')

        # Range rings
        theta_circle = np.linspace(0, 2*np.pi, 100)
        for r in [100, 200, 300, 400]:
            if r < ranges[-1]:
                ax.plot(theta_circle, [r]*100, 'g-', alpha=0.3, lw=0.5)

    fig1.patch.set_facecolor('black')
    plt.tight_layout()
    fig1.savefig('shadow_ppi_comparison.png', dpi=150, facecolor='black', bbox_inches='tight')
    print("  Saved: shadow_ppi_comparison.png")
    plt.close()

    # Figure 2: Range profiles showing asymmetry
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    fig2.suptitle('Range Profiles Showing Leading Edge / Trailing Shadow',
                  fontsize=14, fontweight='bold')

    target_azimuths = [45, 90, 270, 315]  # NE, E, W, NW
    target_names = ['Building NE (30m deep)', 'Ship E (20m deep)',
                   'Containers W (50m deep)', 'Buoy NW (2m)']
    target_ranges = [200, 150, 250, 120]

    for idx, (az, name, t_range) in enumerate(zip(target_azimuths, target_names, target_ranges)):
        ax = axes2[idx // 2, idx % 2]

        az_idx = int(az / 360 * len(az_sym))

        # Get profiles
        profile_sym = ppi_sym[az_idx, :]
        profile_asym = ppi_asym[az_idx, :]

        if profile_sym.max() > 0:
            profile_sym = profile_sym / profile_sym.max()
        if profile_asym.max() > 0:
            profile_asym = profile_asym / profile_asym.max()

        ax.plot(rng_sym, profile_sym, 'c-', linewidth=2,
               label='Point (Symmetric)', alpha=0.7)
        ax.plot(rng_asym, profile_asym, 'r-', linewidth=2,
               label='Extended (Shadow)', alpha=0.9)

        # Mark target position
        ax.axvline(t_range, color='yellow', linestyle='--', alpha=0.5)
        ax.annotate(f'Target\n{t_range}m', xy=(t_range, 0.8),
                   color='yellow', fontsize=9, ha='center')

        ax.set_xlabel('Range (m)')
        ax.set_ylabel('Normalized Intensity')
        ax.set_title(f'{name} @ {az}°', fontsize=11)
        ax.legend(loc='upper right', fontsize=9)
        ax.set_xlim(t_range - 80, t_range + 150)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)

        # Annotate leading/trailing
        ax.annotate('Leading\nEdge', xy=(t_range - 20, 0.3), color='green', fontsize=8)
        ax.annotate('Trailing\nShadow', xy=(t_range + 50, 0.2), color='red', fontsize=8)

    plt.tight_layout()
    fig2.savefig('shadow_range_profiles.png', dpi=150, bbox_inches='tight')
    print("  Saved: shadow_range_profiles.png")
    plt.close()

    # Figure 3: Zoomed Cartesian view of one target
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 6))
    fig3.suptitle('Zoomed View: Container Stack (50m deep) - Symmetric vs Shadow',
                  fontsize=14, fontweight='bold')

    # Convert to Cartesian and zoom on W target (containers at 250m)
    from signal.ppi_processing import polar_to_cartesian

    for ax, ppi, ranges, title in [
        (ax3a, ppi_sym, rng_sym, 'Point Target'),
        (ax3b, ppi_asym, rng_asym, 'Extended + Shadow')
    ]:
        cart, x, y = polar_to_cartesian(ppi, az_sym, ranges, output_size=400)

        # Zoom on W target (at -250, 0)
        target_x, target_y = -250, 0
        zoom = 150

        x_mask = (x > target_x - zoom) & (x < target_x + zoom)
        y_mask = (y > target_y - zoom) & (y < target_y + zoom)

        x_zoom = x[x_mask]
        y_zoom = y[y_mask]
        cart_zoom = cart[np.ix_(y_mask, x_mask)]

        cart_zoom[cart_zoom < 1e-6] = 1e-6

        extent = [x_zoom[0] - target_x, x_zoom[-1] - target_x,
                  y_zoom[0] - target_y, y_zoom[-1] - target_y]

        ax.imshow(cart_zoom, cmap='hot', origin='lower', extent=extent,
                 norm=LogNorm(vmin=1e-4, vmax=1), aspect='equal')

        ax.axhline(0, color='cyan', linestyle='--', alpha=0.3)
        ax.axvline(0, color='cyan', linestyle='--', alpha=0.3)
        ax.plot(0, 0, 'c+', markersize=15, markeredgewidth=2)

        # Arrow showing radar direction
        ax.annotate('← To Radar', xy=(-100, -120), color='lime', fontsize=10)
        ax.annotate('Shadow →', xy=(50, -120), color='red', fontsize=10)

        ax.set_xlabel('Range offset (m)')
        ax.set_ylabel('Cross-range (m)')
        ax.set_title(title, fontsize=11)
        ax.set_facecolor('black')

    plt.tight_layout()
    fig3.savefig('shadow_zoomed_comparison.png', dpi=150, bbox_inches='tight')
    print("  Saved: shadow_zoomed_comparison.png")
    plt.close()


def run_point_target_simulation(targets, pulse_width_us, bandwidth_mhz,
                                n_azimuths, n_rays_per_az, max_range):
    """Run simulation with point targets (no shadow) for comparison."""

    # Convert to simple point targets
    @dataclass
    class PointTarget:
        center: Tuple[float, float, float]
        width_m: float
        rcs_dbsm: float

    point_targets = [
        PointTarget(t.center, t.width_m, t.rcs_dbsm) for t in targets
    ]

    pulse_width_s = pulse_width_us * 1e-6
    bandwidth_hz = bandwidth_mhz * 1e6
    sample_rate = bandwidth_hz * 4
    blind_range = C * pulse_width_s / 2

    wf_config = WaveformConfig(
        pulse_width_s=pulse_width_s,
        bandwidth_hz=bandwidth_hz,
        center_frequency_hz=9.41e9,
        sample_rate_hz=sample_rate
    )

    _, tx_waveform = generate_lfm_chirp(wf_config)

    range_resolution = C / (2 * bandwidth_hz)
    n_range_bins = int(max_range / range_resolution) + 100

    azimuths = np.linspace(0, 360, n_azimuths, endpoint=False)
    ranges = np.arange(n_range_bins) * range_resolution

    ppi = np.zeros((n_azimuths, n_range_bins))
    radar_pos = np.array([0.0, 0.0, 10.0])
    beamwidth_deg = 3.9

    print(f"  Running point target simulation for comparison...")

    for az_idx, az_deg in enumerate(azimuths):
        az_rad = np.radians(az_deg)

        for _ in range(n_rays_per_az):
            az_off = np.random.normal(0, beamwidth_deg / 3)
            el_off = np.random.normal(0, 10 / 3)

            ray_az = az_rad + np.radians(az_off)
            ray_el = np.radians(el_off)

            direction = np.array([
                np.cos(ray_el) * np.sin(ray_az),
                np.cos(ray_el) * np.cos(ray_az),
                np.sin(ray_el)
            ])
            direction = direction / np.linalg.norm(direction)

            for target in point_targets:
                to_target = np.array(target.center) - radar_pos
                dist = np.linalg.norm(to_target)

                if dist < blind_range or dist > max_range:
                    continue

                target_dir = to_target / dist
                dot = np.dot(direction, target_dir)
                angle = np.arccos(np.clip(dot, -1, 1))
                target_angular_size = np.arctan(target.width_m / dist)

                if angle < target_angular_size + np.radians(0.5):
                    rcs = 10 ** (target.rcs_dbsm / 10)
                    received_power = rcs / (dist**4 + 1)

                    delay_samples = int(2 * dist / C * sample_rate)
                    n_samples = len(tx_waveform) + delay_samples + 100

                    rx_signal = np.zeros(n_samples, dtype=complex)
                    if delay_samples + len(tx_waveform) < n_samples:
                        phase = 4 * np.pi * wf_config.center_frequency_hz * dist / C
                        rx_signal[delay_samples:delay_samples + len(tx_waveform)] = (
                            tx_waveform * np.sqrt(received_power) * np.exp(1j * phase)
                        )

                    compressed = matched_filter_fft_windowed(rx_signal, tx_waveform, "hamming")

                    for j, val in enumerate(np.abs(compressed)):
                        range_m = j * C / (2 * sample_rate)
                        range_bin = int(range_m / range_resolution)
                        if 0 <= range_bin < n_range_bins and val > 1e-12:
                            ppi[az_idx, range_bin] += val

    if ppi.max() > 0:
        ppi = ppi / ppi.max()

    ppi_spread = quick_beam_spread(ppi, beamwidth_deg, n_azimuths, range_sigma_bins=1.0)

    return azimuths, ranges[:n_range_bins], ppi_spread


def main():
    print("=" * 70)
    print("  TARGET SHADOW / TRAILING EDGE DEMONSTRATION")
    print("=" * 70)
    print("\nThis simulation shows realistic asymmetric target response:")
    print("  - Sharp leading edge (front surface facing radar)")
    print("  - Extended trailing shadow (back surface + multipath + diffraction)")

    targets = create_test_targets()

    print(f"\nTargets:")
    for i, t in enumerate(targets):
        print(f"  {i+1}. Depth={t.depth_m}m, Trail={t.trailing_extent}x = "
              f"~{t.depth_m * t.trailing_extent:.0f}m shadow")

    # Simulation parameters
    pulse_width_us = 0.5
    bandwidth_mhz = 35.0
    n_azimuths = 180
    n_rays_per_az = 150
    max_range = 500

    print(f"\nSimulation: {pulse_width_us}μs pulse, {bandwidth_mhz}MHz BW")

    # Run point target (symmetric) simulation
    print(f"\n[1/2] Running point target simulation (symmetric)...")
    results_sym = run_point_target_simulation(
        targets, pulse_width_us, bandwidth_mhz,
        n_azimuths, n_rays_per_az, max_range
    )

    # Run extended target (shadow) simulation
    print(f"\n[2/2] Running extended target simulation (with shadow)...")
    results_asym = run_shadow_simulation(
        targets, pulse_width_us, bandwidth_mhz,
        n_azimuths, n_rays_per_az, max_range
    )

    # Plot comparison
    print(f"\nGenerating comparison plots...")
    plot_shadow_comparison(results_sym, results_asym, targets)

    print(f"\n" + "=" * 70)
    print("  COMPLETE!")
    print("=" * 70)
    print("\nKey observations:")
    print("  - Leading edge is sharp (front surface)")
    print("  - Trailing shadow extends behind target")
    print("  - Deeper targets have longer shadows")
    print("  - This matches real radar behavior!")


if __name__ == "__main__":
    main()
