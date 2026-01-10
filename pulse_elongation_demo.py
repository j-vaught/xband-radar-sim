#!/usr/bin/env python3
"""
Pulse Elongation Demonstration
Shows how longer pulses cause targets to appear elongated in range.
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
from typing import Tuple

from signal.waveform import WaveformConfig, generate_lfm_chirp
from signal.matched_filter import matched_filter_fft_windowed
from signal.ppi_processing import quick_beam_spread

# Pulse lengths to compare
# Blind zones: 1μs=150m, 5μs=750m, 10μs=1.5km, 20μs=3km
PULSE_LENGTHS_US = [1, 5, 10, 20]
BANDWIDTH_HZ = 50e6  # 50 MHz constant bandwidth

# Scene parameters
MAX_RANGE = 15000  # 15 km
C = 3e8  # Speed of light

# Simulation parameters
N_AZIMUTHS = 180
N_RAYS_PER_AZ = 400
BEAMWIDTH_DEG = 2.0


@dataclass
class SimpleTarget:
    """Simple point target for demonstration."""
    center: Tuple[float, float, float]
    size: float
    rcs_dbsm: float


def create_scene():
    """Create scene with targets at various ranges."""
    targets = []

    # Target 1: Large reflector at 5km, 45 degrees (NE)
    r1, az1 = 5000, np.radians(45)
    x1, y1 = r1 * np.sin(az1), r1 * np.cos(az1)
    targets.append(SimpleTarget(center=(x1, y1, 30), size=15.0, rcs_dbsm=30))

    # Target 2: Reflector at 5km, 135 degrees (SE)
    r2, az2 = 5000, np.radians(135)
    x2, y2 = r2 * np.sin(az2), r2 * np.cos(az2)
    targets.append(SimpleTarget(center=(x2, y2, 25), size=12.0, rcs_dbsm=25))

    # Target 3: At 7km, 225 degrees (SW)
    r3, az3 = 7000, np.radians(225)
    x3, y3 = r3 * np.sin(az3), r3 * np.cos(az3)
    targets.append(SimpleTarget(center=(x3, y3, 20), size=15.0, rcs_dbsm=28))

    # Target 4: At 4km, 315 degrees (NW)
    r4, az4 = 4000, np.radians(315)
    x4, y4 = r4 * np.sin(az4), r4 * np.cos(az4)
    targets.append(SimpleTarget(center=(x4, y4, 15), size=10.0, rcs_dbsm=22))

    return targets


def run_simulation(pulse_length_us, targets):
    """Run radar simulation for a specific pulse length."""
    pulse_length_s = pulse_length_us * 1e-6

    range_resolution = C / (2 * BANDWIDTH_HZ)
    blind_range = C * pulse_length_s / 2
    tb_product = pulse_length_s * BANDWIDTH_HZ

    print(f"\n  Pulse: {pulse_length_us} μs")
    print(f"  Time-Bandwidth: {tb_product:.0f}")
    print(f"  Blind range: {blind_range:.0f} m")

    # Waveform config
    sample_rate = BANDWIDTH_HZ * 2
    waveform_cfg = WaveformConfig(
        center_frequency_hz=9.4e9,
        bandwidth_hz=BANDWIDTH_HZ,
        pulse_width_s=pulse_length_s,
        sample_rate_hz=sample_rate
    )

    # Generate reference chirp
    _, ref_chirp = generate_lfm_chirp(waveform_cfg)

    # Setup PPI
    azimuths_deg = np.linspace(0, 360, N_AZIMUTHS, endpoint=False)
    n_range_bins = int(2 * MAX_RANGE / range_resolution)
    ppi = np.zeros((N_AZIMUTHS, n_range_bins))

    for i, az_deg in enumerate(azimuths_deg):
        if i % 30 == 0:
            print(f"    Azimuth {i}/{N_AZIMUTHS}")

        az_rad = np.radians(az_deg)

        # Fire rays within beam
        for _ in range(N_RAYS_PER_AZ):
            az_off = np.random.normal(0, BEAMWIDTH_DEG / 3)
            el_off = np.random.normal(0, 10 / 3)

            ray_az = az_rad + np.radians(az_off)
            ray_el = np.radians(el_off)

            # Ray direction
            dx = np.cos(ray_el) * np.sin(ray_az)
            dy = np.cos(ray_el) * np.cos(ray_az)
            dz = np.sin(ray_el)

            origin = np.array([0, 0, 30])
            direction = np.array([dx, dy, dz])
            direction = direction / np.linalg.norm(direction)

            # Check for target hits
            for target in targets:
                to_target = np.array(target.center) - origin
                dist = np.linalg.norm(to_target)

                if dist > MAX_RANGE or dist < blind_range:
                    continue

                # Check if ray points at target
                target_dir = to_target / dist
                dot = np.dot(direction, target_dir)
                angle = np.arccos(np.clip(dot, -1, 1))

                # Target angular size
                target_angular_size = np.arctan(target.size / dist)

                if angle < target_angular_size + np.radians(0.5):
                    # Hit - generate echo
                    rcs = 10 ** (target.rcs_dbsm / 10)
                    received_power = rcs / (dist ** 4 + 1)

                    # Generate signal with delay
                    delay_samples = int(2 * dist / C * sample_rate)
                    n_samples = len(ref_chirp) + delay_samples + 100

                    rx_signal = np.zeros(n_samples, dtype=complex)
                    if delay_samples + len(ref_chirp) <= n_samples:
                        rx_signal[delay_samples:delay_samples + len(ref_chirp)] = (
                            ref_chirp * np.sqrt(received_power)
                        )

                    # Matched filter
                    compressed = matched_filter_fft_windowed(rx_signal, ref_chirp)

                    # Map to range bins
                    range_per_sample = C / (2 * sample_rate)
                    for j, val in enumerate(np.abs(compressed)):
                        range_m = j * range_per_sample
                        range_bin = int(range_m / MAX_RANGE * n_range_bins)
                        if 0 <= range_bin < n_range_bins and val > 1e-10:
                            ppi[i, range_bin] += val

    # Normalize
    if ppi.max() > 0:
        ppi = ppi / ppi.max()

    # Apply beam spreading
    ppi_spread = quick_beam_spread(ppi, BEAMWIDTH_DEG, N_AZIMUTHS, range_sigma_bins=1.5)

    return ppi_spread, blind_range, tb_product


def plot_results(results):
    """Create comparison plots."""

    # Figure 1: Full PPI comparison
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 14), subplot_kw={'projection': 'polar'})
    fig1.suptitle('Pulse Length Effect on Target Appearance\n(Bandwidth = 50 MHz, Range Res = 3.0m)',
                  fontsize=14, fontweight='bold')

    for idx, (pulse_us, ppi, blind_range, tb) in enumerate(results):
        ax = axes1[idx // 2, idx % 2]

        azimuths = np.linspace(0, 2 * np.pi, ppi.shape[0], endpoint=False)
        ranges = np.linspace(0, MAX_RANGE, ppi.shape[1])

        r_mesh, az_mesh = np.meshgrid(ranges, azimuths)

        ppi_plot = ppi.copy()
        ppi_plot[ppi_plot < 1e-6] = 1e-6

        c = ax.pcolormesh(az_mesh, r_mesh, ppi_plot,
                          norm=LogNorm(vmin=1e-4, vmax=1),
                          cmap='hot', shading='auto')

        theta_circle = np.linspace(0, 2 * np.pi, 100)
        ax.fill_between(theta_circle, 0, blind_range, alpha=0.4, color='cyan')

        ax.set_title(f'{pulse_us} μs (TB={int(tb)})\nBlind zone = {blind_range/1000:.1f} km',
                     fontsize=11)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_ylim(0, MAX_RANGE)

        for r in [5000, 10000]:
            ax.plot(theta_circle, [r] * 100, 'w-', alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    fig1.savefig('pulse_elongation_full.png', dpi=150, bbox_inches='tight',
                 facecolor='black', edgecolor='none')
    print("\nSaved: pulse_elongation_full.png")

    # Figure 2: Zoomed Cartesian view of NE target
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 14))
    fig2.suptitle('Zoomed View: Target Elongation in Range (NE Target at 5km)',
                  fontsize=14, fontweight='bold')

    for idx, (pulse_us, ppi, blind_range, tb) in enumerate(results):
        ax = axes2[idx // 2, idx % 2]

        azimuths = np.linspace(0, 2 * np.pi, ppi.shape[0], endpoint=False)
        ranges = np.linspace(0, MAX_RANGE, ppi.shape[1])

        target_x = 5000 * np.sin(np.radians(45))
        target_y = 5000 * np.cos(np.radians(45))

        zoom_size = 400
        x_grid = np.linspace(target_x - zoom_size, target_x + zoom_size, 150)
        y_grid = np.linspace(target_y - zoom_size, target_y + zoom_size, 150)
        X, Y = np.meshgrid(x_grid, y_grid)

        R_grid = np.sqrt(X ** 2 + Y ** 2)
        Az_grid = np.arctan2(X, Y)
        Az_grid[Az_grid < 0] += 2 * np.pi

        from scipy.interpolate import RegularGridInterpolator

        interp = RegularGridInterpolator(
            (azimuths, ranges), ppi,
            method='linear', bounds_error=False, fill_value=0
        )

        points = np.stack([Az_grid.ravel(), R_grid.ravel()], axis=-1)
        cart_img = interp(points).reshape(X.shape)

        cart_img[cart_img < 1e-6] = 1e-6
        c = ax.pcolormesh(x_grid - target_x, y_grid - target_y, cart_img,
                          norm=LogNorm(vmin=1e-4, vmax=1),
                          cmap='hot', shading='auto')

        ax.axhline(0, color='cyan', alpha=0.5, linestyle='--', linewidth=1)
        ax.axvline(0, color='cyan', alpha=0.5, linestyle='--', linewidth=1)
        ax.plot(0, 0, 'c+', markersize=20, markeredgewidth=2)

        ax.annotate('Range', xy=(0, 200), xytext=(0, 350),
                    arrowprops=dict(arrowstyle='->', color='lime'),
                    color='lime', fontsize=10, ha='center')
        ax.annotate('Azimuth', xy=(200, 0), xytext=(350, 0),
                    arrowprops=dict(arrowstyle='->', color='yellow'),
                    color='yellow', fontsize=10, ha='center', va='center')

        ax.set_title(f'{pulse_us} μs (TB={int(tb)})', fontsize=12)
        ax.set_xlabel('Cross-range (m)', fontsize=10)
        ax.set_ylabel('Down-range (m)', fontsize=10)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, color='white')
        ax.set_facecolor('black')

        plt.colorbar(c, ax=ax, label='Intensity')

    plt.tight_layout()
    fig2.savefig('pulse_elongation_zoomed.png', dpi=150, bbox_inches='tight')
    print("Saved: pulse_elongation_zoomed.png")

    # Figure 3: Range profile comparison
    fig3, ax3 = plt.subplots(figsize=(12, 6))

    colors = ['#00ff00', '#00ffff', '#ffff00', '#ff6600']
    az_idx = int(45 / 360 * N_AZIMUTHS)

    for idx, (pulse_us, ppi, blind_range, tb) in enumerate(results):
        ranges_km = np.linspace(0, MAX_RANGE / 1000, ppi.shape[1])

        profile = np.mean(ppi[max(0, az_idx - 2):az_idx + 3, :], axis=0)
        if profile.max() > 0:
            profile = profile / profile.max()

        ax3.plot(ranges_km, profile, color=colors[idx],
                 linewidth=2, label=f'{pulse_us} μs (TB={int(tb)})', alpha=0.9)

        ax3.axvline(blind_range / 1000, color=colors[idx], linestyle=':', alpha=0.4)

    ax3.axvline(5.0, color='white', linestyle='--', alpha=0.7, label='Target (5 km)')

    ax3.set_xlabel('Range (km)', fontsize=12)
    ax3.set_ylabel('Normalized Intensity', fontsize=12)
    ax3.set_title('Range Profile at 45° Azimuth - Target Extent vs Pulse Length', fontsize=14)
    ax3.legend(loc='upper right', fontsize=10)
    ax3.set_xlim(4.5, 6.0)
    ax3.set_ylim(0, 1.1)
    ax3.grid(True, alpha=0.3)
    ax3.set_facecolor('#1a1a1a')

    plt.tight_layout()
    fig3.savefig('pulse_elongation_range_profile.png', dpi=150, bbox_inches='tight',
                 facecolor='#1a1a1a')
    print("Saved: pulse_elongation_range_profile.png")

    plt.close('all')


def main():
    print("=" * 60)
    print("PULSE ELONGATION DEMONSTRATION")
    print("=" * 60)
    print(f"Bandwidth: {BANDWIDTH_HZ / 1e6:.0f} MHz")
    print(f"Range resolution: {C / (2 * BANDWIDTH_HZ):.1f} m")
    print(f"Pulse lengths: {PULSE_LENGTHS_US} μs")

    print("\nCreating scene...")
    targets = create_scene()

    results = []
    for pulse_us in PULSE_LENGTHS_US:
        print(f"\nSimulating {pulse_us} μs pulse...")
        ppi, blind_range, tb = run_simulation(pulse_us, targets)
        results.append((pulse_us, ppi, blind_range, tb))

    print("\nGenerating plots...")
    plot_results(results)

    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
