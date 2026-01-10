#!/usr/bin/env python3
"""
Pulse Compression Sidelobe Demonstration

Shows how longer pulses cause targets to appear elongated due to:
1. Range sidelobes from matched filter (-13.2dB for LFM)
2. Mainlobe widening when windowing is applied
3. Different effective bandwidths for different pulse modes

This simulates real marine radar behavior where longer pulses
result in "bigger diameter" returns as noted in radar manuals.
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

# Marine radar typical pulse configurations
# Based on Furuno and similar marine radars
PULSE_CONFIGS = [
    # (name, pulse_us, bandwidth_hz, description)
    ("Short", 0.08, 50e6, "Close range, best resolution"),
    ("Medium", 0.3, 40e6, "Mid range"),
    ("Long", 0.6, 30e6, "Long range, more energy"),
    ("X-Long", 1.2, 20e6, "Max range, reduced resolution"),
]

MAX_RANGE = 3000  # 3 km for close range demo
C = 3e8

N_AZIMUTHS = 180
N_RAYS_PER_AZ = 300
BEAMWIDTH_DEG = 3.9  # Furuno DRS4D-NXT spec


@dataclass
class SimpleTarget:
    center: Tuple[float, float, float]
    size: float
    rcs_dbsm: float


def create_scene():
    """Create scene with closely spaced targets to show resolution differences."""
    targets = []

    # Two close targets at 500m, 45 degrees - separated by 20m in range
    r_base, az = 500, np.radians(45)
    x1, y1 = r_base * np.sin(az), r_base * np.cos(az)
    targets.append(SimpleTarget(center=(x1, y1, 10), size=5.0, rcs_dbsm=20))

    # Second target 20m further in range
    r2 = r_base + 20
    x2, y2 = r2 * np.sin(az), r2 * np.cos(az)
    targets.append(SimpleTarget(center=(x2, y2, 10), size=5.0, rcs_dbsm=20))

    # Single strong target at 800m, 135 degrees
    r3, az3 = 800, np.radians(135)
    x3, y3 = r3 * np.sin(az3), r3 * np.cos(az3)
    targets.append(SimpleTarget(center=(x3, y3, 15), size=8.0, rcs_dbsm=30))

    # Target at 1200m, 225 degrees
    r4, az4 = 1200, np.radians(225)
    x4, y4 = r4 * np.sin(az4), r4 * np.cos(az4)
    targets.append(SimpleTarget(center=(x4, y4, 12), size=6.0, rcs_dbsm=25))

    # Target at 600m, 315 degrees
    r5, az5 = 600, np.radians(315)
    x5, y5 = r5 * np.sin(az5), r5 * np.cos(az5)
    targets.append(SimpleTarget(center=(x5, y5, 8), size=4.0, rcs_dbsm=18))

    return targets


def generate_lfm_chirp(pulse_width_s, bandwidth_hz, sample_rate_hz):
    """Generate LFM chirp waveform."""
    n_samples = int(pulse_width_s * sample_rate_hz)
    t = np.arange(n_samples) / sample_rate_hz
    K = bandwidth_hz / pulse_width_s  # Chirp rate
    phase = np.pi * K * t ** 2
    return np.exp(1j * phase)


def matched_filter_with_sidelobes(rx_signal, ref_chirp, apply_window=True):
    """
    Apply matched filter, optionally with windowing.

    Without window: -13.2dB sidelobes (theoretical LFM)
    With Hamming window: ~-42dB sidelobes but wider mainlobe
    """
    n_fft = len(rx_signal) + len(ref_chirp) - 1
    n_fft = int(2 ** np.ceil(np.log2(n_fft)))

    # Apply window to reduce sidelobes (but widens mainlobe)
    if apply_window:
        window = np.hamming(len(ref_chirp))
        ref_windowed = ref_chirp * window
    else:
        ref_windowed = ref_chirp

    # FFT correlation
    RX = np.fft.fft(rx_signal, n_fft)
    REF = np.fft.fft(np.conj(ref_windowed[::-1]), n_fft)
    compressed = np.fft.ifft(RX * REF)

    return compressed[:len(rx_signal)]


def run_simulation(pulse_name, pulse_width_us, bandwidth_hz, targets, apply_window=True):
    """Run radar simulation for a specific pulse configuration."""
    pulse_width_s = pulse_width_us * 1e-6

    range_resolution = C / (2 * bandwidth_hz)
    blind_range = C * pulse_width_s / 2
    tb_product = pulse_width_s * bandwidth_hz

    print(f"\n  {pulse_name} Pulse: {pulse_width_us} μs, BW: {bandwidth_hz/1e6:.0f} MHz")
    print(f"  Range resolution: {range_resolution:.2f} m")
    print(f"  Time-Bandwidth: {tb_product:.0f}")
    print(f"  Blind range: {blind_range:.1f} m")

    sample_rate = bandwidth_hz * 2.5  # Oversample for accuracy
    ref_chirp = generate_lfm_chirp(pulse_width_s, bandwidth_hz, sample_rate)

    azimuths_deg = np.linspace(0, 360, N_AZIMUTHS, endpoint=False)
    n_range_bins = int(2 * MAX_RANGE / range_resolution)
    ppi = np.zeros((N_AZIMUTHS, n_range_bins))

    for i, az_deg in enumerate(azimuths_deg):
        if i % 45 == 0:
            print(f"    Azimuth {i}/{N_AZIMUTHS}")

        az_rad = np.radians(az_deg)

        for _ in range(N_RAYS_PER_AZ):
            az_off = np.random.normal(0, BEAMWIDTH_DEG / 3)
            el_off = np.random.normal(0, 12 / 3)

            ray_az = az_rad + np.radians(az_off)
            ray_el = np.radians(el_off)

            dx = np.cos(ray_el) * np.sin(ray_az)
            dy = np.cos(ray_el) * np.cos(ray_az)
            dz = np.sin(ray_el)

            origin = np.array([0, 0, 20])
            direction = np.array([dx, dy, dz])
            direction = direction / np.linalg.norm(direction)

            for target in targets:
                to_target = np.array(target.center) - origin
                dist = np.linalg.norm(to_target)

                if dist > MAX_RANGE or dist < blind_range:
                    continue

                target_dir = to_target / dist
                dot = np.dot(direction, target_dir)
                angle = np.arccos(np.clip(dot, -1, 1))

                target_angular_size = np.arctan(target.size / dist)

                if angle < target_angular_size + np.radians(0.3):
                    rcs = 10 ** (target.rcs_dbsm / 10)
                    received_power = rcs / (dist ** 4 + 1)

                    delay_samples = int(2 * dist / C * sample_rate)
                    n_samples = len(ref_chirp) + delay_samples + 200

                    rx_signal = np.zeros(n_samples, dtype=complex)
                    if delay_samples + len(ref_chirp) <= n_samples:
                        rx_signal[delay_samples:delay_samples + len(ref_chirp)] = (
                            ref_chirp * np.sqrt(received_power)
                        )

                    # Apply matched filter WITH sidelobes
                    compressed = matched_filter_with_sidelobes(
                        rx_signal, ref_chirp, apply_window=apply_window
                    )

                    range_per_sample = C / (2 * sample_rate)
                    for j, val in enumerate(np.abs(compressed)):
                        range_m = j * range_per_sample
                        range_bin = int(range_m / MAX_RANGE * n_range_bins)
                        if 0 <= range_bin < n_range_bins and val > 1e-12:
                            ppi[i, range_bin] += val

    if ppi.max() > 0:
        ppi = ppi / ppi.max()

    # Apply beam spreading (azimuth convolution)
    from signal.ppi_processing import quick_beam_spread
    ppi_spread = quick_beam_spread(ppi, BEAMWIDTH_DEG, N_AZIMUTHS, range_sigma_bins=1.0)

    return ppi_spread, blind_range, range_resolution, tb_product


def plot_results(results):
    """Create comparison plots."""

    # Figure 1: Full PPI comparison (2x2)
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 14), subplot_kw={'projection': 'polar'})
    fig1.suptitle('Marine Radar Pulse Mode Comparison\n(Short/Long Pulse Effect on Target Size)',
                  fontsize=14, fontweight='bold')

    for idx, (name, ppi, blind, res, tb, bw) in enumerate(results):
        ax = axes1[idx // 2, idx % 2]

        azimuths = np.linspace(0, 2 * np.pi, ppi.shape[0], endpoint=False)
        ranges = np.linspace(0, MAX_RANGE, ppi.shape[1])

        r_mesh, az_mesh = np.meshgrid(ranges, azimuths)

        ppi_plot = ppi.copy()
        ppi_plot[ppi_plot < 1e-6] = 1e-6

        c = ax.pcolormesh(az_mesh, r_mesh, ppi_plot,
                          norm=LogNorm(vmin=1e-3, vmax=1),
                          cmap='hot', shading='auto')

        theta_circle = np.linspace(0, 2 * np.pi, 100)
        ax.fill_between(theta_circle, 0, blind, alpha=0.4, color='cyan')

        ax.set_title(f'{name}\nRes={res:.1f}m, TB={int(tb)}, BW={bw/1e6:.0f}MHz',
                     fontsize=11)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_ylim(0, MAX_RANGE)

        for r in [500, 1000, 1500, 2000, 2500]:
            ax.plot(theta_circle, [r] * 100, 'w-', alpha=0.2, linewidth=0.5)

    plt.tight_layout()
    fig1.savefig('pulse_mode_comparison.png', dpi=150, bbox_inches='tight',
                 facecolor='black', edgecolor='none')
    print("\nSaved: pulse_mode_comparison.png")

    # Figure 2: Zoomed view of close targets at 45 degrees (500m)
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 12))
    fig2.suptitle('Zoomed View: Two Close Targets at 500m (20m separation)\nShowing Resolution & Sidelobe Effects',
                  fontsize=14, fontweight='bold')

    for idx, (name, ppi, blind, res, tb, bw) in enumerate(results):
        ax = axes2[idx // 2, idx % 2]

        azimuths = np.linspace(0, 2 * np.pi, ppi.shape[0], endpoint=False)
        ranges = np.linspace(0, MAX_RANGE, ppi.shape[1])

        # Convert to Cartesian centered on targets
        target_range = 510  # Between the two targets
        target_x = target_range * np.sin(np.radians(45))
        target_y = target_range * np.cos(np.radians(45))

        zoom_size = 100
        x_grid = np.linspace(target_x - zoom_size, target_x + zoom_size, 200)
        y_grid = np.linspace(target_y - zoom_size, target_y + zoom_size, 200)
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
                          norm=LogNorm(vmin=1e-3, vmax=1),
                          cmap='hot', shading='auto')

        # Mark true target positions
        t1_offset = -10  # First target 10m closer
        t2_offset = 10   # Second target 10m further
        ax.plot(0, t1_offset * np.sqrt(2) / 2, 'g+', markersize=15, markeredgewidth=2)
        ax.plot(0, t2_offset * np.sqrt(2) / 2, 'g+', markersize=15, markeredgewidth=2)

        ax.annotate('Range\n(to radar)', xy=(0, -80), xytext=(0, -60),
                    color='lime', fontsize=9, ha='center')
        ax.annotate('', xy=(0, -80), xytext=(0, 80),
                    arrowprops=dict(arrowstyle='<->', color='lime', lw=1.5))

        ax.set_title(f'{name}: Res={res:.1f}m, BW={bw/1e6:.0f}MHz', fontsize=11)
        ax.set_xlabel('Cross-range (m)', fontsize=10)
        ax.set_ylabel('Down-range (m)', fontsize=10)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, color='white')
        ax.set_facecolor('black')

        plt.colorbar(c, ax=ax, label='Intensity', shrink=0.8)

    plt.tight_layout()
    fig2.savefig('pulse_mode_zoomed.png', dpi=150, bbox_inches='tight')
    print("Saved: pulse_mode_zoomed.png")

    # Figure 3: Range profile comparison showing resolution
    fig3, ax3 = plt.subplots(figsize=(14, 6))

    colors = ['#00ff00', '#00ffff', '#ffff00', '#ff6600']
    az_idx = int(45 / 360 * N_AZIMUTHS)

    for idx, (name, ppi, blind, res, tb, bw) in enumerate(results):
        ranges_m = np.linspace(0, MAX_RANGE, ppi.shape[1])

        profile = np.mean(ppi[max(0, az_idx - 2):az_idx + 3, :], axis=0)
        if profile.max() > 0:
            profile = profile / profile.max()

        ax3.plot(ranges_m, profile, color=colors[idx],
                 linewidth=2, label=f'{name} (Res={res:.1f}m)', alpha=0.9)

    # Mark target positions
    ax3.axvline(500, color='white', linestyle='--', alpha=0.5, label='Target 1 (500m)')
    ax3.axvline(520, color='gray', linestyle='--', alpha=0.5, label='Target 2 (520m)')

    ax3.set_xlabel('Range (m)', fontsize=12)
    ax3.set_ylabel('Normalized Intensity', fontsize=12)
    ax3.set_title('Range Profile at 45° - Two Targets 20m Apart\n(Can you resolve them?)', fontsize=14)
    ax3.legend(loc='upper right', fontsize=10)
    ax3.set_xlim(400, 650)
    ax3.set_ylim(0, 1.1)
    ax3.grid(True, alpha=0.3)
    ax3.set_facecolor('#1a1a1a')

    plt.tight_layout()
    fig3.savefig('pulse_mode_range_profile.png', dpi=150, bbox_inches='tight',
                 facecolor='#1a1a1a')
    print("Saved: pulse_mode_range_profile.png")

    plt.close('all')


def main():
    print("=" * 60)
    print("MARINE RADAR PULSE MODE COMPARISON")
    print("Showing how longer pulses cause bigger/elongated returns")
    print("=" * 60)

    print("\nCreating scene with close targets...")
    targets = create_scene()

    results = []
    for name, pulse_us, bw, desc in PULSE_CONFIGS:
        print(f"\nSimulating {name} pulse ({desc})...")
        ppi, blind, res, tb = run_simulation(name, pulse_us, bw, targets)
        results.append((name, ppi, blind, res, tb, bw))

    print("\nGenerating plots...")
    plot_results(results)

    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print("\nKey observations:")
    print("- Short pulse: Better resolution, targets appear smaller")
    print("- Long pulse: Worse resolution, targets appear bigger/elongated")
    print("- The 20m-separated targets may merge with long pulse")


if __name__ == "__main__":
    main()
