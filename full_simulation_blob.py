#!/usr/bin/env python3
"""
Realistic Radar Simulation with Smooth Blob Rendering

Uses 2D kernel stamping instead of discrete scatterers to create
natural rounded blobs with smooth tails - no banding artifacts.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import time
from dataclasses import dataclass
from typing import List, Tuple
from scipy.ndimage import gaussian_filter

from config import RadarConfig
from signal.ppi_processing import normalize_for_display, polar_to_cartesian

C = 299792458.0


@dataclass
class PulseMode:
    name: str
    pulse_width_us: float
    bandwidth_mhz: float

    @property
    def pulse_width_s(self):
        return self.pulse_width_us * 1e-6

    @property
    def bandwidth_hz(self):
        return self.bandwidth_mhz * 1e6

    @property
    def range_resolution_m(self):
        return C / (2 * self.bandwidth_hz)

    @property
    def blind_range_m(self):
        return C * self.pulse_width_s / 2


PULSE_MODES = [
    PulseMode("Short", 0.08, 50.0),
    PulseMode("Medium", 0.5, 35.0),
    PulseMode("Long", 1.0, 25.0),
    PulseMode("X-Long", 2.0, 15.0),
]


@dataclass
class Target:
    name: str
    range_m: float
    azimuth_deg: float
    depth_m: float
    width_m: float
    rcs_dbsm: float
    tail_length_factor: float = 2.5  # Tail extends this many depths


def create_targets():
    return [
        Target("Building NE", 200, 45, 25, 30, 28, 2.5),
        Target("Target E1", 140, 90, 10, 8, 20, 1.5),
        Target("Target E2", 160, 90, 10, 8, 20, 1.5),
        Target("Tank W", 100, 270, 15, 20, 22, 2.0),
        Target("Warehouse S", 300, 180, 40, 25, 30, 3.0),
        Target("Structure NW", 250, 315, 20, 18, 25, 2.0),
        Target("Ship SW", 350, 225, 50, 12, 32, 3.5),
    ]


def create_blob_kernel(
    range_bins: int,
    az_bins: int,
    body_range_sigma: float,
    body_az_sigma: float,
    tail_length_bins: int,
    tail_decay: float = 0.15
) -> np.ndarray:
    """
    Create a smooth TEARDROP blob using polar coordinates internally.

    This creates a true comet/teardrop shape by defining the shape
    in polar coordinates relative to the blob center.
    """
    # Grid centered on blob peak
    peak_r = range_bins // 3
    r_coords = np.arange(range_bins) - peak_r
    az_coords = np.arange(az_bins) - az_bins // 2

    kernel = np.zeros((range_bins, az_bins))

    # Body size (semi-major/minor axes of ellipse)
    a = body_range_sigma * 1.5   # Range extent
    b = body_az_sigma * 1.5      # Azimuth extent

    for ri in range(range_bins):
        for ai in range(az_bins):
            x = r_coords[ri]   # Range direction (positive = away from radar)
            y = az_coords[ai]  # Azimuth direction

            # Convert to polar relative to blob center
            r = np.sqrt(x**2 + y**2)
            theta = np.arctan2(y, x)  # Angle from range axis

            # Teardrop shape: radius varies with angle
            # Front (theta near pi): smaller radius (rounder)
            # Back (theta near 0): larger radius (tail extends)

            # Cardioid-like shape for smooth teardrop
            # r_boundary defines the edge of the blob at each angle
            cos_theta = np.cos(theta)

            if cos_theta <= 0:
                # FRONT HALF: Elliptical shape
                # Smooth ellipse boundary
                r_boundary = a * b / np.sqrt((b * np.cos(theta))**2 + (a * np.sin(theta))**2)
                # Gaussian falloff from center
                if r_boundary > 0:
                    normalized_r = r / r_boundary
                    kernel[ri, ai] = np.exp(-2.0 * normalized_r**2)
            else:
                # BACK HALF (TAIL): Expanding with decay
                # Tail length increases, width expands
                tail_x = x  # How far into tail

                # Width expands linearly with tail distance
                local_width = b * (1.0 + tail_x / tail_length_bins * 2.5)

                # Amplitude decays exponentially
                amp = np.exp(-tail_decay * tail_x)

                # Gaussian profile across the width
                width_factor = np.exp(-0.5 * (y / local_width)**2)

                kernel[ri, ai] = amp * width_factor

    # Heavy smoothing for ultra-smooth curved edges
    kernel = gaussian_filter(kernel, sigma=[3.0, 2.5])

    if kernel.max() > 0:
        kernel = kernel / kernel.max()

    return kernel


def stamp_target(
    ppi: np.ndarray,
    target: Target,
    pulse_mode: PulseMode,
    range_per_bin: float,
    az_per_bin: float,
    beamwidth_deg: float
) -> np.ndarray:
    """Stamp a smooth blob onto the PPI at target location."""

    n_az, n_range = ppi.shape

    # Skip if in blind zone
    if target.range_m < pulse_mode.blind_range_m:
        return ppi

    # Target center in bins
    range_bin = int(target.range_m / range_per_bin)
    az_bin = int(target.azimuth_deg / az_per_bin) % n_az

    if range_bin >= n_range:
        return ppi

    # Kernel size based on target size and resolution
    body_range_bins = max(3, target.depth_m / range_per_bin)
    body_az_bins = max(3, (target.width_m / target.range_m) * (180 / np.pi) / az_per_bin)

    # Add beamwidth contribution
    beam_az_bins = beamwidth_deg / az_per_bin
    body_az_bins = np.sqrt(body_az_bins**2 + beam_az_bins**2)

    # Tail length
    tail_length_bins = int(target.depth_m * target.tail_length_factor / range_per_bin)

    # Kernel dimensions (enough to contain body + tail)
    kernel_range = int(body_range_bins * 2 + tail_length_bins + 10)
    kernel_az = int(body_az_bins * 6)

    # Create the smooth blob kernel
    kernel = create_blob_kernel(
        kernel_range, kernel_az,
        body_range_sigma=body_range_bins,
        body_az_sigma=body_az_bins,
        tail_length_bins=tail_length_bins,
        tail_decay=0.12
    )

    # Scale by RCS
    rcs_linear = 10 ** (target.rcs_dbsm / 10)
    amplitude = np.sqrt(rcs_linear) / (target.range_m**2 + 1)
    kernel = kernel * amplitude

    # Stamp onto PPI (handle wraparound in azimuth)
    kr, kaz = kernel.shape

    # Range indices
    r_start = range_bin - kr // 3  # Offset so peak is at target range
    r_end = r_start + kr

    # Clip to valid range
    kr_start = max(0, -r_start)
    kr_end = kr - max(0, r_end - n_range)
    r_start = max(0, r_start)
    r_end = min(n_range, r_end)

    if r_end <= r_start:
        return ppi

    # Azimuth indices (with wraparound)
    az_start = az_bin - kaz // 2

    for ki in range(kr_start, kr_end):
        ri = r_start + (ki - kr_start)
        for kj in range(kaz):
            aj = (az_start + kj) % n_az
            ppi[aj, ri] += kernel[ki, kj]

    return ppi


def run_simulation(targets: List[Target], pulse_mode: PulseMode,
                   n_azimuths: int = 360, max_range: float = 500.0,
                   beamwidth_deg: float = 3.9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run simulation by stamping smooth blobs for each target."""

    range_resolution = pulse_mode.range_resolution_m
    n_range_bins = int(max_range / range_resolution)

    range_per_bin = max_range / n_range_bins
    az_per_bin = 360.0 / n_azimuths

    azimuths = np.linspace(0, 360, n_azimuths, endpoint=False)
    ranges = np.linspace(0, max_range, n_range_bins)

    ppi = np.zeros((n_azimuths, n_range_bins))

    print(f"    Stamping {len(targets)} targets...")

    for target in targets:
        ppi = stamp_target(ppi, target, pulse_mode, range_per_bin, az_per_bin, beamwidth_deg)

    # Final smoothing for ultra-smooth blobs
    smooth_sigma_az = max(1.5, beamwidth_deg / az_per_bin / 3)
    smooth_sigma_r = max(1.5, range_resolution / range_per_bin)
    ppi = gaussian_filter(ppi, sigma=[smooth_sigma_az, smooth_sigma_r], mode='wrap')

    # Normalize
    if ppi.max() > 0:
        ppi = ppi / ppi.max()

    return azimuths, ranges, ppi


def plot_results(results: List[Tuple], output_prefix: str = "blob"):
    """Create comparison plots."""

    # Figure 1: Cartesian comparison
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 14))
    fig1.suptitle('Smooth Blob Rendering - Pulse Mode Comparison\n'
                  '(Rounded blobs with expanding tails)',
                  fontsize=14, fontweight='bold', color='white')

    for idx, (mode, azimuths, ranges, ppi) in enumerate(results):
        ax = axes1[idx // 2, idx % 2]

        cart, x, y = polar_to_cartesian(ppi, azimuths, ranges, output_size=500)
        # Apply HEAVY smoothing in Cartesian space for truly round blobs
        cart = gaussian_filter(cart, sigma=4.0)
        cart_norm = normalize_for_display(cart, 50.0)

        extent = [x[0], x[-1], y[0], y[-1]]
        ax.imshow(cart_norm, cmap='viridis', origin='lower', extent=extent, aspect='equal')

        ax.set_facecolor('black')

        # Range rings
        for r in [100, 200, 300, 400]:
            theta = np.linspace(0, 2*np.pi, 100)
            ax.plot(r * np.cos(theta), r * np.sin(theta), 'g-', alpha=0.3, lw=0.5)

        # Blind zone
        blind_theta = np.linspace(0, 2*np.pi, 100)
        ax.fill(mode.blind_range_m * np.cos(blind_theta),
                mode.blind_range_m * np.sin(blind_theta),
                alpha=0.3, color='red')

        ax.set_title(f'{mode.name}: Res={mode.range_resolution_m:.1f}m, '
                    f'Blind={mode.blind_range_m:.0f}m', color='lime', fontsize=11)
        ax.tick_params(colors='green')
        ax.set_xlabel('East-West (m)', color='green')
        ax.set_ylabel('North-South (m)', color='green')

    fig1.patch.set_facecolor('black')
    plt.tight_layout()
    fig1.savefig(f'{output_prefix}_cartesian.png', dpi=150, facecolor='black', bbox_inches='tight')
    print(f"  Saved: {output_prefix}_cartesian.png")
    plt.close()

    # Figure 2: Zoomed view of Ship SW
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 12))
    fig2.suptitle('Zoomed View: Ship SW - Smooth Blob Shape',
                  fontsize=14, fontweight='bold', color='white')

    ship_range = 350
    ship_az = 225
    ship_x = ship_range * np.sin(np.radians(ship_az))
    ship_y = ship_range * np.cos(np.radians(ship_az))

    for idx, (mode, azimuths, ranges, ppi) in enumerate(results):
        ax = axes2[idx // 2, idx % 2]

        cart, x, y = polar_to_cartesian(ppi, azimuths, ranges, output_size=500)
        # Heavy Cartesian smoothing for round blobs
        cart = gaussian_filter(cart, sigma=4.0)

        zoom = 120
        x_idx = np.argmin(np.abs(x - ship_x))
        y_idx = np.argmin(np.abs(y - ship_y))
        zoom_px = int(zoom / (x[-1] - x[0]) * len(x))

        x_start = max(0, x_idx - zoom_px)
        x_end = min(len(x), x_idx + zoom_px)
        y_start = max(0, y_idx - zoom_px)
        y_end = min(len(y), y_idx + zoom_px)

        zoomed = cart[y_start:y_end, x_start:x_end]
        zoomed_norm = normalize_for_display(zoomed, 50.0)

        extent = [x[x_start] - ship_x, x[min(x_end, len(x)-1)] - ship_x,
                  y[y_start] - ship_y, y[min(y_end, len(y)-1)] - ship_y]

        ax.imshow(zoomed_norm, cmap='hot', origin='lower', extent=extent, aspect='equal')

        ax.axhline(0, color='cyan', linestyle='--', alpha=0.3)
        ax.axvline(0, color='cyan', linestyle='--', alpha=0.3)
        ax.plot(0, 0, 'c+', markersize=15, markeredgewidth=2)

        ax.annotate('← Radar', xy=(-80, -90), color='lime', fontsize=9)
        ax.annotate('Tail →', xy=(40, -90), color='red', fontsize=9)

        ax.set_xlabel('Range offset (m)', color='white')
        ax.set_ylabel('Cross-range (m)', color='white')
        ax.set_title(f'{mode.name}: Res={mode.range_resolution_m:.1f}m', color='lime', fontsize=11)
        ax.set_facecolor('black')
        ax.tick_params(colors='white')

    fig2.patch.set_facecolor('black')
    plt.tight_layout()
    fig2.savefig(f'{output_prefix}_zoomed.png', dpi=150, facecolor='black', bbox_inches='tight')
    print(f"  Saved: {output_prefix}_zoomed.png")
    plt.close()

    # Figure 3: Polar view
    fig3, axes3 = plt.subplots(2, 2, figsize=(14, 14), subplot_kw={'projection': 'polar'})
    fig3.suptitle('Polar PPI - Smooth Blob Rendering',
                  fontsize=14, fontweight='bold', color='white')

    for idx, (mode, azimuths, ranges, ppi) in enumerate(results):
        ax = axes3[idx // 2, idx % 2]

        az_rad = np.deg2rad(azimuths)
        R, AZ = np.meshgrid(ranges, az_rad)

        ppi_norm = normalize_for_display(ppi, 50.0)

        ax.pcolormesh(AZ, R, ppi_norm, cmap='viridis', shading='auto')

        theta_circle = np.linspace(0, 2*np.pi, 100)
        ax.fill_between(theta_circle, 0, mode.blind_range_m, alpha=0.4, color='red')

        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_facecolor('black')

        for r in [100, 200, 300, 400]:
            if r < ranges[-1]:
                ax.plot(theta_circle, [r]*100, 'g-', alpha=0.3, lw=0.5)

        ax.set_title(f'{mode.name}\nRes={mode.range_resolution_m:.1f}m', color='lime', fontsize=10)
        ax.tick_params(colors='green')

    fig3.patch.set_facecolor('black')
    plt.tight_layout()
    fig3.savefig(f'{output_prefix}_polar.png', dpi=150, facecolor='black', bbox_inches='tight')
    print(f"  Saved: {output_prefix}_polar.png")
    plt.close()


def main():
    print("=" * 70)
    print("  SMOOTH BLOB RADAR SIMULATION")
    print("  (No banding, proper rounded blobs with tails)")
    print("=" * 70)

    print("\nPulse Modes:")
    for mode in PULSE_MODES:
        print(f"  {mode.name:8s}: Res={mode.range_resolution_m:.1f}m, "
              f"Blind={mode.blind_range_m:.0f}m")

    targets = create_targets()
    print(f"\nTargets: {len(targets)}")

    n_azimuths = 720  # Higher for smoother blobs
    max_range = 500.0

    results = []
    total_start = time.time()

    for i, mode in enumerate(PULSE_MODES):
        print(f"\n[{i+1}/{len(PULSE_MODES)}] {mode.name} pulse...")
        start = time.time()
        azimuths, ranges, ppi = run_simulation(targets, mode, n_azimuths, max_range)
        print(f"    Completed in {time.time() - start:.1f}s")
        results.append((mode, azimuths, ranges, ppi))

    print(f"\nGenerating plots...")
    plot_results(results, "blob_mode")

    print(f"\n" + "=" * 70)
    print(f"  COMPLETE! Total time: {time.time() - total_start:.1f}s")
    print("=" * 70)
    print("\nOutputs:")
    print("  blob_mode_cartesian.png - Cartesian view")
    print("  blob_mode_zoomed.png    - Zoomed ship view")
    print("  blob_mode_polar.png     - Polar PPI view")


if __name__ == "__main__":
    main()
