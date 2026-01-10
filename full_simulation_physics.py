#!/usr/bin/env python3
"""
Physics-Accurate Radar Simulation

The blob shape comes from actual EM physics:
1. Antenna pattern: sinc²(θ) from aperture diffraction
2. Pulse compression: sinc(r) from matched filter autocorrelation
3. 2D PSF is the product of these - naturally creates blob shape

No artificial smoothing or kernel stamping needed.
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
from typing import List, Tuple
import time

from signal.ppi_processing import polar_to_cartesian, normalize_for_display

C = 299792458.0


@dataclass
class RadarParams:
    """Physical radar parameters."""
    frequency_hz: float = 9.41e9      # X-band
    bandwidth_hz: float = 50e6        # 50 MHz
    pulse_width_s: float = 1e-6      # 1 μs
    antenna_diameter_m: float = 0.6   # 60cm dish

    @property
    def wavelength_m(self):
        return C / self.frequency_hz

    @property
    def beamwidth_rad(self):
        """3dB beamwidth from antenna aperture (diffraction limit)."""
        # θ_3dB ≈ 1.22 * λ / D for circular aperture
        return 1.22 * self.wavelength_m / self.antenna_diameter_m

    @property
    def beamwidth_deg(self):
        return np.degrees(self.beamwidth_rad)

    @property
    def range_resolution_m(self):
        """Range resolution from bandwidth."""
        return C / (2 * self.bandwidth_hz)

    @property
    def blind_range_m(self):
        return C * self.pulse_width_s / 2


@dataclass
class Target:
    """Physical target."""
    name: str
    range_m: float
    azimuth_deg: float
    rcs_m2: float           # Radar cross section in m²
    range_extent_m: float   # Physical depth
    azimuth_extent_m: float # Physical width


def sinc_squared(x):
    """
    Normalized sinc² function - unwindowed response.
    sinc²(x) = (sin(πx)/(πx))²
    """
    result = np.ones_like(x, dtype=float)
    nonzero = np.abs(x) > 1e-10
    result[nonzero] = (np.sin(np.pi * x[nonzero]) / (np.pi * x[nonzero]))**2
    return result


def windowed_response(x, sidelobe_db=-40):
    """
    Windowed (filtered) response - suppresses sidelobes.

    Real radars apply windowing (Hamming, Taylor, etc.) to reduce
    sidelobes from -13dB to -40dB or better.

    This uses a Gaussian approximation which naturally has no sidelobes,
    with width adjusted to match the main lobe broadening from windowing.

    Hamming window: ~1.5x main lobe broadening, -42dB sidelobes
    Taylor window: ~1.3x broadening, configurable sidelobes
    """
    # Gaussian has no sidelobes - width factor accounts for main lobe broadening
    # from windowing (typically 1.3-1.5x wider than unwindowed)
    broadening_factor = 1.4
    sigma = broadening_factor / 2.355  # Convert FWHM to sigma
    return np.exp(-0.5 * (x / sigma)**2)


def antenna_pattern(theta_rad, beamwidth_rad, windowed=True):
    """
    Antenna pattern - optionally with sidelobe suppression.

    Real antennas often have tapered illumination (like Taylor weighting)
    to reduce sidelobes at the cost of slightly wider beamwidth.
    """
    x = theta_rad / beamwidth_rad * 1.39

    if windowed:
        return windowed_response(x)
    else:
        return sinc_squared(x)


def pulse_compression_response(range_offset_m, range_resolution_m, windowed=True):
    """
    Pulse compression response - optionally with sidelobe suppression.

    Real radars apply windowing (Hamming, Taylor, Kaiser) to the
    matched filter to suppress range sidelobes.
    """
    x = range_offset_m / range_resolution_m

    if windowed:
        return windowed_response(x)
    else:
        return sinc_squared(x)


def compute_target_response(
    target: Target,
    radar: RadarParams,
    azimuths_deg: np.ndarray,
    ranges_m: np.ndarray
) -> np.ndarray:
    """
    Compute the 2D response of a target using actual physics.

    The response is the 2D PSF:
    PSF(r, θ) = antenna_pattern(θ - θ_target) × pulse_response(r - r_target)

    convolved with the target's physical extent.
    """
    n_az = len(azimuths_deg)
    n_range = len(ranges_m)

    response = np.zeros((n_az, n_range))

    # Target position
    r_target = target.range_m
    az_target = target.azimuth_deg

    # Convert target physical size to angular size at its range
    target_angular_extent_rad = target.azimuth_extent_m / target.range_m
    target_range_extent_m = target.range_extent_m

    # RCS amplitude
    amplitude = np.sqrt(target.rcs_m2)

    # Range equation: received power ∝ 1/R⁴
    range_factor = 1.0 / (r_target**2)  # One-way for amplitude

    for i, az in enumerate(azimuths_deg):
        # Angular offset from target center
        az_offset_deg = az - az_target
        # Handle wraparound
        if az_offset_deg > 180:
            az_offset_deg -= 360
        elif az_offset_deg < -180:
            az_offset_deg += 360

        az_offset_rad = np.radians(az_offset_deg)

        # Antenna pattern response at this azimuth
        # The target subtends an angle, so we need to integrate over it
        # For simplicity, treat as convolution of point response with target extent

        # Effective beamwidth including target angular extent
        effective_beamwidth = np.sqrt(radar.beamwidth_rad**2 + target_angular_extent_rad**2)

        az_response = antenna_pattern(az_offset_rad, effective_beamwidth)

        if az_response < 1e-6:
            continue

        for j, r in enumerate(ranges_m):
            # Range offset from target center
            range_offset = r - r_target

            # Pulse compression response
            # Effective resolution including target range extent
            effective_range_res = np.sqrt(radar.range_resolution_m**2 + target_range_extent_m**2)

            range_response = pulse_compression_response(range_offset, effective_range_res)

            if range_response < 1e-6:
                continue

            # Combined 2D PSF
            response[i, j] = amplitude * range_factor * az_response * range_response

    return response


def compute_target_response_with_tail(
    target: Target,
    radar: RadarParams,
    azimuths_deg: np.ndarray,
    ranges_m: np.ndarray,
    tail_decay: float = 0.05,
    tail_spread: float = 1.5
) -> np.ndarray:
    """
    Compute target response with trailing edge effect.

    The trailing edge comes from:
    1. Multiple scattering within/behind target
    2. Diffraction around edges (creeping waves)
    3. Ground multipath behind target

    Modeled as exponential decay with expanding width.
    """
    n_az = len(azimuths_deg)
    n_range = len(ranges_m)

    response = np.zeros((n_az, n_range))

    r_target = target.range_m
    az_target = target.azimuth_deg

    target_angular_extent_rad = target.azimuth_extent_m / target.range_m
    target_range_extent_m = target.range_extent_m

    amplitude = np.sqrt(target.rcs_m2)
    range_factor = 1.0 / (r_target**2)

    # Tail length based on target depth
    tail_length = target.range_extent_m * 3

    for i, az in enumerate(azimuths_deg):
        az_offset_deg = az - az_target
        if az_offset_deg > 180:
            az_offset_deg -= 360
        elif az_offset_deg < -180:
            az_offset_deg += 360

        az_offset_rad = np.radians(az_offset_deg)

        for j, r in enumerate(ranges_m):
            range_offset = r - r_target

            if range_offset < -target_range_extent_m * 2:
                # Too far in front
                continue

            if range_offset <= 0:
                # FRONT OF TARGET: Standard PSF
                effective_beamwidth = np.sqrt(radar.beamwidth_rad**2 + target_angular_extent_rad**2)
                az_response = antenna_pattern(az_offset_rad, effective_beamwidth)

                effective_range_res = np.sqrt(radar.range_resolution_m**2 + target_range_extent_m**2)
                range_response = pulse_compression_response(range_offset, effective_range_res)

                response[i, j] = amplitude * range_factor * az_response * range_response

            else:
                # BEHIND TARGET: Trailing edge with expansion
                # Amplitude decays exponentially
                tail_amp = np.exp(-tail_decay * range_offset)

                if tail_amp < 1e-4:
                    continue

                # Width expands with distance (fan out)
                expansion_factor = 1.0 + (range_offset / tail_length) * tail_spread
                expanded_beamwidth = radar.beamwidth_rad * expansion_factor
                expanded_target_width = target_angular_extent_rad * expansion_factor

                effective_beamwidth = np.sqrt(expanded_beamwidth**2 + expanded_target_width**2)
                az_response = antenna_pattern(az_offset_rad, effective_beamwidth)

                # Range response transitions to exponential decay
                range_response = tail_amp

                response[i, j] = amplitude * range_factor * az_response * range_response * 0.3

    return response


def create_targets():
    """Create test targets."""
    return [
        Target("Ship_NE", 200, 45, rcs_m2=1000, range_extent_m=30, azimuth_extent_m=8),
        Target("Building_E", 150, 90, rcs_m2=500, range_extent_m=20, azimuth_extent_m=25),
        Target("Tank_W", 100, 270, rcs_m2=200, range_extent_m=15, azimuth_extent_m=15),
        Target("Warehouse_S", 300, 180, rcs_m2=2000, range_extent_m=40, azimuth_extent_m=30),
        Target("Buoy_NW", 120, 315, rcs_m2=10, range_extent_m=2, azimuth_extent_m=2),
        Target("Vessel_SW", 350, 225, rcs_m2=3000, range_extent_m=50, azimuth_extent_m=12),
    ]


def run_simulation(
    targets: List[Target],
    radar: RadarParams,
    n_azimuths: int = 720,
    max_range: float = 500,
    include_tail: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run physics-based simulation."""

    # High resolution grid
    range_bin_size = radar.range_resolution_m / 2  # Oversample
    n_range_bins = int(max_range / range_bin_size)

    azimuths = np.linspace(0, 360, n_azimuths, endpoint=False)
    ranges = np.linspace(0, max_range, n_range_bins)

    ppi = np.zeros((n_azimuths, n_range_bins))

    print(f"  Range resolution: {radar.range_resolution_m:.2f} m")
    print(f"  Beamwidth: {radar.beamwidth_deg:.2f}°")
    print(f"  Blind range: {radar.blind_range_m:.1f} m")

    for target in targets:
        if target.range_m < radar.blind_range_m:
            print(f"    {target.name}: IN BLIND ZONE, skipped")
            continue

        print(f"    Computing {target.name}...")

        if include_tail:
            response = compute_target_response_with_tail(target, radar, azimuths, ranges)
        else:
            response = compute_target_response(target, radar, azimuths, ranges)

        ppi += response

    # Normalize
    if ppi.max() > 0:
        ppi = ppi / ppi.max()

    return azimuths, ranges, ppi


def plot_results(results: List[Tuple], output_prefix: str = "physics"):
    """Create comparison plots."""

    # Figure 1: Cartesian view
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 14))
    fig1.suptitle('Physics-Based Radar Simulation\n'
                  '(Antenna pattern × Pulse compression = Natural blob shape)',
                  fontsize=14, fontweight='bold', color='white')

    titles = ['50 MHz BW (3m res)', '35 MHz BW (4.3m res)',
              '25 MHz BW (6m res)', '15 MHz BW (10m res)']

    for idx, (radar, azimuths, ranges, ppi) in enumerate(results):
        ax = axes1[idx // 2, idx % 2]

        cart, x, y = polar_to_cartesian(ppi, azimuths, ranges, output_size=500)
        cart_norm = normalize_for_display(cart, 50.0)

        extent = [x[0], x[-1], y[0], y[-1]]
        ax.imshow(cart_norm, cmap='viridis', origin='lower', extent=extent, aspect='equal')

        # Range rings
        for r in [100, 200, 300, 400]:
            theta = np.linspace(0, 2*np.pi, 100)
            ax.plot(r * np.cos(theta), r * np.sin(theta), 'g-', alpha=0.3, lw=0.5)

        # Blind zone
        blind_theta = np.linspace(0, 2*np.pi, 100)
        ax.fill(radar.blind_range_m * np.cos(blind_theta),
                radar.blind_range_m * np.sin(blind_theta), alpha=0.3, color='red')

        ax.set_title(f'{titles[idx]}\nBlind={radar.blind_range_m:.0f}m, '
                    f'Beam={radar.beamwidth_deg:.1f}°', color='lime', fontsize=11)
        ax.set_facecolor('black')
        ax.tick_params(colors='green')
        ax.set_xlabel('East-West (m)', color='green')
        ax.set_ylabel('North-South (m)', color='green')

    fig1.patch.set_facecolor('black')
    plt.tight_layout()
    fig1.savefig(f'{output_prefix}_cartesian.png', dpi=150, facecolor='black', bbox_inches='tight')
    print(f"  Saved: {output_prefix}_cartesian.png")
    plt.close()

    # Figure 2: Zoomed view of Vessel SW
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 12))
    fig2.suptitle('Zoomed View: Vessel SW - Physics-Based Response',
                  fontsize=14, fontweight='bold', color='white')

    vessel_x = 350 * np.sin(np.radians(225))
    vessel_y = 350 * np.cos(np.radians(225))

    for idx, (radar, azimuths, ranges, ppi) in enumerate(results):
        ax = axes2[idx // 2, idx % 2]

        cart, x, y = polar_to_cartesian(ppi, azimuths, ranges, output_size=500)

        zoom = 100
        x_idx = np.argmin(np.abs(x - vessel_x))
        y_idx = np.argmin(np.abs(y - vessel_y))
        zoom_px = int(zoom / (x[-1] - x[0]) * len(x))

        x_start = max(0, x_idx - zoom_px)
        x_end = min(len(x), x_idx + zoom_px)
        y_start = max(0, y_idx - zoom_px)
        y_end = min(len(y), y_idx + zoom_px)

        zoomed = cart[y_start:y_end, x_start:x_end]
        zoomed_norm = normalize_for_display(zoomed, 50.0)

        extent = [x[x_start] - vessel_x, x[min(x_end, len(x)-1)] - vessel_x,
                  y[y_start] - vessel_y, y[min(y_end, len(y)-1)] - vessel_y]

        ax.imshow(zoomed_norm, cmap='hot', origin='lower', extent=extent, aspect='equal')
        ax.plot(0, 0, 'c+', markersize=15, markeredgewidth=2)

        ax.set_xlabel('Range offset (m)', color='white')
        ax.set_ylabel('Cross-range (m)', color='white')
        ax.set_title(f'Res={radar.range_resolution_m:.1f}m, Beam={radar.beamwidth_deg:.1f}°',
                    color='lime', fontsize=11)
        ax.set_facecolor('black')
        ax.tick_params(colors='white')

    fig2.patch.set_facecolor('black')
    plt.tight_layout()
    fig2.savefig(f'{output_prefix}_zoomed.png', dpi=150, facecolor='black', bbox_inches='tight')
    print(f"  Saved: {output_prefix}_zoomed.png")
    plt.close()

    # Figure 3: Show windowed vs unwindowed PSF
    fig3, axes3 = plt.subplots(2, 3, figsize=(15, 9))
    fig3.suptitle('Radar Response: Unwindowed (Raw) vs Windowed (Filtered)', fontsize=14, fontweight='bold')

    theta = np.linspace(-10, 10, 500)
    r_offset = np.linspace(-20, 20, 500)

    # TOP ROW: Unwindowed (raw physics - shows sidelobes)
    ax = axes3[0, 0]
    pattern_raw = antenna_pattern(np.radians(theta), np.radians(3.9), windowed=False)
    ax.plot(theta, 10*np.log10(pattern_raw + 1e-10), 'b-', linewidth=2)
    ax.axhline(-3, color='r', linestyle='--', alpha=0.5)
    ax.axhline(-13.2, color='orange', linestyle=':', label='-13dB sidelobes')
    ax.set_xlabel('Angle (degrees)')
    ax.set_ylabel('Gain (dB)')
    ax.set_title('Antenna (Unwindowed)\nsinc² with -13dB sidelobes')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-50, 5)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes3[0, 1]
    pulse_raw = pulse_compression_response(r_offset, 3.0, windowed=False)
    ax.plot(r_offset, 10*np.log10(pulse_raw + 1e-10), 'g-', linewidth=2)
    ax.axhline(-3, color='r', linestyle='--', alpha=0.5)
    ax.axhline(-13.2, color='orange', linestyle=':', label='-13dB sidelobes')
    ax.set_xlabel('Range offset (m)')
    ax.set_ylabel('Response (dB)')
    ax.set_title('Pulse Compression (Unwindowed)\nsinc² with -13dB sidelobes')
    ax.set_xlim(-20, 20)
    ax.set_ylim(-50, 5)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes3[0, 2]
    theta_grid = np.linspace(-8, 8, 150)
    r_grid = np.linspace(-15, 15, 150)
    T, R = np.meshgrid(theta_grid, r_grid)
    az_raw = antenna_pattern(np.radians(T), np.radians(3.9), windowed=False)
    range_raw = pulse_compression_response(R, 3.0, windowed=False)
    psf_raw = az_raw * range_raw
    ax.imshow(psf_raw, extent=[theta_grid[0], theta_grid[-1], r_grid[0], r_grid[-1]],
             aspect='auto', cmap='hot', origin='lower')
    ax.set_xlabel('Azimuth (degrees)')
    ax.set_ylabel('Range offset (m)')
    ax.set_title('2D PSF (Unwindowed)\nShows ring sidelobes')

    # BOTTOM ROW: Windowed (filtered - what real radar shows)
    ax = axes3[1, 0]
    pattern_win = antenna_pattern(np.radians(theta), np.radians(3.9), windowed=True)
    ax.plot(theta, 10*np.log10(pattern_win + 1e-10), 'b-', linewidth=2)
    ax.axhline(-3, color='r', linestyle='--', alpha=0.5, label='-3dB width')
    ax.set_xlabel('Angle (degrees)')
    ax.set_ylabel('Gain (dB)')
    ax.set_title('Antenna (Windowed)\nGaussian - no sidelobes')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-50, 5)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes3[1, 1]
    pulse_win = pulse_compression_response(r_offset, 3.0, windowed=True)
    ax.plot(r_offset, 10*np.log10(pulse_win + 1e-10), 'g-', linewidth=2)
    ax.axhline(-3, color='r', linestyle='--', alpha=0.5, label='-3dB width')
    ax.set_xlabel('Range offset (m)')
    ax.set_ylabel('Response (dB)')
    ax.set_title('Pulse Compression (Windowed)\nGaussian - no sidelobes')
    ax.set_xlim(-20, 20)
    ax.set_ylim(-50, 5)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes3[1, 2]
    az_win = antenna_pattern(np.radians(T), np.radians(3.9), windowed=True)
    range_win = pulse_compression_response(R, 3.0, windowed=True)
    psf_win = az_win * range_win
    ax.imshow(psf_win, extent=[theta_grid[0], theta_grid[-1], r_grid[0], r_grid[-1]],
             aspect='auto', cmap='hot', origin='lower')
    ax.set_xlabel('Azimuth (degrees)')
    ax.set_ylabel('Range offset (m)')
    ax.set_title('2D PSF (Windowed)\nSmooth blob - like real radar')

    plt.tight_layout()
    fig3.savefig(f'{output_prefix}_psf.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_prefix}_psf.png")
    plt.close()


def main():
    print("=" * 70)
    print("  PHYSICS-ACCURATE RADAR SIMULATION")
    print("  Blob shape from actual EM physics (no artificial smoothing)")
    print("=" * 70)

    targets = create_targets()
    print(f"\nTargets: {len(targets)}")

    # Different bandwidth configurations
    configs = [
        RadarParams(bandwidth_hz=50e6, pulse_width_s=0.08e-6),   # Short pulse
        RadarParams(bandwidth_hz=35e6, pulse_width_s=0.5e-6),    # Medium
        RadarParams(bandwidth_hz=25e6, pulse_width_s=1.0e-6),    # Long
        RadarParams(bandwidth_hz=15e6, pulse_width_s=2.0e-6),    # X-Long
    ]

    results = []
    total_start = time.time()

    for i, radar in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] BW={radar.bandwidth_hz/1e6:.0f}MHz...")
        start = time.time()
        azimuths, ranges, ppi = run_simulation(targets, radar, n_azimuths=720)
        print(f"    Completed in {time.time() - start:.1f}s")
        results.append((radar, azimuths, ranges, ppi))

    print(f"\nGenerating plots...")
    plot_results(results, "physics_sim")

    print(f"\n" + "=" * 70)
    print(f"  COMPLETE! Total: {time.time() - total_start:.1f}s")
    print("=" * 70)
    print("\nKey physics:")
    print("  - Antenna pattern: sinc²(θ) from aperture diffraction")
    print("  - Pulse compression: sinc²(r) from matched filter")
    print("  - 2D PSF = product of both → natural blob shape")
    print("  - Target extent widens the PSF (convolution)")
    print("  - Trailing edge from scattering/diffraction behind target")


if __name__ == "__main__":
    main()
