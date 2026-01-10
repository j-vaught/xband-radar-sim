#!/usr/bin/env python3
"""
Realistic Radar Simulation with Variable Pulse Modes + Shadow Effect

Shows how different pulse/bandwidth combinations affect radar image:
- Short pulse: Best resolution, less energy
- Long pulse: More energy, worse resolution

Full pipeline includes:
- Atmospheric effects (refraction, attenuation)
- Multipath propagation (ground reflections)
- Target depth with trailing shadow (asymmetric response)
- Beam spreading (azimuth convolution)
- Polar-to-Cartesian scan conversion
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

from config import RadarConfig
from propagation.atmosphere import (
    AtmosphereConfig, apply_atmospheric_attenuation,
    effective_earth_radius
)
from propagation.multipath import (
    SurfaceProperties, SURFACE_CONCRETE,
    two_ray_multipath
)
from signal.waveform import WaveformConfig, generate_lfm_chirp
from signal.matched_filter import matched_filter_fft_windowed
from signal.ppi_processing import (
    quick_beam_spread, normalize_for_display, polar_to_cartesian
)

C = 299792458.0

# ============================================================================
# PULSE MODE CONFIGURATIONS
# Based on real marine radar behavior (Furuno DRS4D-NXT style)
# ============================================================================

@dataclass
class PulseMode:
    """Pulse mode configuration."""
    name: str
    pulse_width_us: float
    bandwidth_mhz: float
    description: str
    recommended_range: str

    @property
    def pulse_width_s(self) -> float:
        return self.pulse_width_us * 1e-6

    @property
    def bandwidth_hz(self) -> float:
        return self.bandwidth_mhz * 1e6

    @property
    def range_resolution_m(self) -> float:
        return C / (2 * self.bandwidth_hz)

    @property
    def blind_range_m(self) -> float:
        return C * self.pulse_width_s / 2

    @property
    def time_bandwidth_product(self) -> float:
        return self.pulse_width_s * self.bandwidth_hz


# Define pulse modes (realistic marine radar configurations)
PULSE_MODES = [
    PulseMode("Short", 0.08, 50.0, "Best resolution", "0-1 NM"),
    PulseMode("Medium", 0.5, 35.0, "Balanced", "1-6 NM"),
    PulseMode("Long", 1.0, 25.0, "Extended range", "6-24 NM"),
    PulseMode("X-Long", 2.0, 15.0, "Maximum range", "24-48 NM"),
]


# ============================================================================
# EXTENDED TARGET WITH SHADOW EFFECT
# ============================================================================

@dataclass
class ExtendedTarget:
    """Target with physical depth for realistic asymmetric range response."""
    name: str
    center: Tuple[float, float, float]  # (x, y, z) position
    depth_m: float                       # Physical depth in range direction
    width_m: float                       # Cross-range width
    height_m: float                      # Vertical extent
    rcs_dbsm: float                      # Peak RCS in dBsm
    # Shadow/trailing edge parameters
    trailing_decay: float = 0.3          # Exponential decay rate
    trailing_extent: float = 2.0         # Shadow extends this many depths behind

    @property
    def shadow_length_m(self) -> float:
        return self.depth_m * self.trailing_extent


def create_extended_targets() -> List[ExtendedTarget]:
    """Create targets with realistic depths for shadow demonstration."""
    targets = []

    # Building NE at 200m - large structure with significant shadow
    r1, az1 = 200, np.radians(45)
    targets.append(ExtendedTarget(
        name="Building NE",
        center=(r1 * np.sin(az1), r1 * np.cos(az1), 15),
        depth_m=25,
        width_m=30,
        height_m=30,
        rcs_dbsm=28,
        trailing_decay=0.25,
        trailing_extent=2.5
    ))

    # Two close targets at E (140m and 160m) - to test resolution
    r2a, az2 = 140, np.radians(90)
    targets.append(ExtendedTarget(
        name="Target E1",
        center=(r2a * np.sin(az2), r2a * np.cos(az2), 5),
        depth_m=10,
        width_m=8,
        height_m=10,
        rcs_dbsm=20,
        trailing_decay=0.4,
        trailing_extent=1.5
    ))

    r2b = 160
    targets.append(ExtendedTarget(
        name="Target E2",
        center=(r2b * np.sin(az2), r2b * np.cos(az2), 5),
        depth_m=10,
        width_m=8,
        height_m=10,
        rcs_dbsm=20,
        trailing_decay=0.4,
        trailing_extent=1.5
    ))

    # Dome/Tank W at 100m - rounded structure
    r3, az3 = 100, np.radians(270)
    targets.append(ExtendedTarget(
        name="Tank W",
        center=(r3 * np.sin(az3), r3 * np.cos(az3), 10),
        depth_m=15,
        width_m=20,
        height_m=15,
        rcs_dbsm=22,
        trailing_decay=0.35,
        trailing_extent=2.0
    ))

    # Warehouse S at 300m - long building
    r4, az4 = 300, np.radians(180)
    targets.append(ExtendedTarget(
        name="Warehouse S",
        center=(r4 * np.sin(az4), r4 * np.cos(az4), 8),
        depth_m=40,
        width_m=25,
        height_m=12,
        rcs_dbsm=30,
        trailing_decay=0.2,
        trailing_extent=3.0
    ))

    # Structure NW at 250m
    r5, az5 = 250, np.radians(315)
    targets.append(ExtendedTarget(
        name="Structure NW",
        center=(r5 * np.sin(az5), r5 * np.cos(az5), 12),
        depth_m=20,
        width_m=18,
        height_m=20,
        rcs_dbsm=25,
        trailing_decay=0.3,
        trailing_extent=2.0
    ))

    # Ship SW at 350m - long vessel with prominent shadow
    r6, az6 = 350, np.radians(225)
    targets.append(ExtendedTarget(
        name="Ship SW",
        center=(r6 * np.sin(az6), r6 * np.cos(az6), 8),
        depth_m=50,
        width_m=12,
        height_m=15,
        rcs_dbsm=32,
        trailing_decay=0.15,
        trailing_extent=3.5
    ))

    return targets


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


def generate_target_echo(
    target: ExtendedTarget,
    radar_pos: np.ndarray,
    ray_direction: np.ndarray,
    tx_waveform: np.ndarray,
    wf_config: WaveformConfig,
    atm_config: AtmosphereConfig,
    surface: SurfaceProperties,
    blind_range: float,
    max_range: float,
    antenna_gain: float
) -> Tuple[np.ndarray, bool]:
    """
    Generate echo from extended target with shadow effect.

    Returns:
        rx_signal: Received signal (or None if no hit)
        hit: Whether target was hit
    """
    to_target = np.array(target.center) - radar_pos
    base_range = np.linalg.norm(to_target)

    if base_range < blind_range or base_range > max_range:
        return None, False

    # Check if ray points at target
    target_dir = to_target / base_range
    dot = np.dot(ray_direction, target_dir)
    angle = np.arccos(np.clip(dot, -1, 1))
    target_angular_size = np.arctan(target.width_m / base_range)

    if angle > target_angular_size + np.radians(0.8):
        return None, False

    # Generate scatterers along target depth with trailing shadow
    half_depth = target.depth_m / 2
    scatterers = []  # (range_offset, amplitude)

    # 1. Leading edge (front surface) - strongest
    scatterers.append((-half_depth, 1.0))

    # 2. Internal scatterers (exponentially decaying into target)
    n_internal = max(3, int(target.depth_m / 5))
    for i in range(1, n_internal):
        frac = i / n_internal
        offset = -half_depth + frac * target.depth_m
        amp = np.exp(-target.trailing_decay * frac * 4) * 0.4
        scatterers.append((offset, amp))

    # 3. Back surface (weaker due to shadowing)
    scatterers.append((half_depth, 0.25))

    # 4. Trailing shadow (diffraction, multipath, creeping waves)
    shadow_length = target.depth_m * target.trailing_extent
    n_shadow = max(5, int(shadow_length / 8))

    for i in range(1, n_shadow + 1):
        frac = i / n_shadow
        offset = half_depth + frac * shadow_length
        amp = 0.15 * np.exp(-target.trailing_decay * i * 0.8)
        if amp > 0.005:
            scatterers.append((offset, amp))

    # Build received signal
    sample_rate = wf_config.sample_rate_hz
    pulse_samples = len(tx_waveform)
    max_offset = max(s[0] for s in scatterers) + 50
    max_delay = int(2 * (base_range + max_offset) / C * sample_rate)
    n_samples = pulse_samples + max_delay + 100

    rx_signal = np.zeros(n_samples, dtype=complex)
    rcs_linear = 10 ** (target.rcs_dbsm / 10)

    for offset, amp in scatterers:
        scatter_range = base_range + offset

        if scatter_range < blind_range or scatter_range > max_range:
            continue

        # Radar equation with antenna pattern
        received_power = (rcs_linear * amp**2 * antenna_gain) / (scatter_range**4 + 1)

        # Atmospheric attenuation
        power_atten = apply_atmospheric_attenuation(
            np.array([received_power]), np.array([scatter_range]),
            wf_config.center_frequency_hz, atm_config
        )[0]

        # Multipath factor (simplified)
        target_height = target.center[2]
        if target_height > 0:
            horiz_range = np.sqrt(target.center[0]**2 + target.center[1]**2)
            if horiz_range > 10:
                try:
                    mp_factor, _, _ = two_ray_multipath(
                        radar_pos[2], target_height, horiz_range,
                        wf_config.center_frequency_hz, surface, 'H'
                    )
                    power_atten *= np.abs(mp_factor)**2
                except:
                    pass

        delay_samples = int(2 * scatter_range / C * sample_rate)

        if delay_samples + pulse_samples < n_samples:
            phase = 4 * np.pi * wf_config.center_frequency_hz * scatter_range / C
            # Add phase variation for internal scatterers
            if abs(offset) < half_depth:
                phase += np.random.uniform(-np.pi/6, np.pi/6)

            echo = np.sqrt(power_atten) * np.exp(1j * phase) * tx_waveform
            rx_signal[delay_samples:delay_samples + pulse_samples] += echo

    return rx_signal, True


def run_simulation_for_mode(
    targets: List[ExtendedTarget],
    pulse_mode: PulseMode,
    radar_pos: np.ndarray,
    radar_config: RadarConfig,
    atm_config: AtmosphereConfig,
    surface: SurfaceProperties,
    n_rays_per_az: int,
    n_azimuths: int,
    max_range: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run simulation for a specific pulse mode with shadow effects."""

    az_bw_rad = np.deg2rad(radar_config.horizontal_beamwidth_deg)
    el_bw_rad = np.deg2rad(radar_config.vertical_beamwidth_deg)

    sample_rate = pulse_mode.bandwidth_hz * 4
    wf_config = WaveformConfig(
        pulse_width_s=pulse_mode.pulse_width_s,
        bandwidth_hz=pulse_mode.bandwidth_hz,
        center_frequency_hz=9.41e9,
        sample_rate_hz=sample_rate
    )

    _, tx_waveform = generate_lfm_chirp(wf_config)
    pulse_samples = len(tx_waveform)

    blind_range = pulse_mode.blind_range_m
    range_resolution = pulse_mode.range_resolution_m

    max_time = 2 * max_range / C
    n_samples_max = int(max_time * sample_rate) + pulse_samples * 2
    n_ranges = n_samples_max - pulse_samples + 1

    azimuths = np.linspace(0, 360, n_azimuths, endpoint=False)
    ppi = np.zeros((n_azimuths, n_ranges))

    sample_time = 1.0 / sample_rate
    ranges = np.arange(n_ranges) * sample_time * C / 2

    for az_idx, az_deg in enumerate(azimuths):
        if az_idx % 60 == 0:
            print(f"      Azimuth {az_idx}/{n_azimuths}")

        boresight_az = np.deg2rad(az_deg)

        # Generate ray bundle
        ray_az_offset = np.random.uniform(-1.5 * az_bw_rad, 1.5 * az_bw_rad, n_rays_per_az)
        ray_el_offset = np.random.uniform(-1.5 * el_bw_rad, 1.5 * el_bw_rad, n_rays_per_az)

        for i in range(n_rays_per_az):
            ray_az = boresight_az + ray_az_offset[i]
            ray_el = ray_el_offset[i]

            direction = np.array([
                np.cos(ray_el) * np.cos(ray_az),
                np.cos(ray_el) * np.sin(ray_az),
                np.sin(ray_el)
            ])
            direction = direction / np.linalg.norm(direction)

            antenna_gain = two_way_pattern(ray_az_offset[i], ray_el_offset[i],
                                           az_bw_rad, el_bw_rad)

            # Accumulate echoes from all targets
            total_rx = np.zeros(n_samples_max, dtype=complex)
            any_hit = False

            for target in targets:
                rx_signal, hit = generate_target_echo(
                    target, radar_pos, direction,
                    tx_waveform, wf_config,
                    atm_config, surface,
                    blind_range, max_range,
                    antenna_gain
                )

                if hit and rx_signal is not None:
                    any_hit = True
                    # Add to total (handle different lengths)
                    min_len = min(len(rx_signal), len(total_rx))
                    total_rx[:min_len] += rx_signal[:min_len]

            if any_hit:
                # Matched filter
                compressed = matched_filter_fft_windowed(total_rx, tx_waveform, "hamming")
                valid_len = min(len(compressed), n_ranges)
                ppi[az_idx, :valid_len] += np.abs(compressed[:valid_len])**2

    # Trim to max range
    max_range_idx = int(max_range / (C / 2 / sample_rate))
    if max_range_idx > n_ranges:
        max_range_idx = n_ranges

    # Normalize
    ppi_trimmed = ppi[:, :max_range_idx]
    if ppi_trimmed.max() > 0:
        ppi_trimmed = ppi_trimmed / ppi_trimmed.max()

    # Apply beam spreading
    ppi_spread = quick_beam_spread(ppi_trimmed, radar_config.horizontal_beamwidth_deg,
                                   n_azimuths, range_sigma_bins=1.0)

    return azimuths, ranges[:max_range_idx], ppi_spread


def plot_pulse_mode_comparison(results: List[Tuple], output_prefix: str = "pulse_mode"):
    """Create comparison plots for different pulse modes."""

    # Figure 1: Full PPI comparison (2x2) in polar
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 14), subplot_kw={'projection': 'polar'})
    fig1.suptitle('Pulse Mode Comparison - Full Pipeline with Shadow Effect\n'
                  '(Atmospheric + Multipath + Target Depth + Beam Spreading)',
                  fontsize=14, fontweight='bold', color='white')

    for idx, (mode, azimuths, ranges, ppi_processed) in enumerate(results):
        ax = axes1[idx // 2, idx % 2]

        az_rad = np.deg2rad(azimuths)
        R, AZ = np.meshgrid(ranges, az_rad)

        ppi_norm = normalize_for_display(ppi_processed, 50.0)

        ax.pcolormesh(AZ, R, ppi_norm, cmap='viridis', shading='auto')

        theta_circle = np.linspace(0, 2 * np.pi, 100)
        ax.fill_between(theta_circle, 0, mode.blind_range_m, alpha=0.4, color='red')

        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_facecolor('black')

        for r in [100, 200, 300, 400]:
            if r < ranges[-1]:
                ax.plot(theta_circle, [r] * 100, 'g-', alpha=0.3, lw=0.5)

        title = (f'{mode.name}\n'
                f'Pulse: {mode.pulse_width_us}μs, BW: {mode.bandwidth_mhz:.0f}MHz\n'
                f'Res: {mode.range_resolution_m:.1f}m, Blind: {mode.blind_range_m:.0f}m')
        ax.set_title(title, color='lime', fontsize=10)
        ax.tick_params(colors='green')

    fig1.patch.set_facecolor('black')
    plt.tight_layout()
    fig1.savefig(f'{output_prefix}_polar_comparison.png', dpi=150,
                 facecolor='black', bbox_inches='tight')
    print(f"  Saved: {output_prefix}_polar_comparison.png")
    plt.close()

    # Figure 2: Cartesian comparison
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 14))
    fig2.suptitle('Pulse Mode Comparison - Cartesian View with Shadow\n'
                  '(Note trailing shadows behind targets)',
                  fontsize=14, fontweight='bold', color='white')

    for idx, (mode, azimuths, ranges, ppi_processed) in enumerate(results):
        ax = axes2[idx // 2, idx % 2]

        cart, x, y = polar_to_cartesian(ppi_processed, azimuths, ranges, output_size=400)
        cart_norm = normalize_for_display(cart, 50.0)

        extent = [x[0], x[-1], y[0], y[-1]]
        ax.imshow(cart_norm, cmap='viridis', origin='lower', extent=extent, aspect='equal')

        ax.set_facecolor('black')

        for r in [100, 200, 300, 400]:
            theta = np.linspace(0, 2 * np.pi, 100)
            ax.plot(r * np.cos(theta), r * np.sin(theta), 'g-', alpha=0.3, lw=0.5)

        blind_theta = np.linspace(0, 2 * np.pi, 100)
        ax.fill(mode.blind_range_m * np.cos(blind_theta),
                mode.blind_range_m * np.sin(blind_theta),
                alpha=0.3, color='red')

        title = (f'{mode.name}: Res={mode.range_resolution_m:.1f}m, '
                f'Blind={mode.blind_range_m:.0f}m')
        ax.set_title(title, color='lime', fontsize=11)
        ax.tick_params(colors='green')
        ax.set_xlabel('East-West (m)', color='green')
        ax.set_ylabel('North-South (m)', color='green')

    fig2.patch.set_facecolor('black')
    plt.tight_layout()
    fig2.savefig(f'{output_prefix}_cartesian_comparison.png', dpi=150,
                 facecolor='black', bbox_inches='tight')
    print(f"  Saved: {output_prefix}_cartesian_comparison.png")
    plt.close()

    # Figure 3: Zoomed view showing shadow effect on Ship SW
    fig3, axes3 = plt.subplots(2, 2, figsize=(14, 12))
    fig3.suptitle('Zoomed View: Ship SW (50m deep) - Shadow Effect vs Pulse Mode',
                  fontsize=14, fontweight='bold', color='white')

    # Ship is at 350m, 225 degrees (SW)
    ship_range = 350
    ship_az = 225
    ship_x = ship_range * np.sin(np.radians(ship_az))
    ship_y = ship_range * np.cos(np.radians(ship_az))

    for idx, (mode, azimuths, ranges, ppi_processed) in enumerate(results):
        ax = axes3[idx // 2, idx % 2]

        cart, x, y = polar_to_cartesian(ppi_processed, azimuths, ranges, output_size=500)

        # Zoom window
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

        # Direction annotations
        ax.annotate('← Radar', xy=(-80, -90), color='lime', fontsize=9)
        ax.annotate('Shadow →', xy=(30, -90), color='red', fontsize=9)

        ax.set_xlabel('Range offset (m)', color='white')
        ax.set_ylabel('Cross-range (m)', color='white')
        ax.set_title(f'{mode.name}: Res={mode.range_resolution_m:.1f}m', color='lime', fontsize=11)
        ax.set_facecolor('black')
        ax.tick_params(colors='white')

    fig3.patch.set_facecolor('black')
    plt.tight_layout()
    fig3.savefig(f'{output_prefix}_shadow_zoom.png', dpi=150,
                 facecolor='black', bbox_inches='tight')
    print(f"  Saved: {output_prefix}_shadow_zoom.png")
    plt.close()

    # Figure 4: Range profile comparison showing shadows
    fig4, ax4 = plt.subplots(figsize=(14, 6))

    colors = ['#00ff00', '#00ffff', '#ffff00', '#ff6600']
    # Look at Ship SW direction (225 degrees)
    az_target = 225

    for idx, (mode, azimuths, ranges, ppi_processed) in enumerate(results):
        az_idx = int(az_target / 360 * len(azimuths))

        profile = np.mean(ppi_processed[max(0, az_idx-2):az_idx+3, :], axis=0)
        if profile.max() > 0:
            profile = profile / profile.max()

        ax4.plot(ranges, profile, color=colors[idx], linewidth=2,
                label=f'{mode.name} (Res={mode.range_resolution_m:.1f}m)', alpha=0.9)

        ax4.axvline(mode.blind_range_m, color=colors[idx], linestyle=':', alpha=0.3)

    # Mark ship position
    ax4.axvline(350, color='white', linestyle='--', alpha=0.5, label='Ship (350m)')

    ax4.set_xlabel('Range (m)', fontsize=12, color='white')
    ax4.set_ylabel('Normalized Intensity', fontsize=12, color='white')
    ax4.set_title('Range Profile at 225° (SW) - Ship with Trailing Shadow', fontsize=14, color='white')
    ax4.legend(loc='upper right', fontsize=10)
    ax4.set_xlim(250, 500)
    ax4.set_ylim(0, 1.1)
    ax4.grid(True, alpha=0.3)
    ax4.set_facecolor('#1a1a1a')
    ax4.tick_params(colors='white')

    # Annotate shadow region
    ax4.annotate('Leading\nEdge', xy=(330, 0.7), color='lime', fontsize=10, ha='center')
    ax4.annotate('Trailing\nShadow', xy=(420, 0.3), color='red', fontsize=10, ha='center')
    ax4.annotate('', xy=(350, 0.5), xytext=(450, 0.5),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))

    fig4.patch.set_facecolor('#1a1a1a')
    plt.tight_layout()
    fig4.savefig(f'{output_prefix}_range_profile.png', dpi=150,
                 facecolor='#1a1a1a', bbox_inches='tight')
    print(f"  Saved: {output_prefix}_range_profile.png")
    plt.close()


def main():
    print("=" * 70)
    print("  FULL PIPELINE RADAR SIMULATION")
    print("  With Variable Pulse Modes + Shadow Effect")
    print("=" * 70)

    print("\nPulse Modes:")
    for mode in PULSE_MODES:
        print(f"  {mode.name:8s}: {mode.pulse_width_us:5.2f}μs, "
              f"BW={mode.bandwidth_mhz:5.1f}MHz, "
              f"Res={mode.range_resolution_m:5.1f}m, "
              f"Blind={mode.blind_range_m:6.0f}m")

    # Create targets with depths
    targets = create_extended_targets()
    print(f"\nTargets with shadow effect:")
    for t in targets:
        print(f"  {t.name:15s}: depth={t.depth_m:3.0f}m, "
              f"shadow~{t.shadow_length_m:.0f}m")

    # Configuration
    radar_pos = np.array([0.0, 0.0, 10.0])
    radar_config = RadarConfig()

    atm_config = AtmosphereConfig(
        temperature_k=288.15,
        pressure_hpa=1013.25,
        relative_humidity=0.6,
        enable_refraction=True,
        enable_attenuation=True
    )

    surface = SURFACE_CONCRETE

    # Simulation parameters
    n_rays_per_az = 800
    n_azimuths = 360
    max_range = 500.0

    print(f"\nSimulation Parameters:")
    print(f"  Azimuths: {n_azimuths}")
    print(f"  Rays per azimuth: {n_rays_per_az}")
    print(f"  Max range: {max_range:.0f} m")

    # Run simulation for each pulse mode
    results = []
    total_start = time.time()

    for i, mode in enumerate(PULSE_MODES):
        print(f"\n[{i+1}/{len(PULSE_MODES)}] Running {mode.name} pulse mode...")
        print(f"    Pulse: {mode.pulse_width_us}μs, BW: {mode.bandwidth_mhz}MHz")

        start = time.time()
        azimuths, ranges, ppi = run_simulation_for_mode(
            targets, mode, radar_pos, radar_config, atm_config, surface,
            n_rays_per_az, n_azimuths, max_range
        )
        sim_time = time.time() - start

        results.append((mode, azimuths, ranges, ppi))
        print(f"    Completed in {sim_time:.1f}s")

    total_time = time.time() - total_start

    # Generate comparison plots
    print(f"\nGenerating comparison plots...")
    plot_pulse_mode_comparison(results, "pulse_mode_full")

    # Summary
    print(f"\n" + "=" * 70)
    print(f"  SIMULATION COMPLETE")
    print(f"=" * 70)
    print(f"\nOutputs:")
    print(f"  pulse_mode_full_polar_comparison.png     - Polar PPI comparison")
    print(f"  pulse_mode_full_cartesian_comparison.png - Cartesian view")
    print(f"  pulse_mode_full_shadow_zoom.png          - Zoomed shadow view")
    print(f"  pulse_mode_full_range_profile.png        - Range profiles")

    print(f"\nPhysics included:")
    print(f"  [x] Atmospheric refraction & attenuation")
    print(f"  [x] Multipath (two-ray ground reflection)")
    print(f"  [x] Target depth with trailing shadow")
    print(f"  [x] Beam spreading (azimuth convolution)")
    print(f"  [x] Variable bandwidth per pulse mode")
    print(f"  [x] Blind zone enforcement")

    print(f"\nTotal time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
