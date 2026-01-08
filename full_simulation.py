#!/usr/bin/env python3
"""
Bandwidth vs Pulse Width Comparison.

Shows how BANDWIDTH (not pulse width) determines range resolution
after matched filtering.
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
from propagation.scene import (
    Scene, TargetObject,
    create_corner_reflector, create_flat_plate, create_sphere, create_cylinder
)
from signal.waveform import WaveformConfig, generate_lfm_chirp
from signal.matched_filter import matched_filter_fft_windowed


def antenna_pattern_1d(angle_rad, beamwidth_rad):
    u = 2.783 * angle_rad / beamwidth_rad
    if abs(u) < 1e-6:
        return 1.0
    return max((np.sin(np.pi * u) / (np.pi * u)) ** 2, 1e-6)


def two_way_pattern(az_off, el_off, az_bw, el_bw):
    return (antenna_pattern_1d(az_off, az_bw) * antenna_pattern_1d(el_off, el_bw)) ** 2


def create_scene():
    scene = Scene()
    
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


def run_sim(scene, radar_pos, wf_config, radar_config, n_rays_per_az, n_azimuths, max_range):
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
        rx_signal += np.sqrt(1e-18/2) * (np.random.randn(n_samples) + 1j*np.random.randn(n_samples))
        
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


def main():
    print("=" * 70)
    print("  BANDWIDTH + PULSE WIDTH COMPARISON")
    print("  Shows how BANDWIDTH determines range resolution")
    print("=" * 70)
    print(f"\nNumba: {'ENABLED' if NUMBA_AVAILABLE else 'DISABLED'}")
    
    scene = create_scene()
    radar_pos = np.array([0.0, 0.0, 5.0])
    radar_config = RadarConfig()
    
    c = 299792458.0
    
    # Configs: (name, pulse_width, bandwidth)
    configs = [
        ("1μs, 50MHz\n3m res", 1e-6, 50e6),
        ("10μs, 50MHz\n3m res", 10e-6, 50e6),
        ("10μs, 10MHz\n15m res", 10e-6, 10e6),
        ("50μs, 5MHz\n30m res", 50e-6, 5e6),
    ]
    
    results = []
    
    for name, pulse_width, bandwidth in configs:
        res_m = c / (2 * bandwidth)
        print(f"\nRunning: {name.split(chr(10))[0]} → {res_m:.0f}m resolution")
        
        wf_config = WaveformConfig(
            pulse_width_s=pulse_width,
            bandwidth_hz=bandwidth,
            center_frequency_hz=9.41e9,
            sample_rate_hz=100e6
        )
        
        start = time.time()
        azimuths, ranges, ppi = run_sim(
            scene, radar_pos, wf_config, radar_config,
            n_rays_per_az=3000,  # Fewer rays for speed
            n_azimuths=360,
            max_range=500.0
        )
        print(f"  Done in {time.time()-start:.1f}s")
        results.append((name, azimuths, ranges, ppi))
    
    # Plot
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5), subplot_kw={'projection': 'polar'})
    
    for idx, (name, azimuths, ranges, ppi) in enumerate(results):
        ax = axes[idx]
        az_rad = np.deg2rad(azimuths)
        R, AZ = np.meshgrid(ranges, az_rad)
        
        ppi_db = 10 * np.log10(ppi + 1e-30)
        vmax = np.max(ppi_db)
        ppi_norm = np.clip((ppi_db - (vmax - 50)) / 50, 0, 1)
        
        ax.pcolormesh(AZ, R, ppi_norm, cmap='gray', shading='auto')
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_facecolor('black')
        
        for r in [100, 200, 300, 400]:
            if r < ranges[-1]:
                ax.plot(np.linspace(0, 2*np.pi, 100), [r]*100, 'g-', alpha=0.3, lw=0.5)
        
        ax.set_title(name, color='green', fontsize=11)
        ax.tick_params(colors='green')
        ax.grid(True, color='green', alpha=0.3, lw=0.5)
        ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
        ax.set_xticklabels(['N', 'E', 'S', 'W'], color='green')
    
    fig.patch.set_facecolor('black')
    fig.suptitle('Bandwidth Determines Range Resolution (after matched filter)\n'
                 'Lower bandwidth = wider targets',
                 color='green', fontsize=14, y=1.02)
    
    plt.tight_layout()
    output = os.path.join(os.path.dirname(__file__), 'ppi_bandwidth_comparison.png')
    plt.savefig(output, dpi=150, facecolor='black', bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved: {output}")
    print("\n" + "=" * 70)
    print("  COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
