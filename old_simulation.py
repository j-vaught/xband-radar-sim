#!/usr/bin/env python3
"""
16-Run Grid: Bandwidth × Pulse Width

4 Bandwidths: 100MHz, 50MHz, 10MHz, 5MHz
4 Pulse widths: 1μs, 10μs, 100μs, 1000μs

Saves individual PPIs + combined 4x4 grid.
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


def plot_single_ppi(azimuths, ranges, ppi, title, output_path):
    """Save individual PPI plot."""
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})
    
    az_rad = np.deg2rad(azimuths)
    R, AZ = np.meshgrid(ranges, az_rad)
    
    ppi_db = 10 * np.log10(ppi + 1e-30)
    vmax = np.max(ppi_db)
    ppi_norm = np.clip((ppi_db - (vmax - 50)) / 50, 0, 1)
    
    ax.pcolormesh(AZ, R, ppi_norm, cmap='gray', shading='auto')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    
    for r in [100, 200, 300, 400]:
        if r < ranges[-1]:
            ax.plot(np.linspace(0, 2*np.pi, 100), [r]*100, 'g-', alpha=0.3, lw=0.5)
    
    ax.set_title(title, color='green', fontsize=10)
    ax.tick_params(colors='green')
    ax.grid(True, color='green', alpha=0.3, lw=0.5)
    ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
    ax.set_xticklabels(['N', 'E', 'S', 'W'], color='green')
    
    plt.savefig(output_path, dpi=100, facecolor='black', bbox_inches='tight')
    plt.close()


def main():
    print("=" * 70)
    print("  16-RUN GRID: Bandwidth × Pulse Width")
    print("=" * 70)
    print(f"\nNumba: {'ENABLED' if NUMBA_AVAILABLE else 'DISABLED'}")
    
    scene = create_scene()
    radar_pos = np.array([0.0, 0.0, 5.0])
    radar_config = RadarConfig()
    c = 299792458.0
    
    # Create output folder
    output_dir = os.path.join(os.path.dirname(__file__), 'ppi_grid_outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    # Grid parameters
    bandwidths = [100e6, 50e6, 10e6, 5e6]  # Hz
    pulse_widths = [1e-6, 10e-6, 100e-6, 1000e-6]  # seconds
    
    bw_labels = ['100MHz', '50MHz', '10MHz', '5MHz']
    pw_labels = ['1μs', '10μs', '100μs', '1000μs']
    
    # Store results for combined grid
    all_results = {}  # (bw_idx, pw_idx) -> (azimuths, ranges, ppi)
    
    total_runs = len(bandwidths) * len(pulse_widths)
    run_num = 0
    total_start = time.time()
    
    for bw_idx, bandwidth in enumerate(bandwidths):
        res_m = c / (2 * bandwidth)
        
        for pw_idx, pulse_width in enumerate(pulse_widths):
            run_num += 1
            
            label = f"{bw_labels[bw_idx]}_{pw_labels[pw_idx]}"
            print(f"\n[{run_num}/{total_runs}] {label} → {res_m:.1f}m resolution")
            
            wf_config = WaveformConfig(
                pulse_width_s=pulse_width,
                bandwidth_hz=bandwidth,
                center_frequency_hz=9.41e9,
                sample_rate_hz=200e6  # 200MHz sample rate
            )
            
            start = time.time()
            azimuths, ranges, ppi = run_sim(
                scene, radar_pos, wf_config, radar_config,
                n_rays_per_az=2000,  # Fast
                n_azimuths=360,
                max_range=500.0
            )
            elapsed = time.time() - start
            print(f"    Done in {elapsed:.1f}s")
            
            # Save individual PPI
            individual_path = os.path.join(output_dir, f'ppi_{label}.png')
            title = f'{bw_labels[bw_idx]}, {pw_labels[pw_idx]}\n{res_m:.1f}m res'
            plot_single_ppi(azimuths, ranges, ppi, title, individual_path)
            print(f"    Saved: {individual_path}")
            
            all_results[(bw_idx, pw_idx)] = (azimuths, ranges, ppi)
    
    # Create combined 4x4 grid
    print("\n" + "=" * 70)
    print("Creating combined 4×4 grid...")
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 16), subplot_kw={'projection': 'polar'})
    
    for bw_idx in range(4):
        for pw_idx in range(4):
            ax = axes[bw_idx, pw_idx]
            azimuths, ranges, ppi = all_results[(bw_idx, pw_idx)]
            
            az_rad = np.deg2rad(azimuths)
            R, AZ = np.meshgrid(ranges, az_rad)
            
            ppi_db = 10 * np.log10(ppi + 1e-30)
            vmax = np.max(ppi_db)
            ppi_norm = np.clip((ppi_db - (vmax - 50)) / 50, 0, 1)
            
            ax.pcolormesh(AZ, R, ppi_norm, cmap='gray', shading='auto')
            ax.set_theta_zero_location('N')
            ax.set_theta_direction(-1)
            ax.set_facecolor('black')
            
            # Only show cardinal directions
            ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.grid(True, color='green', alpha=0.2, lw=0.3)
            
            # Title for each cell
            res_m = c / (2 * bandwidths[bw_idx])
            ax.set_title(f'{bw_labels[bw_idx]}\n{pw_labels[pw_idx]}', 
                        color='green', fontsize=9, pad=5)
    
    fig.patch.set_facecolor('black')
    
    # Add row/column labels
    for bw_idx in range(4):
        res_m = c / (2 * bandwidths[bw_idx])
        fig.text(0.02, 0.8 - bw_idx * 0.2, f'{res_m:.1f}m\nres', 
                color='yellow', fontsize=10, va='center')
    
    fig.suptitle('Bandwidth × Pulse Width Grid\n'
                 'Rows: 100→5 MHz BW (resolution increases ↓) | Cols: 1→1000 μs pulse',
                 color='green', fontsize=14, y=0.98)
    
    plt.tight_layout(rect=[0.05, 0, 1, 0.96])
    
    grid_path = os.path.join(os.path.dirname(__file__), 'ppi_bw_pulse_grid.png')
    plt.savefig(grid_path, dpi=150, facecolor='black', bbox_inches='tight')
    plt.close()
    
    total_elapsed = time.time() - total_start
    
    print(f"\nSaved combined grid: {grid_path}")
    print(f"Individual PPIs: {output_dir}/")
    print(f"\nTotal time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print("\n" + "=" * 70)
    print("  COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
