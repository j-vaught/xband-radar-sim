#!/usr/bin/env python3
"""Quick dynamic range comparison."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import RadarConfig
from propagation.cpu_raytrace import trace_rays_cpu
from propagation.scene import Scene, TargetObject, create_corner_reflector, create_sphere
from signal.waveform import WaveformConfig, generate_lfm_chirp
from signal.matched_filter import matched_filter_fft_windowed

def antenna_pattern_1d(angle_rad, beamwidth_rad):
    u = 2.783 * angle_rad / beamwidth_rad
    if abs(u) < 1e-6: return 1.0
    return max((np.sin(np.pi * u) / (np.pi * u)) ** 2, 1e-6)

def two_way_pattern(az_off, el_off, az_bw, el_bw):
    return (antenna_pattern_1d(az_off, az_bw) * antenna_pattern_1d(el_off, el_bw)) ** 2

scene = Scene()
cr_v, cr_f = create_corner_reflector(edge_length_m=50.0)
scene.add_target(TargetObject(name='building_NE', position=(200*np.cos(np.deg2rad(45)), 200*np.sin(np.deg2rad(45)), 0), vertices=cr_v, faces=cr_f))
s_v, s_f = create_sphere(radius_m=20.0, n_segments=16)
scene.add_target(TargetObject(name='dome_W', position=(100*np.cos(np.deg2rad(270)), 100*np.sin(np.deg2rad(270)), 20), vertices=s_v, faces=s_f))
cr_v2, cr_f2 = create_corner_reflector(edge_length_m=40.0)
scene.add_target(TargetObject(name='warehouse_S', position=(300*np.cos(np.deg2rad(180)), 300*np.sin(np.deg2rad(180)), 0), vertices=cr_v2, faces=cr_f2))

radar_pos = np.array([0.0, 0.0, 5.0])
radar_config = RadarConfig()
c = 299792458.0
triangles = scene.get_all_triangles()
az_bw_rad = np.deg2rad(radar_config.horizontal_beamwidth_deg)
el_bw_rad = np.deg2rad(radar_config.vertical_beamwidth_deg)

print('Running with extended dynamic range display...')
wf_config = WaveformConfig(pulse_width_s=2000e-6, bandwidth_hz=50e6, center_frequency_hz=9.41e9, sample_rate_hz=200e6)
_, tx_waveform = generate_lfm_chirp(wf_config)
pulse_samples = len(tx_waveform)
max_range = 500.0
max_time = 2 * max_range / c
n_samples = int(max_time * wf_config.sample_rate_hz) + pulse_samples * 2
n_ranges = n_samples - pulse_samples + 1
azimuths = np.linspace(0, 360, 360, endpoint=False)
ppi = np.zeros((360, n_ranges))
sample_time = 1.0 / wf_config.sample_rate_hz
ranges = np.arange(n_ranges) * sample_time * c / 2

for az_idx, az_deg in enumerate(azimuths):
    boresight_az = np.deg2rad(az_deg)
    n_rays = 2000
    ray_az_offset = np.random.uniform(-1.5*az_bw_rad, 1.5*az_bw_rad, n_rays)
    ray_el_offset = np.random.uniform(-1.5*el_bw_rad, 1.5*el_bw_rad, n_rays)
    ray_az = boresight_az + ray_az_offset
    ray_el = ray_el_offset
    directions = np.column_stack([np.cos(ray_el)*np.cos(ray_az), np.cos(ray_el)*np.sin(ray_az), np.sin(ray_el)])
    origins = np.tile(radar_pos, (n_rays, 1))
    bundle = trace_rays_cpu(origins, directions, triangles, max_range)
    rx_signal = np.zeros(n_samples, dtype=complex)
    rx_signal += np.sqrt(1e-18/2) * (np.random.randn(n_samples) + 1j*np.random.randn(n_samples))
    for i in range(bundle.n_rays):
        if bundle.hit_mask[i]:
            range_m = bundle.path_lengths_m[i]
            antenna_gain = two_way_pattern(ray_az_offset[i], ray_el_offset[i], az_bw_rad, el_bw_rad)
            delay_samples = int(2*range_m/c*wf_config.sample_rate_hz)
            if delay_samples + pulse_samples < n_samples:
                amplitude = antenna_gain / (range_m**2 + 1)
                phase = 4*np.pi*wf_config.center_frequency_hz*range_m/c
                echo = amplitude * np.exp(1j*phase) * tx_waveform
                rx_signal[delay_samples:delay_samples+pulse_samples] += echo
    compressed = matched_filter_fft_windowed(rx_signal, tx_waveform, 'hamming')
    valid_len = min(len(compressed), n_ranges)
    ppi[az_idx, :valid_len] = np.abs(compressed[:valid_len])**2
    if (az_idx+1) % 90 == 0: print(f'  {az_idx+1}/360')

max_range_idx = int(max_range / (c / 2 / wf_config.sample_rate_hz))
ranges = ranges[:max_range_idx]
ppi = ppi[:, :max_range_idx]

fig, axes = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={'projection': 'polar'})
az_rad = np.deg2rad(azimuths)
R, AZ = np.meshgrid(ranges, az_rad)
ppi_db = 10 * np.log10(ppi + 1e-30)
vmax = np.max(ppi_db)

for idx, (dyn_range, title) in enumerate([(50, '50dB (default)'), (80, '80dB (all returns)')]):
    ax = axes[idx]
    ppi_norm = np.clip((ppi_db - (vmax - dyn_range)) / dyn_range, 0, 1)
    ax.pcolormesh(AZ, R, ppi_norm, cmap='gray', shading='auto')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_facecolor('black')
    ax.set_title(title, color='green', fontsize=12)
    ax.tick_params(colors='green')
    ax.grid(True, color='green', alpha=0.3)
    ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
    ax.set_xticklabels(['N', 'E', 'S', 'W'], color='green')

fig.patch.set_facecolor('black')
fig.suptitle('Dynamic Range Comparison - See ALL Returns', color='green', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'ppi_dynamic_range.png'), dpi=150, facecolor='black', bbox_inches='tight')
print('Saved: ppi_dynamic_range.png')
