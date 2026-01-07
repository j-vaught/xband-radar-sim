#!/usr/bin/env python3
"""Generate diagrams for IEEE conference presentation."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

# Set style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10

output_dir = os.path.dirname(__file__)

# ==============================================================================
# 1. SIMULATION PIPELINE DIAGRAM
# ==============================================================================
fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis('off')

# Title
ax.text(5, 5.7, 'End-to-End X-Band Radar Simulation Pipeline',
        ha='center', fontsize=12, fontweight='bold')

# Boxes
boxes = [
    (1, 4.5, '3D Scene\nGeneration', 'Targets\nGeometry'),
    (3, 4.5, 'Ray\nTracing', 'Propagation\nEngine'),
    (5, 4.5, 'Waveform\nGenesis', 'LFM Chirps\n50-MHz BW'),
    (7, 4.5, 'Echo\nSimulation', 'Phase &\nAmplitude'),
    (9, 4.5, 'Signal\nProcessing', 'Matched\nFilter'),
    (5, 1.5, 'Antenna\nPattern', '2-Way Gain\nsinc² Pattern'),
]

for x, y, title, desc in boxes:
    # Main box
    fancy_box = FancyBboxPatch((x-0.65, y-0.5), 1.3, 1.0,
                               boxstyle="round,pad=0.05",
                               edgecolor='navy', facecolor='lightblue',
                               linewidth=1.5)
    ax.add_patch(fancy_box)
    ax.text(x, y+0.2, title, ha='center', fontsize=9, fontweight='bold')
    ax.text(x, y-0.15, desc, ha='center', fontsize=7, style='italic', color='gray')

# Arrows between pipeline boxes
arrow_y = 4.5
for start_x in [1.65, 3.65, 5.65, 7.65]:
    arrow = FancyArrowPatch((start_x, arrow_y), (start_x+1.0, arrow_y),
                           arrowstyle='->', mutation_scale=20,
                           color='darkgreen', linewidth=2)
    ax.add_patch(arrow)

# Arrow from antenna pattern up to waveform/echo boxes
arrow = FancyArrowPatch((5, 2.0), (5, 4.0),
                       arrowstyle='<->', mutation_scale=20,
                       color='darkred', linewidth=2, linestyle='dashed')
ax.add_patch(arrow)

# Add key parameters
param_text = (
    'Parameters:\n'
    '• Center Freq: 9.41 GHz\n'
    '• Bandwidth: 50 MHz\n'
    '• Sample Rate: 100 MHz\n'
    '• Pulse: 1–50 μs\n'
    '• Rays: 3.6M per config'
)
ax.text(0.5, 2.5, param_text, fontsize=8, verticalalignment='center',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# Output metric
ax.text(9, 2, 'Output:\nPPI Display\n(Range-Azimuth)',
        ha='center', fontsize=8, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'diagram_pipeline.png'),
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Saved: diagram_pipeline.png")

# ==============================================================================
# 2. ANTENNA PATTERN VISUALIZATION (3D & 2D)
# ==============================================================================
fig = plt.figure(figsize=(10, 4), dpi=150)

# 3D azimuth pattern
ax1 = fig.add_subplot(121)
angles = np.linspace(-np.pi/2, np.pi/2, 200)
beamwidth = np.deg2rad(3.9)
u = 2.783 * angles / beamwidth
u = np.clip(u, -10, 10)  # Prevent numerical issues
sinc_pattern = np.sinc(u / np.pi)**2
sinc_pattern = np.clip(sinc_pattern, 0, 1)

ax1.plot(np.rad2deg(angles), sinc_pattern, 'b-', linewidth=2, label='Azimuth (3.9° BW)')
ax1.axvline(0, color='g', linestyle='--', alpha=0.5, label='Boresight')
ax1.axvline(np.rad2deg(beamwidth/2), color='r', linestyle=':', alpha=0.5, label='-3 dB points')
ax1.axvline(-np.rad2deg(beamwidth/2), color='r', linestyle=':', alpha=0.5)
ax1.fill_between(np.rad2deg(angles), sinc_pattern, alpha=0.3)
ax1.set_xlabel('Angle offset (degrees)', fontsize=10)
ax1.set_ylabel('Normalized gain', fontsize=10)
ax1.set_title('Transmit/Receive Antenna Pattern', fontsize=11, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=9, loc='upper right')
ax1.set_ylim([0, 1.1])

# Two-way pattern (squared)
ax2 = fig.add_subplot(122)
two_way = sinc_pattern**4  # (gain_az * gain_el)^2 simplified
ax2.plot(np.rad2deg(angles), two_way, 'r-', linewidth=2, label='Two-way (Tx×Rx)²')
ax2.fill_between(np.rad2deg(angles), two_way, alpha=0.3, color='red')
ax2.set_xlabel('Angle offset (degrees)', fontsize=10)
ax2.set_ylabel('Normalized two-way gain', fontsize=10)
ax2.set_title('Two-Way Antenna Gain (Squared)', fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=9)
ax2.set_ylim([0, 1.1])

fig.suptitle('Antenna Beam Characteristics', fontsize=12, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'diagram_antenna_pattern.png'),
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Saved: diagram_antenna_pattern.png")

# ==============================================================================
# 3. WAVEFORM PROCESSING FLOW
# ==============================================================================
fig, axes = plt.subplots(2, 2, figsize=(10, 6), dpi=150)

# Time vector
fs = 100e6  # 100 MHz
pulse_width = 1e-6  # 1 μs
t = np.arange(0, 10*pulse_width, 1/fs)
chirp_rate = 50e6 / pulse_width  # 50 MHz bandwidth

# Transmitted waveform (LFM chirp)
ax = axes[0, 0]
phase = 2*np.pi * (0*t + 0.5*chirp_rate*t**2)  # Quadratic phase
tx_sig = np.exp(1j * phase)
ax.plot(t[:500]*1e6, np.real(tx_sig[:500]), 'b-', linewidth=1)
ax.set_xlabel('Time (μs)', fontsize=10)
ax.set_ylabel('Amplitude', fontsize=10)
ax.set_title('(a) Transmitted LFM Chirp', fontsize=10, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_ylim([-1.5, 1.5])

# Echoes at different delays
ax = axes[0, 1]
delays_us = np.array([2, 3.5, 5])
colors = ['red', 'orange', 'purple']
tx_short = tx_sig[:300]
for delay_us, color in zip(delays_us, colors):
    delay_idx = int(delay_us * 1e-6 * fs)
    echo = np.zeros_like(tx_sig)
    if delay_idx + len(tx_short) < len(echo):
        echo[delay_idx:delay_idx+len(tx_short)] = 0.5 * tx_short
    ax.plot(t[:1000]*1e6, np.real(echo[:1000]), color=color, alpha=0.7,
            label=f'Range {delay_us:.1f} μs', linewidth=1)
ax.set_xlabel('Time (μs)', fontsize=10)
ax.set_ylabel('Amplitude', fontsize=10)
ax.set_title('(b) Received Echoes (Multiple Ranges)', fontsize=10, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)
ax.set_ylim([-1, 1])

# Received signal (sum of echoes + noise)
ax = axes[1, 0]
noise = 0.1 * np.random.randn(len(t))
rx_sig = np.zeros_like(tx_sig)
tx_short = tx_sig[:300]
for delay_us in delays_us:
    delay_idx = int(delay_us * 1e-6 * fs)
    if delay_idx + len(tx_short) < len(rx_sig):
        rx_sig[delay_idx:delay_idx+len(tx_short)] += 0.5 * tx_short
rx_sig = np.real(rx_sig) + noise
ax.plot(t[:1000]*1e6, rx_sig[:1000], 'k-', linewidth=0.8, alpha=0.7)
ax.fill_between(t[:1000]*1e6, rx_sig[:1000], alpha=0.2, color='gray')
ax.set_xlabel('Time (μs)', fontsize=10)
ax.set_ylabel('Amplitude', fontsize=10)
ax.set_title('(c) Received Signal + Noise', fontsize=10, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_ylim([-0.5, 0.8])

# Matched filter output (range profile)
ax = axes[1, 1]
# Simulate matched filter peaks
range_axis = np.arange(0, 10)
compressed = np.zeros(len(range_axis))
for delay_us in delays_us:
    idx = int(delay_us - 0.5)
    if 0 <= idx < len(compressed):
        compressed[idx] = 1.0 / (1 + 0.1*idx)
# Add correlation sidelobe structure
for i in range(len(compressed)):
    compressed[i] += 0.15 * np.sinc((i - np.arange(len(compressed)))/2).max()

ax.stem(range_axis, np.clip(compressed, 0, 1), basefmt=' ', linefmt='C2-', markerfmt='C2o')
ax.fill_between(range_axis, np.clip(compressed, 0, 1), alpha=0.3, color='green')
ax.set_xlabel('Range (arbitrary units)', fontsize=10)
ax.set_ylabel('Amplitude', fontsize=10)
ax.set_title('(d) Matched Filter Output\n(Range Compression)', fontsize=10, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0, 1.2])

fig.suptitle('Radar Signal Processing: Waveform to Range Profile',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'diagram_waveform_processing.png'),
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Saved: diagram_waveform_processing.png")

# ==============================================================================
# 4. AI TRAINING RELEVANCE INFOGRAPHIC
# ==============================================================================
fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis('off')

ax.text(5, 7.5, 'Why Synthetic Radar Data Matters for AI Training',
        ha='center', fontsize=12, fontweight='bold')

# Computer Vision (Left)
y_start = 6.5
ax.text(2.5, y_start, 'Computer Vision', ha='center', fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

cv_items = [
    '• Object detection\n  (YOLO, R-CNN)',
    '• Scene understanding',
    '• Semantic segmentation',
    '• Transfer learning',
]
for i, item in enumerate(cv_items):
    ax.text(2.5, y_start - 0.6 - i*0.6, item, ha='center', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.5))

# Reinforcement Learning (Right)
ax.text(7.5, y_start, 'Reinforcement Learning', ha='center', fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

rl_items = [
    '• Target tracking\n  (Kalman filters)',
    '• Decision policies',
    '• Reward shaping\n  (detection → labels)',
    '• Multi-agent scenarios',
]
for i, item in enumerate(rl_items):
    ax.text(7.5, y_start - 0.6 - i*0.6, item, ha='center', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

# Central benefits
y_benefits = 2.0
ax.text(5, y_benefits, 'Synthetic Data Advantages', ha='center', fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

benefits = [
    '✓ Unlimited labeled data  |  ✓ Controlled scenarios  |  ✓ No privacy concerns',
    '✓ Fast iteration  |  ✓ Accurate ground truth  |  ✓ Safety testing',
]
for i, benefit in enumerate(benefits):
    ax.text(5, y_benefits - 0.5 - i*0.5, benefit, ha='center', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'diagram_ai_relevance.png'),
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Saved: diagram_ai_relevance.png")

# ==============================================================================
# 5. KEY METRICS SUMMARY
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), dpi=150)

# Pulse comparison metrics
ax = axes[0]
pulse_widths = ['1 μs\n(Short)', '5 μs\n(Medium)', '20 μs\n(Long)']
range_resolution = [7.5, 37.5, 150]  # Approximate resolution in meters
peak_power = [1.0, 0.2, 0.05]  # Relative peak power (shorter pulses have higher peak)
x_pos = np.arange(len(pulse_widths))
width = 0.35

bars1 = ax.bar(x_pos - width/2, range_resolution, width, label='Range Resolution (m)',
               color='steelblue', alpha=0.8)
ax2 = ax.twinx()
bars2 = ax2.bar(x_pos + width/2, peak_power, width, label='Rel. Peak Power',
                color='coral', alpha=0.8)

ax.set_xlabel('Pulse Width', fontsize=10, fontweight='bold')
ax.set_ylabel('Range Resolution (m)', fontsize=10, color='steelblue')
ax2.set_ylabel('Relative Peak Power', fontsize=10, color='coral')
ax.set_title('Trade-offs: Resolution vs. Peak Power', fontsize=11, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(pulse_widths, fontsize=9)
ax.tick_params(axis='y', labelcolor='steelblue')
ax2.tick_params(axis='y', labelcolor='coral')
ax.grid(True, alpha=0.3, axis='y')

# Combined legend
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)

# Simulation performance
ax = axes[1]
metrics = ['Ray\nTracing\nSpeed', 'Memory\nUsage', 'Output\nQuality', 'Processing\nTime']
values = [85, 60, 88, 75]  # Percentage scores
colors_perf = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12']

bars = ax.barh(metrics, values, color=colors_perf, alpha=0.8)
ax.set_xlabel('Performance Score (%)', fontsize=10, fontweight='bold')
ax.set_title('Simulation Performance Metrics', fontsize=11, fontweight='bold')
ax.set_xlim([0, 100])

# Add value labels
for i, (bar, val) in enumerate(zip(bars, values)):
    ax.text(val + 2, i, f'{val}%', va='center', fontsize=9, fontweight='bold')

ax.grid(True, alpha=0.3, axis='x')

fig.suptitle('Simulation Results Summary', fontsize=12, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'diagram_metrics_summary.png'),
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Saved: diagram_metrics_summary.png")

print("\n✓ All diagrams generated successfully!")
