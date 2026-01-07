"""Far-field pattern computation from openEMS simulation."""
from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple, Optional
import numpy as np
import os

if TYPE_CHECKING:
    from openEMS import openEMS

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from interfaces import AntennaPattern


@dataclass
class FarFieldConfig:
    """Configuration for far-field computation."""
    theta_range: Tuple[float, float] = (-90, 90)
    theta_step: float = 1.0
    phi_range: Tuple[float, float] = (-180, 180)
    phi_step: float = 1.0
    frequency_hz: float = 9.41e9


def setup_nf2ff_box(
    fdtd: "openEMS",
    name: str = "nf2ff",
    margin_cells: int = 3
) -> object:
    """
    Create near-field to far-field transformation box.
    
    The NF2FF box records tangential E and H fields on its surface
    for post-processing into far-field patterns.
    
    Args:
        fdtd: openEMS FDTD object
        name: Name for NF2FF box
        margin_cells: Margin from simulation boundary in cells
    
    Returns:
        NF2FF box object
    """
    nf2ff = fdtd.CreateNF2FFBox(name=name)
    return nf2ff


def compute_farfield(
    nf2ff: object,
    sim_path: str,
    config: FarFieldConfig
) -> AntennaPattern:
    """
    Compute far-field pattern from NF2FF box.
    
    Args:
        nf2ff: NF2FF box object from simulation
        sim_path: Path to simulation data directory
        config: Far-field configuration
    
    Returns:
        AntennaPattern with gain and phase
    """
    # Create angle arrays
    theta = np.arange(
        config.theta_range[0], 
        config.theta_range[1] + config.theta_step/2, 
        config.theta_step
    )
    phi = np.arange(
        config.phi_range[0], 
        config.phi_range[1] + config.phi_step/2, 
        config.phi_step
    )
    
    # Convert to radians for NF2FF
    theta_rad = np.deg2rad(theta)
    phi_rad = np.deg2rad(phi)
    
    # Compute far-field
    result = nf2ff.CalcNF2FF(
        sim_path=sim_path,
        freq=config.frequency_hz,
        theta=theta_rad,
        phi=phi_rad,
        center=[0, 0, 0]
    )
    
    # Extract gain from directivity
    # result.Dmax is peak directivity
    # result.E_norm is normalized E-field pattern
    E_theta = np.array(result.E_theta)
    E_phi = np.array(result.E_phi)
    
    # Total field magnitude
    E_total_sq = np.abs(E_theta)**2 + np.abs(E_phi)**2
    
    # Normalize and scale by directivity
    if np.max(E_total_sq) > 0:
        E_norm = E_total_sq / np.max(E_total_sq)
        gain_linear = result.Dmax * E_norm
    else:
        gain_linear = np.zeros_like(E_total_sq)
    
    # Phase from complex E_theta (dominant for horizontal polarization)
    phase_rad = np.angle(E_theta)
    
    return AntennaPattern(
        frequency_hz=config.frequency_hz,
        theta_deg=theta,
        phi_deg=phi,
        gain_linear=gain_linear,
        phase_rad=phase_rad
    )


def extract_principal_cuts(pattern: AntennaPattern) -> dict:
    """
    Extract principal plane cuts from 2D pattern.
    
    Returns:
        Dict with E-plane and H-plane cuts
    """
    # Find peak gain location
    peak_idx = np.unravel_index(
        np.argmax(pattern.gain_linear), 
        pattern.gain_linear.shape
    )
    peak_theta_idx, peak_phi_idx = peak_idx
    
    # E-plane cut (phi = 0°, vary theta)
    phi_zero_idx = np.argmin(np.abs(pattern.phi_deg))
    e_plane = {
        'theta_deg': pattern.theta_deg,
        'gain_db': 10 * np.log10(pattern.gain_linear[:, phi_zero_idx] + 1e-20),
        'phi_deg': pattern.phi_deg[phi_zero_idx]
    }
    
    # H-plane cut (phi = 90°, vary theta) - or cut at peak theta
    h_plane = {
        'phi_deg': pattern.phi_deg,
        'gain_db': 10 * np.log10(pattern.gain_linear[peak_theta_idx, :] + 1e-20),
        'theta_deg': pattern.theta_deg[peak_theta_idx]
    }
    
    return {
        'e_plane': e_plane,
        'h_plane': h_plane,
        'peak_gain_dbi': 10 * np.log10(pattern.gain_linear[peak_idx]),
        'peak_theta_deg': pattern.theta_deg[peak_theta_idx],
        'peak_phi_deg': pattern.phi_deg[peak_phi_idx]
    }


def extract_beamwidths(pattern: AntennaPattern) -> dict:
    """
    Extract 3dB beamwidths from pattern.
    
    Returns:
        Dict with 'horizontal_deg' (H-plane) and 'vertical_deg' (E-plane)
    """
    # Find peak gain
    peak_idx = np.unravel_index(
        np.argmax(pattern.gain_linear), 
        pattern.gain_linear.shape
    )
    peak_gain = pattern.gain_linear[peak_idx]
    half_power = peak_gain / 2  # -3dB point
    
    # Horizontal beamwidth (phi variation at peak theta)
    h_cut = pattern.gain_linear[peak_idx[0], :]
    h_above = np.where(h_cut >= half_power)[0]
    if len(h_above) > 1:
        h_beamwidth = pattern.phi_deg[h_above[-1]] - pattern.phi_deg[h_above[0]]
    else:
        h_beamwidth = 0.0
    
    # Vertical beamwidth (theta variation at peak phi)
    v_cut = pattern.gain_linear[:, peak_idx[1]]
    v_above = np.where(v_cut >= half_power)[0]
    if len(v_above) > 1:
        v_beamwidth = pattern.theta_deg[v_above[-1]] - pattern.theta_deg[v_above[0]]
    else:
        v_beamwidth = 0.0
    
    return {
        'horizontal_deg': abs(h_beamwidth),
        'vertical_deg': abs(v_beamwidth),
        'peak_gain_dbi': 10 * np.log10(peak_gain) if peak_gain > 0 else -100
    }


def export_pattern_to_file(
    pattern: AntennaPattern,
    filepath: str,
    format: str = "npz"
) -> None:
    """
    Export antenna pattern to file.
    
    Args:
        pattern: AntennaPattern to export
        filepath: Output file path
        format: 'npz' (numpy) or 'csv'
    """
    if format == "npz":
        np.savez(
            filepath,
            frequency_hz=pattern.frequency_hz,
            theta_deg=pattern.theta_deg,
            phi_deg=pattern.phi_deg,
            gain_linear=pattern.gain_linear,
            phase_rad=pattern.phase_rad
        )
    elif format == "csv":
        # Export as flattened CSV
        theta_grid, phi_grid = np.meshgrid(
            pattern.theta_deg, pattern.phi_deg, indexing='ij'
        )
        data = np.column_stack([
            theta_grid.flatten(),
            phi_grid.flatten(),
            pattern.gain_linear.flatten(),
            pattern.phase_rad.flatten()
        ])
        header = "theta_deg,phi_deg,gain_linear,phase_rad"
        np.savetxt(filepath, data, delimiter=',', header=header)
    else:
        raise ValueError(f"Unknown format: {format}")


def load_pattern_from_file(filepath: str) -> AntennaPattern:
    """Load antenna pattern from NPZ file."""
    data = np.load(filepath)
    return AntennaPattern(
        frequency_hz=float(data['frequency_hz']),
        theta_deg=data['theta_deg'],
        phi_deg=data['phi_deg'],
        gain_linear=data['gain_linear'],
        phase_rad=data['phase_rad']
    )
