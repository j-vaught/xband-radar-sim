"""Integration of scattering with ray tracing."""
import numpy as np
from typing import Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from interfaces import RayBundle, ScatterResult
from config import RadarConfig
from scattering.physical_optics import compute_po_rcs
from scattering.analytical import corner_reflector_rcs


def compute_scattering(
    ray_bundle: RayBundle,
    config: RadarConfig,
    mesh_vertices: Optional[np.ndarray] = None,
    mesh_faces: Optional[np.ndarray] = None,
    use_po: bool = True
) -> ScatterResult:
    """
    Compute scattering for all ray hits.
    
    Args:
        ray_bundle: RayBundle from propagation
        config: Radar configuration
        mesh_vertices: Target mesh vertices (optional)
        mesh_faces: Target mesh faces (optional)
        use_po: Use Physical Optics (True) or analytical fallback
        
    Returns:
        ScatterResult for signal processing
    """
    n_hits = int(np.sum(ray_bundle.hit_mask))
    hit_indices = np.where(ray_bundle.hit_mask)[0]
    
    if n_hits == 0:
        return ScatterResult(
            n_scatterers=0,
            scatter_points=np.empty((0, 3)),
            rcs_m2=np.empty(0),
            scattered_power_w=np.empty(0),
            path_to_rx_m=np.empty(0),
            phase_rad=np.empty(0)
        )
    
    # Initialize arrays
    rcs_values = np.zeros(n_hits)
    scattered_power = np.zeros(n_hits)
    phases = np.zeros(n_hits)
    
    k = config.wavenumber
    
    # Compute RCS for each hit
    if use_po and mesh_vertices is not None and mesh_faces is not None:
        # Use Physical Optics for each distinct incidence angle
        for i, hit_idx in enumerate(hit_indices):
            direction = ray_bundle.directions[hit_idx]
            
            # Convert direction to spherical angles
            theta = np.arccos(np.clip(direction[2], -1, 1))
            phi = np.arctan2(direction[1], direction[0])
            
            rcs_values[i] = compute_po_rcs(
                mesh_vertices, mesh_faces, k, theta, phi
            )
    else:
        # Use analytical RCS (assume corner reflector)
        analytical_rcs = corner_reflector_rcs(0.0762, config.wavelength_m)
        rcs_values[:] = analytical_rcs
    
    # Compute scattered power using radar equation
    for i, hit_idx in enumerate(hit_indices):
        R = ray_bundle.path_lengths_m[hit_idx]
        P_inc = ray_bundle.incident_powers_w[hit_idx]
        
        # Scattered power = incident_power × RCS / (4π R²)
        # This is power scattered back toward radar
        scattered_power[i] = P_inc * rcs_values[i] / (4 * np.pi * R**2)
        
        # Phase from round-trip path
        phases[i] = 2 * k * R
    
    return ScatterResult(
        n_scatterers=n_hits,
        scatter_points=ray_bundle.hit_points[ray_bundle.hit_mask],
        rcs_m2=rcs_values,
        scattered_power_w=scattered_power,
        path_to_rx_m=ray_bundle.path_lengths_m[ray_bundle.hit_mask],
        phase_rad=phases
    )


def compute_point_target_scatter(
    position: np.ndarray,
    rcs_m2: float,
    config: RadarConfig,
    tx_position: np.ndarray = np.array([0, 0, 5])
) -> ScatterResult:
    """
    Compute scattering from a single point target.
    
    Args:
        position: (3,) target position
        rcs_m2: Target RCS in m²
        config: Radar configuration
        tx_position: Transmitter position
        
    Returns:
        ScatterResult
    """
    # Compute range
    range_vec = position - tx_position
    range_m = np.linalg.norm(range_vec)
    
    # Compute received power using radar equation
    G = 10 ** (config.antenna_gain_dbi / 10)  # Linear gain
    wavelength = config.wavelength_m
    
    # Pr = (Pt × G² × λ² × σ) / ((4π)³ × R⁴)
    Pr = (config.tx_power_w * G**2 * wavelength**2 * rcs_m2) / \
         ((4 * np.pi)**3 * range_m**4)
    
    # Phase from two-way path
    phase = 2 * config.wavenumber * range_m
    
    return ScatterResult(
        n_scatterers=1,
        scatter_points=position.reshape(1, 3),
        rcs_m2=np.array([rcs_m2]),
        scattered_power_w=np.array([Pr]),
        path_to_rx_m=np.array([range_m]),
        phase_rad=np.array([phase])
    )


def combine_scatter_results(*results: ScatterResult) -> ScatterResult:
    """
    Combine multiple scatter results into one.
    
    Args:
        *results: Variable number of ScatterResult objects
        
    Returns:
        Combined ScatterResult
    """
    if not results:
        return ScatterResult(
            n_scatterers=0,
            scatter_points=np.empty((0, 3)),
            rcs_m2=np.empty(0),
            scattered_power_w=np.empty(0),
            path_to_rx_m=np.empty(0),
            phase_rad=np.empty(0)
        )
    
    scatter_points = np.vstack([r.scatter_points for r in results if r.n_scatterers > 0])
    rcs_m2 = np.concatenate([r.rcs_m2 for r in results if r.n_scatterers > 0])
    scattered_power = np.concatenate([r.scattered_power_w for r in results if r.n_scatterers > 0])
    path_to_rx = np.concatenate([r.path_to_rx_m for r in results if r.n_scatterers > 0])
    phase = np.concatenate([r.phase_rad for r in results if r.n_scatterers > 0])
    
    return ScatterResult(
        n_scatterers=len(rcs_m2),
        scatter_points=scatter_points,
        rcs_m2=rcs_m2,
        scattered_power_w=scattered_power,
        path_to_rx_m=path_to_rx,
        phase_rad=phase
    )


def export_scatter_result(result: ScatterResult, filepath: str) -> None:
    """Export ScatterResult to pickle file."""
    import pickle
    with open(filepath, 'wb') as f:
        pickle.dump(result, f)


def load_scatter_result(filepath: str) -> ScatterResult:
    """Load ScatterResult from pickle file."""
    import pickle
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def summarize_scattering(result: ScatterResult) -> dict:
    """
    Generate summary statistics for ScatterResult.
    
    Returns:
        Dictionary with statistics
    """
    if result.n_scatterers == 0:
        return {
            'n_scatterers': 0,
            'total_rcs_m2': 0.0,
            'mean_rcs_dbsm': None,
            'total_scattered_power_w': 0.0,
            'min_range_m': None,
            'max_range_m': None
        }
    
    return {
        'n_scatterers': result.n_scatterers,
        'total_rcs_m2': float(np.sum(result.rcs_m2)),
        'mean_rcs_dbsm': float(10 * np.log10(np.mean(result.rcs_m2) + 1e-20)),
        'total_scattered_power_w': float(np.sum(result.scattered_power_w)),
        'min_range_m': float(np.min(result.path_to_rx_m)),
        'max_range_m': float(np.max(result.path_to_rx_m))
    }
