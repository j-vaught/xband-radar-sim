"""Antenna validation and full simulation runner."""
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple
import os
import tempfile
import numpy as np

if TYPE_CHECKING:
    from openEMS import openEMS
    from CSXCAD import ContinuousStructure

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from interfaces import AntennaPattern
from config import RadarConfig

from .waveguide import WaveguideConfig, create_waveguide, get_guide_wavelength, validate_operating_frequency
from .slots import SlotArrayConfig, SlotConfig, add_slots_to_waveguide, calculate_array_length
from .excitation import ExcitationConfig, setup_gaussian_excitation, create_feed_port, setup_boundaries
from .farfield import FarFieldConfig, setup_nf2ff_box, compute_farfield, extract_beamwidths


@dataclass
class AntennaSimConfig:
    """Complete antenna simulation configuration."""
    waveguide: WaveguideConfig = None
    slots: SlotArrayConfig = None
    excitation: ExcitationConfig = None
    farfield: FarFieldConfig = None
    
    # Mesh settings
    mesh_res_mm: float = 1.0      # Base mesh resolution
    mesh_ratio: float = 1.5       # Graded mesh ratio
    
    # Simulation settings
    end_criteria: float = 1e-5    # Energy decay for termination
    max_timesteps: int = 100000   # Maximum timesteps
    
    def __post_init__(self):
        if self.waveguide is None:
            self.waveguide = WaveguideConfig()
        if self.slots is None:
            self.slots = SlotArrayConfig()
        if self.excitation is None:
            self.excitation = ExcitationConfig()
        if self.farfield is None:
            self.farfield = FarFieldConfig()


def create_slotted_waveguide_antenna(
    config: AntennaSimConfig
) -> Tuple["ContinuousStructure", "openEMS"]:
    """
    Create complete slotted waveguide antenna model.
    
    Args:
        config: Complete simulation configuration
    
    Returns:
        (csx, fdtd) tuple - CSXCAD structure and openEMS FDTD object
    """
    from CSXCAD import CSXCAD
    from openEMS import openEMS as oEMS
    
    # Create CSXCAD structure
    csx = CSXCAD.ContinuousStructure()
    
    # Create waveguide
    wg_length = calculate_array_length(config.slots) + 100  # Extra length
    wg_config = WaveguideConfig(
        width_mm=config.waveguide.width_mm,
        height_mm=config.waveguide.height_mm,
        length_mm=wg_length,
        wall_thickness_mm=config.waveguide.wall_thickness_mm
    )
    create_waveguide(csx, wg_config)
    
    # Add slots
    add_slots_to_waveguide(
        csx,
        wg_height_mm=config.waveguide.height_mm,
        wg_wall_thickness_mm=config.waveguide.wall_thickness_mm,
        array_config=config.slots
    )
    
    # Setup mesh
    mesh = csx.GetGrid()
    mesh.SetDeltaUnit(1e-3)  # mm units
    
    # Generate graded mesh
    _generate_mesh(mesh, config)
    
    # Create FDTD
    fdtd = oEMS(NrTS=config.max_timesteps, EndCriteria=config.end_criteria)
    fdtd.SetCSX(csx)
    
    # Setup excitation
    setup_gaussian_excitation(fdtd, config.excitation)
    
    # Setup boundaries (PML absorbing)
    setup_boundaries(fdtd, absorbing=True)
    
    # Add feed port
    create_feed_port(
        fdtd,
        wg_width_mm=config.waveguide.width_mm,
        wg_height_mm=config.waveguide.height_mm,
        z_position_mm=5.0,
        config=config.excitation
    )
    
    # Setup NF2FF box
    setup_nf2ff_box(fdtd)
    
    return csx, fdtd


def _generate_mesh(mesh, config: AntennaSimConfig) -> None:
    """Generate graded mesh for antenna simulation."""
    wg = config.waveguide
    slots = config.slots
    res = config.mesh_res_mm
    
    # Compute extents
    x_min = -wg.width_mm/2 - wg.wall_thickness_mm - 20
    x_max = wg.width_mm/2 + wg.wall_thickness_mm + 20
    y_min = -wg.height_mm/2 - wg.wall_thickness_mm - 20
    y_max = wg.height_mm/2 + wg.wall_thickness_mm + 50  # Extra for radiation
    z_min = -20
    z_max = calculate_array_length(slots) + 100
    
    # Simple uniform mesh (for production, use graded mesh)
    mesh.AddLine('x', np.arange(x_min, x_max, res))
    mesh.AddLine('y', np.arange(y_min, y_max, res))
    mesh.AddLine('z', np.arange(z_min, z_max, res))
    
    # Smooth mesh
    mesh.SmoothMeshLines('all', res * config.mesh_ratio)


def run_antenna_simulation(
    config: AntennaSimConfig,
    sim_path: Optional[str] = None,
    verbose: bool = True
) -> AntennaPattern:
    """
    Run complete antenna simulation and extract pattern.
    
    Args:
        config: Simulation configuration
        sim_path: Directory for simulation files (temp if None)
        verbose: Print progress information
    
    Returns:
        AntennaPattern from simulation
    """
    # Create model
    csx, fdtd = create_slotted_waveguide_antenna(config)
    
    # Use temp directory if not specified
    if sim_path is None:
        sim_path = tempfile.mkdtemp(prefix="antenna_sim_")
    
    if verbose:
        print(f"Running simulation in: {sim_path}")
    
    # Run FDTD
    fdtd.Run(sim_path, verbose=1 if verbose else 0)
    
    # Get NF2FF box and compute far-field
    nf2ff = fdtd.CreateNF2FFBox()
    pattern = compute_farfield(nf2ff, sim_path, config.farfield)
    
    if verbose:
        bw = extract_beamwidths(pattern)
        print(f"Simulation complete:")
        print(f"  Peak gain: {bw['peak_gain_dbi']:.1f} dBi")
        print(f"  H-plane beamwidth: {bw['horizontal_deg']:.1f}°")
        print(f"  V-plane beamwidth: {bw['vertical_deg']:.1f}°")
    
    return pattern


def validate_antenna_design(
    config: AntennaSimConfig,
    radar_config: Optional[RadarConfig] = None
) -> dict:
    """
    Validate antenna design against specifications.
    
    Args:
        config: Antenna configuration
        radar_config: Radar specs to validate against
    
    Returns:
        Dict with validation results
    """
    if radar_config is None:
        radar_config = RadarConfig()
    
    results = {
        'frequency_hz': radar_config.center_frequency_hz,
        'checks': []
    }
    
    # Check waveguide operating frequency
    wg_check = validate_operating_frequency(
        config.waveguide, 
        radar_config.center_frequency_hz
    )
    results['waveguide'] = wg_check
    
    if not wg_check['single_mode']:
        results['checks'].append({
            'name': 'Single mode operation',
            'passed': False,
            'message': f"Frequency outside single-mode range"
        })
    else:
        results['checks'].append({
            'name': 'Single mode operation',
            'passed': True,
            'message': f"TE10 mode propagates correctly"
        })
    
    # Check slot spacing vs guide wavelength
    if wg_check['single_mode']:
        lambda_g_mm = wg_check['guide_wavelength_m'] * 1000
        spacing_ratio = config.slots.spacing_mm / (lambda_g_mm / 2)
        
        results['slot_spacing'] = {
            'spacing_mm': config.slots.spacing_mm,
            'lambda_g_half_mm': lambda_g_mm / 2,
            'ratio': spacing_ratio
        }
        
        if 0.8 < spacing_ratio < 1.2:
            results['checks'].append({
                'name': 'Slot spacing',
                'passed': True,
                'message': f"Spacing is {spacing_ratio:.2f} × λg/2"
            })
        else:
            results['checks'].append({
                'name': 'Slot spacing',
                'passed': False,
                'message': f"Spacing {spacing_ratio:.2f} × λg/2 (should be ~1.0)"
            })
    
    # Estimate beamwidth
    from .slots import estimate_beamwidth
    lambda_0_mm = (299792458.0 / radar_config.center_frequency_hz) * 1000
    est_bw = estimate_beamwidth(
        config.slots.n_slots,
        config.slots.spacing_mm,
        lambda_0_mm
    )
    
    results['estimated_beamwidth_deg'] = est_bw
    bw_error = abs(est_bw - radar_config.horizontal_beamwidth_deg)
    
    if bw_error < 1.0:
        results['checks'].append({
            'name': 'Beamwidth target',
            'passed': True,
            'message': f"Estimated {est_bw:.1f}° vs target {radar_config.horizontal_beamwidth_deg}°"
        })
    else:
        results['checks'].append({
            'name': 'Beamwidth target',
            'passed': False,
            'message': f"Estimated {est_bw:.1f}° vs target {radar_config.horizontal_beamwidth_deg}° (Δ={bw_error:.1f}°)"
        })
    
    # Overall pass/fail
    results['all_passed'] = all(c['passed'] for c in results['checks'])
    
    return results


def create_default_config_for_furuno() -> AntennaSimConfig:
    """
    Create default antenna configuration for Furuno DRS4D-NXT.
    
    Based on specs:
    - 9.41 GHz
    - 3.9° horizontal beamwidth
    - 25° vertical beamwidth
    """
    # Get guide wavelength at 9.41 GHz
    wg = WaveguideConfig()
    lambda_g = get_guide_wavelength(wg, 9.41e9)
    lambda_g_mm = lambda_g * 1000
    
    # Free-space wavelength
    lambda_0_mm = (299792458.0 / 9.41e9) * 1000  # ~31.9 mm
    
    # Design for 3.9° beamwidth
    # Array length = 0.886 × λ / θ
    target_bw_rad = np.deg2rad(3.9)
    array_length_mm = 0.886 * lambda_0_mm / target_bw_rad
    
    # Slot spacing = λg/2
    spacing_mm = lambda_g_mm / 2
    n_slots = int(np.ceil(array_length_mm / spacing_mm))
    
    return AntennaSimConfig(
        waveguide=WaveguideConfig(
            length_mm=array_length_mm + 100
        ),
        slots=SlotArrayConfig(
            n_slots=n_slots,
            spacing_mm=spacing_mm,
            slot=SlotConfig(
                length_mm=lambda_0_mm / 2,  # λ/2 slot
                width_mm=2.0,
                offset_mm=2.5
            ),
            alternating_offset=True,
            taper_type="taylor"  # For low sidelobes
        ),
        excitation=ExcitationConfig(
            center_freq_hz=9.41e9,
            bandwidth_hz=2e9
        ),
        farfield=FarFieldConfig(
            frequency_hz=9.41e9,
            theta_range=(-90, 90),
            phi_range=(-180, 180),
            theta_step=0.5,
            phi_step=0.5
        ),
        mesh_res_mm=1.0,
        end_criteria=1e-5
    )
