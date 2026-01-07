"""Slot array generation for slotted waveguide antenna."""
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional
import numpy as np

if TYPE_CHECKING:
    from CSXCAD import ContinuousStructure


@dataclass
class SlotConfig:
    """Configuration for individual slot."""
    length_mm: float = 15.0      # ~λ/2 at 9.4 GHz
    width_mm: float = 2.0        # Narrow slot
    offset_mm: float = 2.5       # Offset from waveguide centerline


@dataclass
class SlotArrayConfig:
    """Configuration for slot array."""
    n_slots: int = 25
    spacing_mm: float = 20.0     # ~λg/2 at 9.4 GHz
    start_z_mm: float = 50.0     # Distance from feed end
    slot: SlotConfig = field(default_factory=SlotConfig)
    alternating_offset: bool = True  # Alternate +/- offset for phasing
    taper_type: str = "uniform"  # "uniform", "taylor", "cosine"


def calculate_slot_positions(config: SlotArrayConfig) -> List[dict]:
    """
    Calculate positions for all slots in array.
    
    Returns:
        List of dicts with: slot_index, z_center, x_offset, amplitude_weight
    """
    positions = []
    
    # Compute amplitude taper for sidelobe control
    weights = _compute_taper_weights(config.n_slots, config.taper_type)
    
    for i in range(config.n_slots):
        z_center = config.start_z_mm + i * config.spacing_mm
        
        # Alternating offset for proper phasing (180° between slots)
        if config.alternating_offset:
            x_offset = config.slot.offset_mm if i % 2 == 0 else -config.slot.offset_mm
        else:
            x_offset = config.slot.offset_mm
        
        positions.append({
            'slot_index': i,
            'z_center': z_center,
            'x_offset': x_offset,
            'amplitude_weight': weights[i]
        })
    
    return positions


def _compute_taper_weights(n_slots: int, taper_type: str) -> np.ndarray:
    """Compute amplitude taper weights for sidelobe control."""
    if taper_type == "uniform":
        return np.ones(n_slots)
    
    elif taper_type == "cosine":
        # Raised cosine taper
        x = np.linspace(-1, 1, n_slots)
        return 0.5 * (1 + np.cos(np.pi * x))
    
    elif taper_type == "taylor":
        # Simplified Taylor taper for -25dB sidelobes
        x = np.linspace(-1, 1, n_slots)
        return 1 - 0.4 * x**2
    
    else:
        raise ValueError(f"Unknown taper type: {taper_type}")


def add_slots_to_waveguide(
    csx: "ContinuousStructure",
    wg_height_mm: float,
    wg_wall_thickness_mm: float,
    array_config: SlotArrayConfig,
    name: str = "slot"
) -> int:
    """
    Add slot apertures to waveguide top wall.
    
    Slots are cut through the top wall by adding high-priority air boxes.
    
    Args:
        csx: CSXCAD object with waveguide already created
        wg_height_mm: Waveguide height (b dimension)
        wg_wall_thickness_mm: Wall thickness
        array_config: Slot array configuration
        name: Name prefix for slot elements
    
    Returns:
        Number of slots created
    """
    slot = array_config.slot
    positions = calculate_slot_positions(array_config)
    
    # Slot cuts through top wall - using high priority
    # y range: from inside wall to outside
    y_start = wg_height_mm / 2 - 0.5  # Slightly into waveguide
    y_stop = wg_height_mm / 2 + wg_wall_thickness_mm + 0.5  # Through wall
    
    for pos in positions:
        z = pos['z_center']
        x = pos['x_offset']
        idx = pos['slot_index']
        
        # Scale slot length by amplitude weight for tapered arrays
        eff_length = slot.length_mm * pos['amplitude_weight']
        
        # Add slot as metal removal (use AddBox with priority > waveguide)
        # Note: In openEMS, we create an "air" box with higher priority
        # that overwrites the metal
        slot_material = csx.AddMaterial(f'{name}_{idx}', epsilon=1.0)
        slot_material.AddBox(
            start=[x - eff_length/2, y_start, z - slot.width_mm/2],
            stop=[x + eff_length/2, y_stop, z + slot.width_mm/2],
            priority=10  # Higher than waveguide walls (5)
        )
    
    return len(positions)


def calculate_array_length(config: SlotArrayConfig) -> float:
    """Calculate total length of slot array in mm."""
    return config.start_z_mm + (config.n_slots - 1) * config.spacing_mm


def estimate_beamwidth(
    n_slots: int,
    spacing_mm: float,
    wavelength_mm: float
) -> float:
    """
    Estimate 3dB beamwidth for uniform linear array.
    
    Approximate formula: θ_3dB ≈ 0.886 × λ / (N × d)
    
    Args:
        n_slots: Number of slots
        spacing_mm: Slot spacing in mm
        wavelength_mm: Free-space wavelength in mm
    
    Returns:
        Estimated beamwidth in degrees
    """
    array_length_mm = n_slots * spacing_mm
    beamwidth_rad = 0.886 * wavelength_mm / array_length_mm
    return np.rad2deg(beamwidth_rad)


def design_for_beamwidth(
    target_beamwidth_deg: float,
    frequency_hz: float,
    guide_wavelength_m: float
) -> SlotArrayConfig:
    """
    Design slot array for target beamwidth.
    
    Args:
        target_beamwidth_deg: Desired 3dB beamwidth
        frequency_hz: Operating frequency
        guide_wavelength_m: Guide wavelength in waveguide
    
    Returns:
        SlotArrayConfig with computed parameters
    """
    C0 = 299792458.0
    lambda_0_mm = (C0 / frequency_hz) * 1000  # Free-space wavelength in mm
    lambda_g_mm = guide_wavelength_m * 1000   # Guide wavelength in mm
    
    # Slot spacing = λg/2 for in-phase addition
    spacing_mm = lambda_g_mm / 2
    
    # Array length needed: L = 0.886 × λ / θ
    target_rad = np.deg2rad(target_beamwidth_deg)
    array_length_mm = 0.886 * lambda_0_mm / target_rad
    
    # Number of slots (round up)
    n_slots = int(np.ceil(array_length_mm / spacing_mm))
    
    # Slot length ~ λ₀/2
    slot_length_mm = lambda_0_mm / 2
    
    return SlotArrayConfig(
        n_slots=n_slots,
        spacing_mm=spacing_mm,
        start_z_mm=50.0,
        slot=SlotConfig(length_mm=slot_length_mm),
        alternating_offset=True
    )
