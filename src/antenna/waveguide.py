"""Rectangular waveguide geometry generation for openEMS."""
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional
import math

if TYPE_CHECKING:
    from CSXCAD import ContinuousStructure

# Physical constants
C0 = 299792458.0  # Speed of light m/s

# Standard X-band waveguide dimensions (WR-90)
WR90_WIDTH_MM = 22.86   # a dimension
WR90_HEIGHT_MM = 10.16  # b dimension


@dataclass
class WaveguideConfig:
    """Configuration for rectangular waveguide."""
    width_mm: float = WR90_WIDTH_MM
    height_mm: float = WR90_HEIGHT_MM
    length_mm: float = 500.0
    wall_thickness_mm: float = 2.0
    material: str = "PEC"  # Perfect Electric Conductor


def create_waveguide(
    csx: "ContinuousStructure",
    config: WaveguideConfig,
    name: str = "waveguide"
) -> None:
    """
    Create rectangular waveguide geometry in CSXCAD.
    
    Waveguide is centered at origin (x,y), extending in +z direction.
    Interior is air, walls are PEC.
    
    Args:
        csx: CSXCAD ContinuousStructure object
        config: Waveguide configuration
        name: Name prefix for geometry elements
    """
    a = config.width_mm
    b = config.height_mm
    L = config.length_mm
    t = config.wall_thickness_mm
    
    # Create metal for waveguide walls
    metal = csx.AddMetal(f'{name}_walls')
    
    # Bottom wall (y = -b/2 - t to -b/2)
    metal.AddBox(
        start=[-a/2 - t, -b/2 - t, 0],
        stop=[a/2 + t, -b/2, L],
        priority=5
    )
    
    # Top wall (y = b/2 to b/2 + t)
    metal.AddBox(
        start=[-a/2 - t, b/2, 0],
        stop=[a/2 + t, b/2 + t, L],
        priority=5
    )
    
    # Left wall (x = -a/2 - t to -a/2)
    metal.AddBox(
        start=[-a/2 - t, -b/2, 0],
        stop=[-a/2, b/2, L],
        priority=5
    )
    
    # Right wall (x = a/2 to a/2 + t)
    metal.AddBox(
        start=[a/2, -b/2, 0],
        stop=[a/2 + t, b/2, L],
        priority=5
    )


def get_cutoff_frequency(config: WaveguideConfig, mode: str = "TE10") -> float:
    """
    Calculate cutoff frequency for given mode.
    
    Args:
        config: Waveguide configuration
        mode: Mode designation (e.g., "TE10", "TE20", "TE01")
    
    Returns:
        Cutoff frequency in Hz
    """
    import re
    
    # Parse mode string
    match = re.match(r"TE(\d)(\d)", mode)
    if not match:
        raise ValueError(f"Invalid mode: {mode}. Use format 'TExy' e.g. 'TE10'")
    
    m = int(match.group(1))
    n = int(match.group(2))
    
    # Convert to meters
    a = config.width_mm / 1000
    b = config.height_mm / 1000
    
    # Cutoff frequency: fc = c/(2π) * sqrt((mπ/a)² + (nπ/b)²)
    # Simplifies to: fc = c/2 * sqrt((m/a)² + (n/b)²)
    fc = (C0 / 2) * math.sqrt((m/a)**2 + (n/b)**2)
    return fc


def get_guide_wavelength(config: WaveguideConfig, freq_hz: float, mode: str = "TE10") -> float:
    """
    Calculate guide wavelength at given frequency.
    
    λg = λ₀ / sqrt(1 - (fc/f)²)
    
    Args:
        config: Waveguide configuration
        freq_hz: Operating frequency in Hz
        mode: Propagation mode
    
    Returns:
        Guide wavelength in meters
    """
    lambda_0 = C0 / freq_hz  # Free-space wavelength
    fc = get_cutoff_frequency(config, mode)
    
    if freq_hz <= fc:
        raise ValueError(
            f"Frequency {freq_hz/1e9:.2f} GHz is below cutoff {fc/1e9:.2f} GHz for {mode}"
        )
    
    lambda_g = lambda_0 / math.sqrt(1 - (fc/freq_hz)**2)
    return lambda_g


def get_wave_impedance(config: WaveguideConfig, freq_hz: float, mode: str = "TE10") -> float:
    """
    Calculate wave impedance in waveguide.
    
    For TE modes: Zg = η₀ / sqrt(1 - (fc/f)²)
    where η₀ = 377Ω (free-space impedance)
    
    Args:
        config: Waveguide configuration
        freq_hz: Operating frequency in Hz
        mode: Propagation mode
    
    Returns:
        Wave impedance in ohms
    """
    ETA_0 = 376.730313668  # Free-space impedance
    fc = get_cutoff_frequency(config, mode)
    
    if freq_hz <= fc:
        raise ValueError(f"Frequency below cutoff for {mode}")
    
    Zg = ETA_0 / math.sqrt(1 - (fc/freq_hz)**2)
    return Zg


def validate_operating_frequency(config: WaveguideConfig, freq_hz: float) -> dict:
    """
    Validate that frequency is suitable for single-mode operation.
    
    For WR-90: TE10 cutoff ~6.56 GHz, TE20 cutoff ~13.1 GHz
    Recommended operating range: 8.2 - 12.4 GHz
    
    Returns:
        Dict with validation results
    """
    fc_te10 = get_cutoff_frequency(config, "TE10")
    fc_te20 = get_cutoff_frequency(config, "TE20")
    fc_te01 = get_cutoff_frequency(config, "TE01")
    
    # Next higher mode cutoff
    fc_next = min(fc_te20, fc_te01)
    
    results = {
        'frequency_hz': freq_hz,
        'te10_cutoff_hz': fc_te10,
        'next_mode_cutoff_hz': fc_next,
        'above_cutoff': freq_hz > fc_te10,
        'single_mode': fc_te10 < freq_hz < fc_next,
        'recommended': 1.25 * fc_te10 < freq_hz < 0.95 * fc_next,
    }
    
    if results['single_mode']:
        results['guide_wavelength_m'] = get_guide_wavelength(config, freq_hz)
        results['wave_impedance_ohm'] = get_wave_impedance(config, freq_hz)
    
    return results
