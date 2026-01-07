"""Waveguide excitation and port setup for openEMS."""
from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple, Optional

if TYPE_CHECKING:
    from openEMS import openEMS


@dataclass
class ExcitationConfig:
    """Configuration for waveguide excitation."""
    center_freq_hz: float = 9.41e9
    bandwidth_hz: float = 2e9      # Gaussian pulse -20dB bandwidth
    port_impedance: float = 50.0   # Reference impedance for S-params


def setup_gaussian_excitation(
    fdtd: "openEMS",
    config: ExcitationConfig
) -> None:
    """
    Configure Gaussian pulse excitation centered at operating frequency.
    
    Args:
        fdtd: openEMS FDTD object
        config: Excitation configuration
    """
    fdtd.SetGaussExcite(config.center_freq_hz, config.bandwidth_hz)


def setup_sinusoidal_excitation(
    fdtd: "openEMS",
    frequency_hz: float
) -> None:
    """
    Configure sinusoidal (CW) excitation at single frequency.
    
    Args:
        fdtd: openEMS FDTD object
        frequency_hz: Excitation frequency
    """
    fdtd.SetSinusExcite(frequency_hz)


def add_waveguide_port(
    fdtd: "openEMS",
    port_number: int,
    start: Tuple[float, float, float],
    stop: Tuple[float, float, float],
    direction: str = 'z',
    excite: bool = True,
    mode: str = "TE10"
) -> object:
    """
    Add waveguide port for S-parameter calculation.
    
    Uses MSL (microstrip line) port approximation for rectangular waveguide.
    
    Args:
        fdtd: openEMS FDTD object
        port_number: Unique port identifier (1-based)
        start: Port start coordinates (x, y, z) in mm
        stop: Port stop coordinates (x, y, z) in mm
        direction: Propagation direction ('x', 'y', or 'z')
        excite: Whether this port is excited (source)
        mode: Waveguide mode (currently only TE10)
    
    Returns:
        Port object for later S-parameter extraction
    """
    # For waveguide, we use a lumped port across the aperture
    # This is an approximation - proper waveguide ports need mode matching
    
    port = fdtd.AddRectWaveGuidePort(
        port_nr=port_number,
        start=list(start),
        stop=list(stop),
        dir=direction,
        a=abs(stop[0] - start[0]),  # Width
        b=abs(stop[1] - start[1]),  # Height
        mode_name='TE10',
        excite=1.0 if excite else 0.0
    )
    return port


def add_lumped_port(
    fdtd: "openEMS",
    port_number: int,
    start: Tuple[float, float, float],
    stop: Tuple[float, float, float],
    direction: str = 'z',
    impedance: float = 50.0,
    excite: bool = True
) -> object:
    """
    Add lumped port (simpler alternative to waveguide port).
    
    Args:
        fdtd: openEMS FDTD object
        port_number: Unique port identifier
        start: Port start coordinates (x, y, z) in mm
        stop: Port stop coordinates (x, y, z) in mm
        direction: Field direction ('x', 'y', or 'z')
        impedance: Reference impedance in ohms
        excite: Whether this port is excited
    
    Returns:
        Port object
    """
    port = fdtd.AddLumpedPort(
        port_nr=port_number,
        R=impedance,
        start=list(start),
        stop=list(stop),
        p_dir=direction,
        excite=1.0 if excite else 0.0
    )
    return port


def create_feed_port(
    fdtd: "openEMS",
    wg_width_mm: float,
    wg_height_mm: float,
    z_position_mm: float = 0.0,
    config: Optional[ExcitationConfig] = None
) -> object:
    """
    Create feed port at waveguide input.
    
    Args:
        fdtd: openEMS FDTD object
        wg_width_mm: Waveguide width (a dimension)
        wg_height_mm: Waveguide height (b dimension)
        z_position_mm: Z position of port (0 = waveguide start)
        config: Excitation configuration
    
    Returns:
        Port object
    """
    if config is None:
        config = ExcitationConfig()
    
    return add_waveguide_port(
        fdtd=fdtd,
        port_number=1,
        start=(-wg_width_mm/2, -wg_height_mm/2, z_position_mm),
        stop=(wg_width_mm/2, wg_height_mm/2, z_position_mm),
        direction='z',
        excite=True,
        mode="TE10"
    )


def create_matched_load_port(
    fdtd: "openEMS",
    wg_width_mm: float,
    wg_height_mm: float,
    z_position_mm: float,
    wave_impedance: float = 500.0
) -> object:
    """
    Create matched load at waveguide end to prevent reflections.
    
    Args:
        fdtd: openEMS FDTD object
        wg_width_mm: Waveguide width
        wg_height_mm: Waveguide height
        z_position_mm: Z position (end of waveguide)
        wave_impedance: Waveguide wave impedance
    
    Returns:
        Port object (non-excited)
    """
    return add_lumped_port(
        fdtd=fdtd,
        port_number=2,
        start=(-wg_width_mm/2, -wg_height_mm/2, z_position_mm),
        stop=(wg_width_mm/2, wg_height_mm/2, z_position_mm),
        direction='z',
        impedance=wave_impedance,
        excite=False
    )


def setup_boundaries(fdtd: "openEMS", absorbing: bool = True) -> None:
    """
    Configure simulation boundary conditions.
    
    Args:
        fdtd: openEMS FDTD object
        absorbing: Use absorbing (PML) boundaries if True, else PEC
    """
    if absorbing:
        # MUR absorbing boundary (simple) or PML (better)
        fdtd.SetBoundaryCond(['PML_8', 'PML_8', 'PML_8', 'PML_8', 'PML_8', 'PML_8'])
    else:
        # Perfect Electric Conductor on all boundaries
        fdtd.SetBoundaryCond(['PEC', 'PEC', 'PEC', 'PEC', 'PEC', 'PEC'])
