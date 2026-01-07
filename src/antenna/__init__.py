"""Antenna simulation modules using openEMS."""
from .waveguide import WaveguideConfig, create_waveguide, get_cutoff_frequency, get_guide_wavelength
from .slots import SlotConfig, SlotArrayConfig, calculate_slot_positions, add_slots_to_waveguide
from .excitation import ExcitationConfig, setup_gaussian_excitation, create_feed_port
from .farfield import FarFieldConfig, compute_farfield, extract_beamwidths
from .validation import AntennaSimConfig, validate_antenna_design, create_default_config_for_furuno

__all__ = [
    'WaveguideConfig', 'create_waveguide', 'get_cutoff_frequency', 'get_guide_wavelength',
    'SlotConfig', 'SlotArrayConfig', 'calculate_slot_positions', 'add_slots_to_waveguide',
    'ExcitationConfig', 'setup_gaussian_excitation', 'create_feed_port',
    'FarFieldConfig', 'compute_farfield', 'extract_beamwidths',
    'AntennaSimConfig', 'validate_antenna_design', 'create_default_config_for_furuno',
]
