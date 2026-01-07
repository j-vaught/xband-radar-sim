"""Tests for slot array module."""
import pytest
import numpy as np
from src.antenna.slots import (
    SlotConfig,
    SlotArrayConfig,
    calculate_slot_positions,
    calculate_array_length,
    estimate_beamwidth,
    design_for_beamwidth,
)


class TestSlotConfig:
    """Tests for SlotConfig dataclass."""
    
    def test_default_values(self):
        """Default slot should have reasonable dimensions."""
        slot = SlotConfig()
        assert slot.length_mm > 0
        assert slot.width_mm > 0
        assert slot.offset_mm > 0


class TestSlotArrayConfig:
    """Tests for SlotArrayConfig dataclass."""
    
    def test_default_array(self):
        """Default array should have 25 slots."""
        config = SlotArrayConfig()
        assert config.n_slots == 25
        assert config.spacing_mm > 0
    
    def test_slot_auto_creation(self):
        """Slot should be auto-created if not provided."""
        config = SlotArrayConfig(n_slots=10)
        assert config.slot is not None
        assert isinstance(config.slot, SlotConfig)


class TestSlotPositions:
    """Tests for slot position calculations."""
    
    def test_position_count(self):
        """Should generate correct number of positions."""
        config = SlotArrayConfig(n_slots=25)
        positions = calculate_slot_positions(config)
        assert len(positions) == 25
    
    def test_alternating_offset(self):
        """Alternating offset should flip sign for phasing."""
        config = SlotArrayConfig(n_slots=4, alternating_offset=True)
        positions = calculate_slot_positions(config)
        # Even indices positive, odd indices negative
        assert positions[0]['x_offset'] > 0
        assert positions[1]['x_offset'] < 0
        assert positions[2]['x_offset'] > 0
        assert positions[3]['x_offset'] < 0
    
    def test_uniform_offset(self):
        """Non-alternating should have same offset for all."""
        config = SlotArrayConfig(n_slots=4, alternating_offset=False)
        positions = calculate_slot_positions(config)
        offsets = [p['x_offset'] for p in positions]
        assert all(o == offsets[0] for o in offsets)
    
    def test_z_spacing(self):
        """Z positions should be evenly spaced."""
        config = SlotArrayConfig(n_slots=3, spacing_mm=20.0, start_z_mm=50.0)
        positions = calculate_slot_positions(config)
        assert positions[0]['z_center'] == pytest.approx(50.0)
        assert positions[1]['z_center'] == pytest.approx(70.0)
        assert positions[2]['z_center'] == pytest.approx(90.0)
    
    def test_amplitude_weights(self):
        """Positions should include amplitude weights."""
        config = SlotArrayConfig(n_slots=5, taper_type="uniform")
        positions = calculate_slot_positions(config)
        assert all('amplitude_weight' in p for p in positions)


class TestArrayLength:
    """Tests for array length calculation."""
    
    def test_array_length(self):
        """Array length should be start + (n-1) * spacing."""
        config = SlotArrayConfig(n_slots=10, spacing_mm=20.0, start_z_mm=50.0)
        length = calculate_array_length(config)
        expected = 50.0 + 9 * 20.0  # 230 mm
        assert length == pytest.approx(expected)


class TestBeamwidth:
    """Tests for beamwidth estimation."""
    
    def test_beamwidth_estimate(self):
        """25 slots at 20mm spacing with 32mm wavelength → ~3-4° beamwidth."""
        bw = estimate_beamwidth(n_slots=25, spacing_mm=20.0, wavelength_mm=32.0)
        assert 2.5 < bw < 5.0  # Should be close to 3.9°
    
    def test_more_slots_narrower_beam(self):
        """More slots should give narrower beam."""
        bw_10 = estimate_beamwidth(10, 20.0, 32.0)
        bw_50 = estimate_beamwidth(50, 20.0, 32.0)
        assert bw_50 < bw_10


class TestDesignForBeamwidth:
    """Tests for beamwidth-based design."""
    
    def test_design_for_4deg(self):
        """Design for 4° beamwidth should give reasonable config."""
        config = design_for_beamwidth(
            target_beamwidth_deg=4.0,
            frequency_hz=9.41e9,
            guide_wavelength_m=0.044
        )
        assert config.n_slots > 10
        assert config.spacing_mm > 0
        assert config.slot is not None
