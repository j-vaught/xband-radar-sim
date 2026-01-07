"""Integration tests for end-to-end pipeline."""
import pytest
import numpy as np
from src.config import RadarConfig
from src.pipeline import (
    run_simple_simulation,
    simulate_target_return,
    RadarSimulator,
)


class TestSimulateTargetReturn:
    """Tests for target return simulation."""
    
    def test_return_signal_exists(self):
        """Should generate non-zero return signal."""
        config = RadarConfig()
        return_signal = simulate_target_return(
            config=config,
            target_range_m=1000,
            target_rcs_m2=0.1
        )
        assert len(return_signal) > 0
        assert np.max(np.abs(return_signal)) > 0
    
    def test_closer_target_stronger(self):
        """Closer target should have stronger return."""
        config = RadarConfig()
        
        return_1km = simulate_target_return(config, 1000, 0.1)
        return_2km = simulate_target_return(config, 2000, 0.1)
        
        power_1km = np.max(np.abs(return_1km)**2)
        power_2km = np.max(np.abs(return_2km)**2)
        
        assert power_1km > power_2km
    
    def test_larger_rcs_stronger(self):
        """Larger RCS should give stronger return."""
        config = RadarConfig()
        
        return_small = simulate_target_return(config, 1000, 0.01)
        return_large = simulate_target_return(config, 1000, 1.0)
        
        power_small = np.max(np.abs(return_small)**2)
        power_large = np.max(np.abs(return_large)**2)
        
        assert power_large > power_small


class TestRunSimpleSimulation:
    """Tests for simplified simulation interface."""
    
    def test_simple_sim_returns_result(self):
        """Simple simulation should return valid result."""
        result = run_simple_simulation(
            target_range_m=1000,
            target_rcs_m2=0.1
        )
        
        assert result is not None
        assert hasattr(result, 'range_bins_m')
        assert hasattr(result, 'amplitude_db')
        assert hasattr(result, 'snr_db')
        assert hasattr(result, 'detected_ranges_m')
    
    def test_simple_sim_detects_target(self):
        """Should detect target at specified range."""
        result = run_simple_simulation(
            target_range_m=1500,
            target_rcs_m2=1.0  # Large RCS for reliable detection
        )
        
        # Should have at least one detection
        assert len(result.detected_ranges_m) >= 1
        
        # Detection should be near target range
        closest = min(result.detected_ranges_m, key=lambda x: abs(x - 1500))
        assert abs(closest - 1500) < 50  # Within 50m
    
    def test_simple_sim_positive_snr(self):
        """Strong target should have positive SNR."""
        result = run_simple_simulation(
            target_range_m=1000,
            target_rcs_m2=1.0
        )
        assert result.snr_db > 0


@pytest.mark.slow
class TestRadarSimulator:
    """Tests for full RadarSimulator class."""
    
    def test_simulator_initialization(self):
        """Simulator should initialize with config."""
        config = RadarConfig()
        sim = RadarSimulator(config)
        assert sim.config == config
    
    def test_simulator_run(self):
        """Simulator should run without errors."""
        config = RadarConfig()
        sim = RadarSimulator(config)
        
        result = sim.run(target_range_m=1000, target_rcs_m2=0.1)
        assert result is not None


class TestPipelineIntegration:
    """Full integration tests."""
    
    def test_furuno_spec_detection(self):
        """Should detect 3-inch corner reflector per Furuno specs."""
        # Furuno DRS4D-NXT specs:
        # - 9.41 GHz
        # - 25W TX power
        # - 22 dBi antenna gain
        # - 3-inch corner reflector RCS = 0.082 m²
        
        result = run_simple_simulation(
            target_range_m=2000,
            target_rcs_m2=0.082
        )
        
        # Should detect at 2km
        assert len(result.detected_ranges_m) >= 1
    
    def test_range_accuracy(self):
        """Detected range should match true range within resolution."""
        true_range = 1234
        result = run_simple_simulation(
            target_range_m=true_range,
            target_rcs_m2=1.0
        )
        
        if len(result.detected_ranges_m) > 0:
            detected = result.detected_ranges_m[0]
            # Should be within 2× range resolution (~3m)
            assert abs(detected - true_range) < 10
