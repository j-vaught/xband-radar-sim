"""Shared interface dataclasses for inter-module communication."""
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


@dataclass
class AntennaPattern:
    """Antenna radiation pattern (WP-1 output → WP-2 input).
    
    Attributes:
        frequency_hz: Operating frequency
        theta_deg: Elevation angles array (N_theta,)
        phi_deg: Azimuth angles array (N_phi,)
        gain_linear: Gain pattern (N_theta, N_phi) in linear scale
        phase_rad: Phase pattern (N_theta, N_phi) in radians
    """
    frequency_hz: float
    theta_deg: NDArray[np.float64]
    phi_deg: NDArray[np.float64]
    gain_linear: NDArray[np.float64]
    phase_rad: NDArray[np.float64]
    
    def gain_at(self, theta: float, phi: float) -> float:
        """Interpolate gain at given angles.
        
        Args:
            theta: Elevation angle in degrees
            phi: Azimuth angle in degrees
            
        Returns:
            Interpolated gain (linear)
        """
        from scipy.interpolate import RegularGridInterpolator
        interp = RegularGridInterpolator(
            (self.theta_deg, self.phi_deg),
            self.gain_linear,
            bounds_error=False,
            fill_value=0.0
        )
        return float(interp([[theta, phi]])[0])
    
    def gain_dbi(self) -> float:
        """Return peak gain in dBi."""
        return 10 * np.log10(np.max(self.gain_linear))


@dataclass
class RayBundle:
    """Ray tracing results (WP-2 output → WP-3 input).
    
    Attributes:
        n_rays: Number of rays
        origins: Ray origins (N, 3)
        directions: Ray direction unit vectors (N, 3)
        hit_points: Intersection points (N, 3)
        hit_normals: Surface normals at hits (N, 3)
        path_lengths_m: Path lengths in meters (N,)
        incident_powers_w: Incident power at hits (N,)
        hit_mask: Boolean mask for valid hits (N,)
    """
    n_rays: int
    origins: NDArray[np.float64]
    directions: NDArray[np.float64]
    hit_points: NDArray[np.float64]
    hit_normals: NDArray[np.float64]
    path_lengths_m: NDArray[np.float64]
    incident_powers_w: NDArray[np.float64]
    hit_mask: NDArray[np.bool_]


@dataclass
class ScatterResult:
    """Scattering computation results (WP-3 output → WP-4 input).
    
    Attributes:
        n_scatterers: Number of scattering points
        scatter_points: Scatter locations (N, 3)
        rcs_m2: RCS at each point in m² (N,)
        scattered_power_w: Scattered power toward RX (N,)
        path_to_rx_m: Distance to receiver (N,)
        phase_rad: Accumulated phase (N,)
    """
    n_scatterers: int
    scatter_points: NDArray[np.float64]
    rcs_m2: NDArray[np.float64]
    scattered_power_w: NDArray[np.float64]
    path_to_rx_m: NDArray[np.float64]
    phase_rad: NDArray[np.float64]


@dataclass
class RangeProfile:
    """Final radar output (WP-4 output).
    
    Attributes:
        range_bins_m: Range axis (N_bins,)
        amplitude_db: Power in dB (N_bins,)
        snr_db: Peak signal-to-noise ratio
        detected_ranges_m: Detected target ranges
    """
    range_bins_m: NDArray[np.float64]
    amplitude_db: NDArray[np.float64]
    snr_db: float
    detected_ranges_m: NDArray[np.float64]
