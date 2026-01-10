"""PPI display processing for realistic radar visualization.

Implements:
- Beam spreading (azimuth convolution based on beamwidth)
- 2D Point Spread Function (PSF) convolution
- Polar-to-Cartesian scan conversion with interpolation
- Display normalization and scaling

References:
- NWS NEXRAD Training: Target subtension effects
- RadSimReal: PSF-based radar simulation
- CSU-CHILL: Scan conversion algorithms
"""
import numpy as np
from scipy.ndimage import gaussian_filter, convolve
from scipy.interpolate import RegularGridInterpolator
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class PPIProcessingConfig:
    """Configuration for PPI processing.

    Attributes:
        apply_beam_spreading: Enable azimuth spreading based on beamwidth
        apply_psf: Enable 2D point spread function convolution
        apply_scan_conversion: Enable polar-to-Cartesian interpolation
        beamwidth_deg: Antenna horizontal beamwidth in degrees
        range_resolution_m: Range resolution in meters
        output_size: Size of Cartesian output (pixels)
        dynamic_range_db: Dynamic range for display normalization
    """
    apply_beam_spreading: bool = True
    apply_psf: bool = True
    apply_scan_conversion: bool = True
    beamwidth_deg: float = 3.9  # Typical X-band radar
    range_resolution_m: float = 3.0  # After pulse compression
    output_size: int = 512
    dynamic_range_db: float = 50.0


def create_azimuth_spreading_kernel(beamwidth_deg: float,
                                     azimuth_spacing_deg: float) -> np.ndarray:
    """
    Create 1D kernel for azimuth spreading based on antenna beamwidth.

    Real radar targets appear spread across the full beamwidth.
    This kernel approximates the antenna pattern main lobe.

    Args:
        beamwidth_deg: Antenna 3dB beamwidth
        azimuth_spacing_deg: Spacing between azimuth samples

    Returns:
        1D Gaussian kernel for convolution in azimuth
    """
    # Convert beamwidth to sigma (FWHM = 2.355 * sigma)
    sigma_deg = beamwidth_deg / 2.355

    # Kernel size (3 sigma each side)
    kernel_size_deg = 6 * sigma_deg
    kernel_size_bins = int(np.ceil(kernel_size_deg / azimuth_spacing_deg))

    # Ensure odd size for symmetric kernel
    if kernel_size_bins % 2 == 0:
        kernel_size_bins += 1
    kernel_size_bins = max(kernel_size_bins, 3)

    # Create Gaussian kernel
    x = np.arange(kernel_size_bins) - kernel_size_bins // 2
    x_deg = x * azimuth_spacing_deg
    sigma = sigma_deg
    kernel = np.exp(-x_deg**2 / (2 * sigma**2))

    # Normalize
    kernel = kernel / kernel.sum()

    return kernel


def apply_beam_spreading(ppi: np.ndarray,
                          beamwidth_deg: float,
                          n_azimuths: int) -> np.ndarray:
    """
    Apply beam spreading to PPI data.

    Convolves the PPI in the azimuth dimension to simulate
    target spreading across the antenna beamwidth.

    Args:
        ppi: PPI data (n_azimuths, n_ranges)
        beamwidth_deg: Antenna beamwidth in degrees
        n_azimuths: Number of azimuth samples

    Returns:
        PPI with beam spreading applied
    """
    azimuth_spacing_deg = 360.0 / n_azimuths

    # Create spreading kernel
    kernel = create_azimuth_spreading_kernel(beamwidth_deg, azimuth_spacing_deg)

    # Apply convolution with wraparound (circular convolution for azimuth)
    # Pad for wraparound
    pad_size = len(kernel) // 2
    ppi_padded = np.concatenate([ppi[-pad_size:], ppi, ppi[:pad_size]], axis=0)

    # Convolve along azimuth axis
    ppi_spread = np.apply_along_axis(
        lambda x: np.convolve(x, kernel, mode='same'),
        axis=0,
        arr=ppi_padded
    )

    # Remove padding
    ppi_spread = ppi_spread[pad_size:-pad_size]

    return ppi_spread


def create_radar_psf(range_resolution_bins: float,
                     azimuth_resolution_bins: float,
                     use_sinc: bool = False) -> np.ndarray:
    """
    Create 2D radar Point Spread Function.

    The PSF represents how a point target appears in the radar image.
    - Range dimension: sinc-like (from matched filter)
    - Azimuth dimension: Gaussian (from antenna pattern)

    Args:
        range_resolution_bins: Range resolution in bins (samples)
        azimuth_resolution_bins: Azimuth resolution in bins
        use_sinc: If True, use sinc function for range; else use Gaussian

    Returns:
        2D PSF array (azimuth, range)
    """
    # PSF size (extend to capture sidelobes if using sinc)
    if use_sinc:
        range_size = int(range_resolution_bins * 10)  # Wider for sidelobes
        az_size = int(azimuth_resolution_bins * 6)
    else:
        range_size = int(range_resolution_bins * 6)
        az_size = int(azimuth_resolution_bins * 6)

    # Ensure odd sizes
    range_size = max(range_size, 3) | 1
    az_size = max(az_size, 3) | 1

    # Create coordinate grids
    r = np.arange(range_size) - range_size // 2
    a = np.arange(az_size) - az_size // 2

    R, A = np.meshgrid(r, a)

    # Range dimension (matched filter response)
    if use_sinc:
        # Sinc function with windowing to reduce sidelobes
        r_norm = R / (range_resolution_bins / 2)
        r_norm = np.where(np.abs(r_norm) < 0.01, 0.01, r_norm)
        range_response = np.sinc(r_norm)
        # Apply Hamming window to reduce sidelobes
        range_window = 0.54 + 0.46 * np.cos(np.pi * R / (range_size // 2))
        range_window = np.clip(range_window, 0, 1)
        range_response = range_response * range_window
    else:
        # Gaussian approximation
        range_sigma = range_resolution_bins / 2.355
        range_response = np.exp(-R**2 / (2 * range_sigma**2))

    # Azimuth dimension (antenna pattern - always Gaussian-like)
    az_sigma = azimuth_resolution_bins / 2.355
    az_response = np.exp(-A**2 / (2 * az_sigma**2))

    # Combine
    psf = range_response * az_response

    # Normalize to preserve energy
    psf = psf / psf.sum()

    return psf


def apply_psf_convolution(ppi: np.ndarray,
                           range_resolution_bins: float,
                           azimuth_resolution_bins: float,
                           use_sinc: bool = False) -> np.ndarray:
    """
    Apply 2D PSF convolution to PPI data.

    Args:
        ppi: PPI data (n_azimuths, n_ranges)
        range_resolution_bins: Range resolution in bins
        azimuth_resolution_bins: Azimuth resolution in bins
        use_sinc: Use sinc for range response

    Returns:
        PPI with PSF convolution applied
    """
    psf = create_radar_psf(range_resolution_bins, azimuth_resolution_bins, use_sinc)

    # Pad for wraparound in azimuth
    pad_az = psf.shape[0] // 2
    ppi_padded = np.concatenate([ppi[-pad_az:], ppi, ppi[:pad_az]], axis=0)

    # Convolve
    ppi_convolved = convolve(ppi_padded, psf, mode='constant')

    # Remove azimuth padding
    ppi_convolved = ppi_convolved[pad_az:-pad_az]

    return ppi_convolved


def polar_to_cartesian(ppi: np.ndarray,
                        azimuths_deg: np.ndarray,
                        ranges_m: np.ndarray,
                        output_size: int = 512,
                        interpolation: str = 'linear') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert polar PPI data to Cartesian coordinates with interpolation.

    Uses bilinear interpolation for smooth scan conversion.

    Args:
        ppi: PPI data (n_azimuths, n_ranges)
        azimuths_deg: Azimuth angles in degrees
        ranges_m: Range values in meters
        output_size: Output image size in pixels
        interpolation: 'linear' or 'nearest'

    Returns:
        cartesian: Cartesian image (output_size, output_size)
        x: X coordinates in meters
        y: Y coordinates in meters
    """
    # Extend azimuth for wraparound
    azimuths_rad = np.deg2rad(azimuths_deg)
    az_extended = np.concatenate([
        azimuths_rad - 2*np.pi,
        azimuths_rad,
        azimuths_rad + 2*np.pi
    ])
    ppi_extended = np.concatenate([ppi, ppi, ppi], axis=0)

    # Create interpolator
    interp = RegularGridInterpolator(
        (az_extended, ranges_m),
        ppi_extended,
        method=interpolation,
        bounds_error=False,
        fill_value=0.0
    )

    # Create Cartesian grid
    max_range = ranges_m[-1]
    x = np.linspace(-max_range, max_range, output_size)
    y = np.linspace(-max_range, max_range, output_size)
    X, Y = np.meshgrid(x, y)

    # Convert to polar coordinates
    R = np.sqrt(X**2 + Y**2)
    # Use atan2 with Y, X for standard math convention (0° = East, CCW positive)
    # Then adjust for radar convention (0° = North, CW positive)
    Theta = np.arctan2(X, Y)  # This gives 0° at North, CW positive

    # Clip range to valid values
    R = np.clip(R, ranges_m[0], ranges_m[-1])

    # Interpolate
    points = np.column_stack([Theta.ravel(), R.ravel()])
    cartesian = interp(points).reshape(output_size, output_size)

    return cartesian, x, y


def normalize_for_display(ppi: np.ndarray,
                           dynamic_range_db: float = 50.0,
                           log_scale: bool = True) -> np.ndarray:
    """
    Normalize PPI data for display.

    Args:
        ppi: PPI data (power, not dB)
        dynamic_range_db: Dynamic range to display
        log_scale: Apply logarithmic (dB) scaling

    Returns:
        Normalized data in range [0, 1]
    """
    if log_scale:
        # Convert to dB
        ppi_db = 10 * np.log10(ppi + 1e-30)

        # Find max and set floor
        vmax = np.max(ppi_db)
        vmin = vmax - dynamic_range_db

        # Normalize
        ppi_norm = (ppi_db - vmin) / (vmax - vmin)
    else:
        # Linear normalization
        vmax = np.max(ppi)
        vmin = vmax / (10 ** (dynamic_range_db / 10))
        ppi_norm = (ppi - vmin) / (vmax - vmin)

    return np.clip(ppi_norm, 0, 1)


def process_ppi(ppi: np.ndarray,
                 azimuths_deg: np.ndarray,
                 ranges_m: np.ndarray,
                 config: PPIProcessingConfig) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Complete PPI processing pipeline.

    Applies beam spreading, PSF convolution, and optional scan conversion.

    Args:
        ppi: Raw PPI data (n_azimuths, n_ranges)
        azimuths_deg: Azimuth angles
        ranges_m: Range values
        config: Processing configuration

    Returns:
        processed_ppi: Processed PPI (polar or Cartesian based on config)
        x: X coordinates if Cartesian, None otherwise
        y: Y coordinates if Cartesian, None otherwise
    """
    n_azimuths = len(azimuths_deg)
    azimuth_spacing_deg = 360.0 / n_azimuths

    # Calculate resolution in bins
    range_spacing_m = ranges_m[1] - ranges_m[0] if len(ranges_m) > 1 else 1.0
    range_resolution_bins = config.range_resolution_m / range_spacing_m
    azimuth_resolution_bins = config.beamwidth_deg / azimuth_spacing_deg

    processed = ppi.copy()

    # Apply beam spreading (azimuth convolution)
    if config.apply_beam_spreading:
        processed = apply_beam_spreading(processed, config.beamwidth_deg, n_azimuths)

    # Apply 2D PSF convolution
    if config.apply_psf:
        processed = apply_psf_convolution(processed,
                                           range_resolution_bins,
                                           azimuth_resolution_bins,
                                           use_sinc=False)

    # Convert to Cartesian if requested
    if config.apply_scan_conversion:
        cartesian, x, y = polar_to_cartesian(processed, azimuths_deg, ranges_m,
                                              config.output_size)
        return cartesian, x, y

    return processed, None, None


def quick_beam_spread(ppi: np.ndarray,
                       beamwidth_deg: float,
                       n_azimuths: int,
                       range_sigma_bins: float = 1.5) -> np.ndarray:
    """
    Quick beam spreading using scipy's gaussian_filter.

    Simpler alternative to full PSF convolution.

    Args:
        ppi: PPI data (n_azimuths, n_ranges)
        beamwidth_deg: Antenna beamwidth
        n_azimuths: Number of azimuths
        range_sigma_bins: Range spreading in bins

    Returns:
        PPI with spreading applied
    """
    # Calculate azimuth sigma in bins
    azimuth_spacing_deg = 360.0 / n_azimuths
    az_sigma_bins = (beamwidth_deg / 2.355) / azimuth_spacing_deg

    # Apply Gaussian filter with wraparound in azimuth
    return gaussian_filter(ppi, sigma=[az_sigma_bins, range_sigma_bins], mode=['wrap', 'nearest'])


def sinc_range_response(n_samples: int,
                         bandwidth_hz: float,
                         sample_rate_hz: float) -> np.ndarray:
    """
    Generate theoretical sinc range response from matched filter.

    Args:
        n_samples: Number of samples
        bandwidth_hz: Signal bandwidth
        sample_rate_hz: Sample rate

    Returns:
        Sinc response (power, not amplitude)
    """
    # Resolution in samples
    resolution_samples = sample_rate_hz / bandwidth_hz

    # Sample positions relative to center
    n = np.arange(n_samples) - n_samples // 2

    # Normalized position
    x = n / resolution_samples

    # Sinc function (handle zero)
    response = np.sinc(x)

    # Return power
    return response ** 2


def create_antenna_pattern_2d(theta_deg: np.ndarray,
                               phi_deg: np.ndarray,
                               beamwidth_h_deg: float,
                               beamwidth_v_deg: float) -> np.ndarray:
    """
    Create 2D antenna pattern (sinc approximation).

    Args:
        theta_deg: Elevation angles
        phi_deg: Azimuth angles
        beamwidth_h_deg: Horizontal beamwidth
        beamwidth_v_deg: Vertical beamwidth

    Returns:
        2D pattern (n_theta, n_phi)
    """
    THETA, PHI = np.meshgrid(theta_deg, phi_deg, indexing='ij')

    # Normalize by beamwidth
    u_h = 2.783 * PHI / beamwidth_h_deg
    u_v = 2.783 * THETA / beamwidth_v_deg

    # Sinc pattern (handle zeros)
    pattern_h = np.where(np.abs(u_h) < 1e-6, 1.0,
                         (np.sin(np.pi * u_h) / (np.pi * u_h)) ** 2)
    pattern_v = np.where(np.abs(u_v) < 1e-6, 1.0,
                         (np.sin(np.pi * u_v) / (np.pi * u_v)) ** 2)

    # Combined pattern
    pattern = pattern_h * pattern_v

    return np.clip(pattern, 1e-6, 1.0)
