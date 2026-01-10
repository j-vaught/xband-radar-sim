"""
Magnetron Radar Simulation Pipeline

End-to-end simulation for magnetron radar in coastal/land scenarios.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import time

from config_magnetron import MagnetronConfig
from terrain.coastal_scene import CoastalScene, TargetInfo
from propagation.terrain_masking import (
    compute_visibility_mask,
    compute_visibility_along_azimuth,
    check_target_visible,
    compute_range_map,
    compute_azimuth_map,
    compute_grazing_angle,
)
from clutter.models import (
    LandClutterParams,
    SeaClutterParams,
    generate_clutter_map,
    add_clutter_variation,
)
from targets.maritime import (
    compute_target_return_power,
    aspect_angle_from_positions,
)
from signal.ppi_processing import quick_beam_spread, polar_to_cartesian, normalize_for_display

C = 299792458.0


@dataclass
class MagnetronSimulationResult:
    """Complete magnetron simulation output."""
    config: MagnetronConfig
    scene: CoastalScene

    # Visibility and masking
    visibility_mask: np.ndarray       # [ny, nx] boolean
    grazing_angles: np.ndarray        # [ny, nx] radians

    # Clutter maps
    sigma0_map: np.ndarray           # [ny, nx] normalized RCS
    clutter_power_map: np.ndarray    # [ny, nx] received power

    # PPI data
    ppi_polar: np.ndarray            # [n_azimuths, n_range_bins]
    ppi_cartesian: np.ndarray        # [size, size]

    # Range/azimuth axes
    azimuths_deg: np.ndarray
    range_bins_m: np.ndarray

    # Detected targets
    detected_targets: List[dict] = field(default_factory=list)

    # Timing
    computation_time_s: float = 0.0


def compute_cell_area(
    range_m: float,
    range_resolution_m: float,
    azimuth_beamwidth_rad: float
) -> float:
    """Compute area of a resolution cell.

    Area = range_resolution * range * azimuth_beamwidth

    Args:
        range_m: Range to cell
        range_resolution_m: Range resolution
        azimuth_beamwidth_rad: Azimuth beamwidth in radians

    Returns:
        Cell area in m²
    """
    cross_range = range_m * azimuth_beamwidth_rad
    return range_resolution_m * cross_range


def run_magnetron_simulation(
    config: MagnetronConfig,
    scene: CoastalScene,
    n_azimuths: int = 360,
    n_range_bins: int = 1024,
    land_params: LandClutterParams = None,
    sea_params: SeaClutterParams = None,
    verbose: bool = True
) -> MagnetronSimulationResult:
    """Run complete magnetron radar simulation.

    Steps:
    1. Compute visibility/masking for scene
    2. Generate clutter maps
    3. For each azimuth, compute returns
    4. Generate PPI display

    Args:
        config: Magnetron radar configuration
        scene: Coastal scene with terrain and targets
        n_azimuths: Number of azimuth samples
        land_params: Land clutter parameters
        sea_params: Sea clutter parameters
        verbose: Print progress

    Returns:
        MagnetronSimulationResult with all outputs
    """
    start_time = time.time()

    if land_params is None:
        land_params = LandClutterParams()
    if sea_params is None:
        sea_params = SeaClutterParams()

    if verbose:
        print(f"Magnetron Radar Simulation")
        print(f"  Range: {config.max_range_m:.0f}m")
        print(f"  Resolution: {config.range_resolution_m:.1f}m")
        print(f"  Beamwidth: {config.horizontal_beamwidth_deg:.1f}°")

    # Setup arrays
    azimuths_deg = np.linspace(0, 360, n_azimuths, endpoint=False)
    range_bins_m = np.linspace(0, config.max_range_m, n_range_bins)

    radar_pos = scene.radar_position

    # Step 1: Compute visibility mask
    if verbose:
        print("  Computing visibility mask...")

    visibility_mask = compute_visibility_mask(
        radar_pos,
        scene.heightmap,
        scene.x_coords,
        scene.y_coords,
        n_steps=100
    )

    # Step 2: Compute grazing angles
    grazing_angles = compute_grazing_angle(
        radar_pos,
        scene.heightmap,
        scene.x_coords,
        scene.y_coords
    )

    # Step 3: Generate clutter map
    if verbose:
        print("  Generating clutter map...")

    sigma0_map = generate_clutter_map(
        scene.heightmap,
        scene.x_coords,
        scene.y_coords,
        radar_pos,
        scene.water_mask,
        visibility_mask,
        land_params,
        sea_params,
        config.center_frequency_hz
    )

    # Add variation for realistic texture
    sigma0_map = add_clutter_variation(sigma0_map, variation_db=2.0, seed=42)

    # Step 4: Convert sigma0 to received power
    range_map = compute_range_map(radar_pos, scene.x_coords, scene.y_coords)

    # Cell area depends on range
    cell_area_map = np.zeros_like(range_map)
    for j, r in enumerate(range_map[:, 0]):
        cell_area_map[j, :] = compute_cell_area(
            max(r, 10),
            config.range_resolution_m,
            config.horizontal_beamwidth_rad
        )

    # Clutter RCS = sigma0 * area
    clutter_rcs_map = sigma0_map * cell_area_map

    # Received power from clutter (radar equation)
    clutter_power_map = np.zeros_like(clutter_rcs_map)
    valid = range_map > 10
    clutter_power_map[valid] = compute_target_return_power(
        clutter_rcs_map[valid],
        range_map[valid],
        config.peak_power_w,
        config.antenna_gain_linear,
        config.wavelength_m
    )

    # Step 5: Generate PPI
    if verbose:
        print("  Generating PPI...")

    ppi_polar = np.zeros((n_azimuths, n_range_bins))

    # Compute azimuth map for scene
    azimuth_map = compute_azimuth_map(radar_pos, scene.x_coords, scene.y_coords)

    # For each azimuth, collect returns
    for i, az_deg in enumerate(azimuths_deg):
        az_rad = np.radians(az_deg)

        # Visibility along this azimuth
        vis_along_az = compute_visibility_along_azimuth(
            radar_pos,
            az_rad,
            range_bins_m,
            scene.heightmap,
            scene.x_coords,
            scene.y_coords,
            n_steps=50
        )

        # Find cells in this azimuth beam
        az_diff = np.abs(azimuth_map - az_rad)
        az_diff = np.minimum(az_diff, 2*np.pi - az_diff)
        in_beam = az_diff < config.horizontal_beamwidth_rad / 2

        # Sum clutter power from cells in beam
        for j, r in enumerate(range_bins_m):
            if r < config.blind_range_m:
                continue
            if not vis_along_az[j]:
                continue

            # Find cells at this range
            range_diff = np.abs(range_map - r)
            in_range = range_diff < config.range_resolution_m / 2

            # Cells in beam and at this range
            cells = in_beam & in_range

            if np.any(cells):
                ppi_polar[i, j] = np.sum(clutter_power_map[cells])

    # Step 6: Add target returns
    if verbose:
        print(f"  Adding {len(scene.targets)} targets...")

    detected_targets = []

    for target in scene.targets:
        # Check if target is visible
        if not check_target_visible(
            radar_pos,
            target.position,
            scene.heightmap,
            scene.x_coords,
            scene.y_coords
        ):
            continue

        # Compute target range and azimuth
        dx = target.position[0] - radar_pos[0]
        dy = target.position[1] - radar_pos[1]
        target_range = np.sqrt(dx**2 + dy**2)
        target_az = np.degrees(np.arctan2(dx, dy)) % 360

        if target_range > config.max_range_m or target_range < config.blind_range_m:
            continue

        # Get target RCS
        rcs = target.rcs_m2

        # Compute received power
        power = compute_target_return_power(
            rcs,
            target_range,
            config.peak_power_w,
            config.antenna_gain_linear,
            config.wavelength_m
        )

        # Add to PPI
        az_idx = int(target_az / 360 * n_azimuths) % n_azimuths
        range_idx = int(target_range / config.max_range_m * n_range_bins)

        if 0 <= range_idx < n_range_bins:
            ppi_polar[az_idx, range_idx] += power

        detected_targets.append({
            'name': target.name,
            'type': target.target_type,
            'range_m': target_range,
            'azimuth_deg': target_az,
            'rcs_m2': rcs,
            'power': power
        })

    # Step 7: Apply STC (Sensitivity Time Control) - range compensation
    # Radar equation has R^4 dependence, so close targets are way brighter
    # Multiply by R^2 to flatten the display (partial compensation)
    if verbose:
        print("  Applying STC (range compensation)...")

    stc_gain = np.zeros(n_range_bins)
    ref_range = 500.0  # Reference range for normalization
    for j, r in enumerate(range_bins_m):
        if r > config.blind_range_m:
            # R^3 compensation - aggressive to flatten the display
            # Full R^4 would over-compensate and boost far targets too much
            stc_gain[j] = (r / ref_range) ** 3
        else:
            stc_gain[j] = 0.0

    # Apply STC to each azimuth
    ppi_stc = ppi_polar * stc_gain[np.newaxis, :]

    # Step 8: Apply beam spreading
    if verbose:
        print("  Applying beam spreading...")

    ppi_spread = quick_beam_spread(
        ppi_stc,
        config.horizontal_beamwidth_deg,
        n_azimuths,
        range_sigma_bins=0.3  # Minimal range spreading for magnetron
    )

    # Step 9: Convert to Cartesian
    ppi_cartesian, x_coords, y_coords = polar_to_cartesian(
        ppi_spread,
        azimuths_deg,
        range_bins_m,
        output_size=500
    )

    elapsed = time.time() - start_time

    if verbose:
        print(f"  Complete! ({elapsed:.1f}s)")
        print(f"  Detected {len(detected_targets)} targets")

    return MagnetronSimulationResult(
        config=config,
        scene=scene,
        visibility_mask=visibility_mask,
        grazing_angles=grazing_angles,
        sigma0_map=sigma0_map,
        clutter_power_map=clutter_power_map,
        ppi_polar=ppi_spread,
        ppi_cartesian=ppi_cartesian,
        azimuths_deg=azimuths_deg,
        range_bins_m=range_bins_m,
        detected_targets=detected_targets,
        computation_time_s=elapsed
    )


def run_quick_simulation(
    max_range_m: float = 2000.0,
    shoreline_distance_m: float = 600.0,
    n_boats: int = 5,
    n_buoys: int = 8,
    seed: int = 42
) -> MagnetronSimulationResult:
    """Quick simulation with default parameters.

    Args:
        max_range_m: Maximum radar range
        shoreline_distance_m: Distance to shoreline
        n_boats: Number of boats
        n_buoys: Number of buoys
        seed: Random seed

    Returns:
        Simulation result
    """
    from terrain.coastal_scene import create_simple_coastal_scene

    # Create scene
    scene = create_simple_coastal_scene(
        max_range_m=max_range_m,
        shoreline_distance_m=shoreline_distance_m,
        n_boats=n_boats,
        n_buoys=n_buoys,
        seed=seed
    )

    # Create radar config
    config = MagnetronConfig(
        max_range_m=max_range_m,
        pulse_width_s=0.5e-6,  # 75m resolution
        antenna_height_m=10.0
    )

    # Run simulation
    return run_magnetron_simulation(config, scene)
