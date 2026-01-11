"""
Real Terrain Data Loading

Load real-world elevation data from various sources:
- GeoTIFF files (local)
- SRTM data (download)
"""
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
import os

# Try to import optional dependencies
try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


@dataclass
class RealTerrainConfig:
    """Configuration for real terrain loading."""
    # Geographic bounds (lat/lon)
    center_lat: float = 47.6062  # Seattle default
    center_lon: float = -122.3321

    # Scene size in meters
    scene_radius_m: float = 2000.0
    resolution_m: float = 5.0

    # Radar position
    radar_height_m: float = 10.0

    # Water level (elevation below this is water)
    water_level_m: float = 0.0

    # Data source
    data_source: str = "file"  # "file", "srtm", "synthetic"
    file_path: Optional[str] = None


def meters_per_degree_lat() -> float:
    """Approximate meters per degree of latitude."""
    return 111320.0


def meters_per_degree_lon(lat: float) -> float:
    """Approximate meters per degree of longitude at given latitude."""
    return 111320.0 * np.cos(np.radians(lat))


def load_geotiff(file_path: str) -> Tuple[np.ndarray, dict]:
    """Load elevation data from a GeoTIFF file.

    Args:
        file_path: Path to GeoTIFF file

    Returns:
        Tuple of (elevation_array, metadata)
    """
    if not HAS_RASTERIO:
        raise ImportError("rasterio is required to load GeoTIFF files. Install with: pip install rasterio")

    with rasterio.open(file_path) as src:
        elevation = src.read(1)  # Read first band
        metadata = {
            'bounds': src.bounds,
            'crs': src.crs,
            'transform': src.transform,
            'width': src.width,
            'height': src.height,
            'resolution': src.res,
        }

    return elevation, metadata


def load_image_as_heightmap(file_path: str, max_elevation: float = 200.0) -> np.ndarray:
    """Load an image file as a heightmap (grayscale = elevation).

    Useful for PNG/JPG heightmaps where white = high, black = low.

    Args:
        file_path: Path to image file
        max_elevation: Maximum elevation in meters

    Returns:
        Elevation array
    """
    if not HAS_PIL:
        raise ImportError("PIL/Pillow is required to load image files. Install with: pip install Pillow")

    img = Image.open(file_path)

    # Convert to grayscale if needed
    if img.mode != 'L':
        img = img.convert('L')

    # Convert to numpy array and scale to elevation
    heightmap = np.array(img, dtype=np.float32)
    heightmap = heightmap / 255.0 * max_elevation

    return heightmap


def extract_region(
    heightmap: np.ndarray,
    metadata: dict,
    center_lat: float,
    center_lon: float,
    radius_m: float,
    output_resolution_m: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract a region from a larger heightmap centered on lat/lon.

    Args:
        heightmap: Full elevation array
        metadata: GeoTIFF metadata with bounds/transform
        center_lat: Center latitude
        center_lon: Center longitude
        radius_m: Radius of region to extract in meters
        output_resolution_m: Desired output resolution

    Returns:
        Tuple of (extracted_heightmap, x_coords, y_coords)
    """
    bounds = metadata['bounds']

    # Calculate pixel coordinates of center
    transform = metadata['transform']

    # Inverse transform: lat/lon to pixel
    col = int((center_lon - bounds.left) / (bounds.right - bounds.left) * metadata['width'])
    row = int((bounds.top - center_lat) / (bounds.top - bounds.bottom) * metadata['height'])

    # Calculate pixel radius
    m_per_deg_lat = meters_per_degree_lat()
    m_per_deg_lon = meters_per_degree_lon(center_lat)

    src_res_lat = (bounds.top - bounds.bottom) / metadata['height'] * m_per_deg_lat
    src_res_lon = (bounds.right - bounds.left) / metadata['width'] * m_per_deg_lon
    src_res = (src_res_lat + src_res_lon) / 2

    pixel_radius = int(radius_m / src_res)

    # Extract region
    row_min = max(0, row - pixel_radius)
    row_max = min(metadata['height'], row + pixel_radius)
    col_min = max(0, col - pixel_radius)
    col_max = min(metadata['width'], col + pixel_radius)

    region = heightmap[row_min:row_max, col_min:col_max]

    # Create coordinate arrays in meters (centered on radar at 0,0)
    ny, nx = region.shape
    x_coords = np.linspace(-radius_m, radius_m, nx)
    y_coords = np.linspace(-radius_m, radius_m, ny)

    # Resample if needed
    if output_resolution_m != src_res:
        from scipy.ndimage import zoom
        scale = src_res / output_resolution_m
        region = zoom(region, scale, order=1)

        nx_new, ny_new = region.shape[1], region.shape[0]
        x_coords = np.linspace(-radius_m, radius_m, nx_new)
        y_coords = np.linspace(-radius_m, radius_m, ny_new)

    return region, x_coords, y_coords


def create_coastal_mask(
    heightmap: np.ndarray,
    water_level: float = 0.0
) -> np.ndarray:
    """Create water mask from elevation data.

    Args:
        heightmap: Elevation array
        water_level: Elevation considered as water surface

    Returns:
        Boolean mask (True = water)
    """
    return heightmap <= water_level


def download_srtm_tile(lat: float, lon: float, cache_dir: str = "./terrain_cache") -> str:
    """Download SRTM tile for given coordinates.

    Note: Requires elevation package or manual download.

    Args:
        lat: Latitude
        lon: Longitude
        cache_dir: Directory to cache downloaded tiles

    Returns:
        Path to downloaded file
    """
    try:
        import elevation
    except ImportError:
        raise ImportError(
            "elevation package required for SRTM download. "
            "Install with: pip install elevation\n"
            "Or download GeoTIFF manually from: https://dwtkns.com/srtm30m/"
        )

    os.makedirs(cache_dir, exist_ok=True)

    # Calculate bounds (1 degree tile)
    west = np.floor(lon)
    east = west + 1
    south = np.floor(lat)
    north = south + 1

    output_path = os.path.join(cache_dir, f"srtm_{int(south)}_{int(west)}.tif")

    if not os.path.exists(output_path):
        elevation.clip(bounds=(west, south, east, north), output=output_path)

    return output_path


def load_real_terrain(config: RealTerrainConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load real terrain data based on configuration.

    Args:
        config: Terrain configuration

    Returns:
        Tuple of (heightmap, x_coords, y_coords, water_mask)
    """
    if config.data_source == "file" and config.file_path:
        # Load from GeoTIFF
        if config.file_path.lower().endswith(('.tif', '.tiff')):
            heightmap, metadata = load_geotiff(config.file_path)
            heightmap, x_coords, y_coords = extract_region(
                heightmap, metadata,
                config.center_lat, config.center_lon,
                config.scene_radius_m, config.resolution_m
            )
        else:
            # Assume image heightmap
            heightmap = load_image_as_heightmap(config.file_path)
            ny, nx = heightmap.shape
            x_coords = np.linspace(-config.scene_radius_m, config.scene_radius_m, nx)
            y_coords = np.linspace(-config.scene_radius_m, config.scene_radius_m, ny)

    elif config.data_source == "srtm":
        # Download SRTM data
        tile_path = download_srtm_tile(config.center_lat, config.center_lon)
        heightmap, metadata = load_geotiff(tile_path)
        heightmap, x_coords, y_coords = extract_region(
            heightmap, metadata,
            config.center_lat, config.center_lon,
            config.scene_radius_m, config.resolution_m
        )

    else:
        raise ValueError(f"Unknown data source: {config.data_source}")

    # Create water mask
    water_mask = create_coastal_mask(heightmap, config.water_level_m)

    # Ensure water areas are at water level
    heightmap = np.where(water_mask, config.water_level_m, heightmap)

    return heightmap, x_coords, y_coords, water_mask


# Example coastal locations with interesting terrain
EXAMPLE_LOCATIONS = {
    "seattle": {"lat": 47.6062, "lon": -122.3321, "desc": "Puget Sound, WA"},
    "san_francisco": {"lat": 37.8199, "lon": -122.4783, "desc": "Golden Gate, CA"},
    "rio": {"lat": -22.9519, "lon": -43.2105, "desc": "Rio de Janeiro, Brazil"},
    "hong_kong": {"lat": 22.2855, "lon": 114.1577, "desc": "Victoria Harbour, HK"},
    "sydney": {"lat": -33.8568, "lon": 151.2153, "desc": "Sydney Harbour, AU"},
    "bergen": {"lat": 60.3913, "lon": 5.3221, "desc": "Bergen Fjord, Norway"},
    "vancouver": {"lat": 49.2827, "lon": -123.1207, "desc": "Vancouver Harbour, BC"},
    "cape_town": {"lat": -33.9249, "lon": 18.4241, "desc": "Table Bay, SA"},
}


def print_available_locations():
    """Print available example locations."""
    print("Available example coastal locations:")
    for name, info in EXAMPLE_LOCATIONS.items():
        print(f"  {name}: {info['desc']} ({info['lat']:.4f}, {info['lon']:.4f})")
