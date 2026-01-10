"""Terrain generation and coastal scene modules."""
from .generator import (
    TerrainParams,
    generate_heightmap,
    heightmap_to_mesh,
    apply_shoreline_mask,
)
from .coastal_scene import (
    CoastalSceneConfig,
    CoastalScene,
    build_coastal_scene,
)

__all__ = [
    'TerrainParams',
    'generate_heightmap',
    'heightmap_to_mesh',
    'apply_shoreline_mask',
    'CoastalSceneConfig',
    'CoastalScene',
    'build_coastal_scene',
]
