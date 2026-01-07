"""Tests for scene module."""
import pytest
import numpy as np
import tempfile
import os
from src.propagation.scene import (
    Scene,
    SceneConfig,
    TargetObject,
    create_ground_plane,
    create_corner_reflector,
    create_flat_plate,
    create_test_scene,
    save_stl,
    load_stl,
)


class TestGeometryPrimitives:
    """Tests for geometry creation."""
    
    def test_ground_plane(self):
        """Ground plane should be two triangles."""
        config = SceneConfig()
        vertices, faces = create_ground_plane(config)
        assert len(vertices) == 4  # 4 corners
        assert len(faces) == 2     # 2 triangles
    
    def test_corner_reflector_geometry(self):
        """Corner reflector should have geometry."""
        vertices, faces = create_corner_reflector(edge_length_m=0.0762)
        assert len(vertices) > 0
        assert len(faces) > 0
    
    def test_flat_plate(self):
        """Flat plate should be two triangles."""
        vertices, faces = create_flat_plate(width_m=1.0, height_m=0.5)
        assert len(vertices) == 4
        assert len(faces) == 2


class TestScene:
    """Tests for Scene class."""
    
    def test_empty_scene(self):
        """Empty scene should have no targets."""
        scene = Scene()
        assert len(scene.targets) == 0
    
    def test_add_target(self):
        """Should be able to add targets."""
        scene = Scene()
        vertices, faces = create_corner_reflector(0.1)
        target = TargetObject(
            name="corner",
            position=(100, 0, 0),
            vertices=vertices,
            faces=faces
        )
        scene.add_target(target)
        assert len(scene.targets) == 1
    
    def test_get_all_triangles(self):
        """Should return all triangles from all targets."""
        scene = Scene()
        vertices, faces = create_flat_plate(1.0, 1.0)
        target = TargetObject(
            name="plate",
            position=(0, 0, 0),
            vertices=vertices,
            faces=faces
        )
        scene.add_target(target)
        triangles = scene.get_all_triangles()
        assert len(triangles) >= 2


class TestTestScene:
    """Tests for test scene creation."""
    
    def test_create_test_scene(self):
        """Test scene should have targets."""
        scene = create_test_scene(target_range_m=1000)
        assert len(scene.targets) >= 1
    
    def test_test_scene_triangles(self):
        """Test scene should have triangles."""
        scene = create_test_scene(target_range_m=500)
        triangles = scene.get_all_triangles()
        assert len(triangles) > 0


class TestSTLIO:
    """Tests for STL file I/O."""
    
    def test_save_and_load_stl(self):
        """Should be able to save and load STL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.stl")
            
            # Create simple geometry
            vertices, faces = create_flat_plate(1.0, 1.0)
            
            # Save
            save_stl(filepath, vertices, faces)
            assert os.path.exists(filepath)
            
            # Load
            loaded_v, loaded_f = load_stl(filepath)
            assert len(loaded_f) == len(faces)
