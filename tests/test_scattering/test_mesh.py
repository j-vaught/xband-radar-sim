"""Tests for mesh processing module."""
import pytest
import numpy as np
from src.scattering.mesh import (
    compute_mesh_quality,
    compute_electrical_size,
    compute_face_centers,
    compute_face_normals,
    compute_face_areas,
    subdivide_mesh,
    mesh_stats,
    MeshConfig,
)


class TestMeshQuality:
    """Tests for mesh quality computation."""
    
    def test_equilateral_triangle_quality(self):
        """Equilateral triangle should have quality ≈ 1."""
        # Equilateral triangle
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, np.sqrt(3)/2, 0.0]
        ])
        faces = np.array([[0, 1, 2]])
        
        quality = compute_mesh_quality(vertices, faces)
        assert quality[0] == pytest.approx(1.0, abs=0.01)
    
    def test_degenerate_triangle_quality(self):
        """Very thin triangle should have low quality."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.01, 0.0]  # Very thin
        ])
        faces = np.array([[0, 1, 2]])
        
        quality = compute_mesh_quality(vertices, faces)
        assert quality[0] < 0.5


class TestElectricalSize:
    """Tests for electrical size computation."""
    
    def test_electrical_size(self):
        """Should compute L/λ correctly."""
        # Object 1m wide
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0]
        ])
        wavelength = 0.1  # 100mm
        
        L_over_lambda = compute_electrical_size(vertices, wavelength)
        assert L_over_lambda == pytest.approx(10.0, rel=0.1)  # 1m / 0.1m = 10


class TestFaceProperties:
    """Tests for face center, normal, area computations."""
    
    def test_face_centers(self):
        """Face center should be at centroid."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, 3.0, 0.0]
        ])
        faces = np.array([[0, 1, 2]])
        
        centers = compute_face_centers(vertices, faces)
        assert np.allclose(centers[0], [1, 1, 0])
    
    def test_face_normals_unit(self):
        """Face normals should be unit vectors."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ])
        faces = np.array([[0, 1, 2]])
        
        normals = compute_face_normals(vertices, faces)
        assert np.linalg.norm(normals[0]) == pytest.approx(1.0)
    
    def test_face_areas(self):
        """Face area computation should be correct."""
        # Right triangle with legs 1, 1
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ])
        faces = np.array([[0, 1, 2]])
        
        areas = compute_face_areas(vertices, faces)
        assert areas[0] == pytest.approx(0.5)


class TestMeshSubdivision:
    """Tests for mesh subdivision."""
    
    def test_subdivision_increases_faces(self):
        """Subdivision should produce 4x faces."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0]
        ])
        faces = np.array([[0, 1, 2]])
        
        new_v, new_f = subdivide_mesh(vertices, faces)
        assert len(new_f) == 4 * len(faces)


class TestMeshStats:
    """Tests for mesh statistics."""
    
    def test_mesh_stats(self):
        """Should return complete statistics."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0]
        ])
        faces = np.array([[0, 1, 2]])
        
        stats = mesh_stats(vertices, faces)
        assert 'n_vertices' in stats
        assert 'n_faces' in stats
        assert 'total_area' in stats
        assert 'min_quality' in stats
        assert stats['n_vertices'] == 3
        assert stats['n_faces'] == 1
