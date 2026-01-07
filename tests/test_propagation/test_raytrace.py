"""Tests for ray tracing module."""
import pytest
import numpy as np
from src.propagation.cpu_raytrace import (
    trace_rays_cpu,
    generate_rays_uniform,
    ray_triangle_intersect,
)


class TestRayTriangleIntersect:
    """Tests for ray-triangle intersection."""
    
    def test_ray_hits_triangle(self):
        """Ray should hit triangle directly in front."""
        # Triangle in xy plane at z=1
        v0 = np.array([-1.0, -1.0, 1.0])
        v1 = np.array([1.0, -1.0, 1.0])
        v2 = np.array([0.0, 1.0, 1.0])
        
        # Ray from origin pointing +z
        origin = np.array([0.0, 0.0, 0.0])
        direction = np.array([0.0, 0.0, 1.0])
        
        t = ray_triangle_intersect(origin, direction, v0, v1, v2)
        assert t > 0
        assert t == pytest.approx(1.0, rel=0.01)
    
    def test_ray_misses_triangle(self):
        """Ray should miss triangle that's off to the side."""
        v0 = np.array([10.0, 10.0, 1.0])
        v1 = np.array([12.0, 10.0, 1.0])
        v2 = np.array([11.0, 12.0, 1.0])
        
        origin = np.array([0.0, 0.0, 0.0])
        direction = np.array([0.0, 0.0, 1.0])
        
        t = ray_triangle_intersect(origin, direction, v0, v1, v2)
        assert t < 0  # Miss returns negative


class TestGenerateRays:
    """Tests for ray generation."""
    
    def test_ray_count(self):
        """Should generate requested number of rays."""
        origin = np.array([0.0, 0.0, 0.0])
        origins, directions = generate_rays_uniform(origin, n_rays=100)
        assert origins.shape == (100, 3)
        assert directions.shape == (100, 3)
    
    def test_directions_normalized(self):
        """All direction vectors should be approximately unit length."""
        origin = np.array([0.0, 0.0, 0.0])
        _, directions = generate_rays_uniform(origin, n_rays=50)
        norms = np.linalg.norm(directions, axis=1)
        assert np.allclose(norms, 1.0, atol=0.01)
    
    def test_origins_at_source(self):
        """All origins should be at specified source."""
        origin = np.array([1.0, 2.0, 3.0])
        origins, _ = generate_rays_uniform(origin, n_rays=10)
        assert np.allclose(origins, origin)


class TestTraceRays:
    """Tests for full ray tracing."""
    
    def test_trace_single_triangle(self):
        """Should trace rays against single triangle."""
        # Ground plane triangle
        triangles = np.array([[
            [-100.0, -100.0, 0.0],
            [100.0, -100.0, 0.0],
            [0.0, 100.0, 0.0]
        ]])
        
        # Rays from above pointing down
        origins = np.array([[0.0, 0.0, 10.0], [1.0, 1.0, 10.0]])
        directions = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]])
        
        bundle = trace_rays_cpu(origins, directions, triangles, max_range=100)
        assert bundle.n_rays == 2
    
    def test_no_hits_when_facing_away(self):
        """Should not hit triangle when facing away."""
        triangles = np.array([[
            [-10.0, -10.0, -100.0],
            [10.0, -10.0, -100.0],
            [0.0, 10.0, -100.0]
        ]])
        
        # Rays pointing up (away from triangle below)
        origins = np.array([[0.0, 0.0, 0.0]])
        directions = np.array([[0.0, 0.0, 1.0]])
        
        bundle = trace_rays_cpu(origins, directions, triangles, max_range=1000)
        assert np.sum(bundle.hit_mask) == 0
