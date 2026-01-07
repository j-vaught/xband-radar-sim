"""CPU-based ray tracing using Numba JIT compilation."""
import numpy as np
from typing import Tuple

try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback decorators
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from interfaces import RayBundle


@jit(nopython=True, parallel=True)
def _ray_triangle_intersect_batch(
    origins: np.ndarray,
    directions: np.ndarray,
    triangles: np.ndarray,
    max_range: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba-accelerated batch ray-triangle intersection.
    
    Uses Möller-Trumbore algorithm.
    """
    EPSILON = 1e-8
    n_rays = origins.shape[0]
    n_triangles = triangles.shape[0]
    
    hit_points = np.zeros((n_rays, 3), dtype=np.float64)
    hit_normals = np.zeros((n_rays, 3), dtype=np.float64)
    path_lengths = np.full(n_rays, max_range, dtype=np.float64)
    hit_mask = np.zeros(n_rays, dtype=np.bool_)
    
    for i in prange(n_rays):
        origin = origins[i]
        direction = directions[i]
        closest_t = max_range
        closest_normal = np.zeros(3, dtype=np.float64)
        
        for j in range(n_triangles):
            v0 = triangles[j, 0]
            v1 = triangles[j, 1]
            v2 = triangles[j, 2]
            
            # Möller-Trumbore intersection
            edge1 = v1 - v0
            edge2 = v2 - v0
            
            h = np.cross(direction, edge2)
            a = np.dot(edge1, h)
            
            if -EPSILON < a < EPSILON:
                continue  # Parallel
            
            f = 1.0 / a
            s = origin - v0
            u = f * np.dot(s, h)
            
            if u < 0.0 or u > 1.0:
                continue
            
            q = np.cross(s, edge1)
            v = f * np.dot(direction, q)
            
            if v < 0.0 or u + v > 1.0:
                continue
            
            t = f * np.dot(edge2, q)
            
            if EPSILON < t < closest_t:
                closest_t = t
                # Compute normal
                n = np.cross(edge1, edge2)
                norm = np.sqrt(n[0]**2 + n[1]**2 + n[2]**2)
                if norm > EPSILON:
                    closest_normal = n / norm
                hit_mask[i] = True
        
        if hit_mask[i]:
            hit_points[i] = origin + closest_t * direction
            hit_normals[i] = closest_normal
            path_lengths[i] = closest_t
    
    return hit_points, hit_normals, path_lengths, hit_mask


def ray_triangle_intersect(
    origin: np.ndarray,
    direction: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray
) -> float:
    """
    Single ray-triangle intersection using Möller-Trumbore.
    
    Returns:
        t parameter (distance along ray), or -1 if no intersection
    """
    EPSILON = 1e-8
    
    edge1 = v1 - v0
    edge2 = v2 - v0
    
    h = np.cross(direction, edge2)
    a = np.dot(edge1, h)
    
    if -EPSILON < a < EPSILON:
        return -1.0  # Parallel
    
    f = 1.0 / a
    s = origin - v0
    u = f * np.dot(s, h)
    
    if u < 0.0 or u > 1.0:
        return -1.0
    
    q = np.cross(s, edge1)
    v = f * np.dot(direction, q)
    
    if v < 0.0 or u + v > 1.0:
        return -1.0
    
    t = f * np.dot(edge2, q)
    
    return t if t > EPSILON else -1.0


def trace_rays_cpu(
    origins: np.ndarray,
    directions: np.ndarray,
    triangles: np.ndarray,
    max_range: float = 5000.0
) -> RayBundle:
    """
    CPU ray tracing with optional Numba acceleration.
    
    Args:
        origins: (N, 3) ray origins
        directions: (N, 3) ray direction unit vectors
        triangles: (M, 3, 3) triangle vertices
        max_range: Maximum ray range
        
    Returns:
        RayBundle with intersection results
    """
    # Ensure correct dtypes
    origins = origins.astype(np.float64)
    directions = directions.astype(np.float64)
    triangles = triangles.astype(np.float64)
    
    if NUMBA_AVAILABLE:
        hit_points, hit_normals, path_lengths, hit_mask = _ray_triangle_intersect_batch(
            origins, directions, triangles, max_range
        )
    else:
        # Pure NumPy fallback (slower)
        hit_points, hit_normals, path_lengths, hit_mask = _trace_rays_numpy(
            origins, directions, triangles, max_range
        )
    
    # Compute placeholder powers
    incident_powers = np.ones(len(origins))
    
    return RayBundle(
        n_rays=len(origins),
        origins=origins,
        directions=directions,
        hit_points=hit_points,
        hit_normals=hit_normals,
        path_lengths_m=path_lengths,
        incident_powers_w=incident_powers,
        hit_mask=hit_mask
    )


def _trace_rays_numpy(
    origins: np.ndarray,
    directions: np.ndarray,
    triangles: np.ndarray,
    max_range: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pure NumPy ray tracing (no Numba)."""
    n_rays = len(origins)
    
    hit_points = np.zeros((n_rays, 3))
    hit_normals = np.zeros((n_rays, 3))
    path_lengths = np.full(n_rays, max_range)
    hit_mask = np.zeros(n_rays, dtype=bool)
    
    for i in range(n_rays):
        origin = origins[i]
        direction = directions[i]
        closest_t = max_range
        closest_normal = np.zeros(3)
        
        for tri in triangles:
            t = ray_triangle_intersect(origin, direction, tri[0], tri[1], tri[2])
            
            if 0 < t < closest_t:
                closest_t = t
                edge1 = tri[1] - tri[0]
                edge2 = tri[2] - tri[0]
                n = np.cross(edge1, edge2)
                closest_normal = n / (np.linalg.norm(n) + 1e-10)
                hit_mask[i] = True
        
        if hit_mask[i]:
            hit_points[i] = origin + closest_t * direction
            hit_normals[i] = closest_normal
            path_lengths[i] = closest_t
    
    return hit_points, hit_normals, path_lengths, hit_mask


def generate_rays_uniform(
    origin: np.ndarray,
    n_rays: int,
    fov_h_deg: float = 180.0,
    fov_v_deg: float = 90.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate uniformly distributed rays.
    
    Args:
        origin: (3,) ray origin
        n_rays: Number of rays
        fov_h_deg: Horizontal field of view
        fov_v_deg: Vertical field of view
        
    Returns:
        origins: (N, 3) ray origins
        directions: (N, 3) ray directions
    """
    phi = np.random.uniform(-np.deg2rad(fov_h_deg/2), np.deg2rad(fov_h_deg/2), n_rays)
    theta = np.random.uniform(-np.deg2rad(fov_v_deg/2), np.deg2rad(fov_v_deg/2), n_rays)
    
    directions = np.column_stack([
        np.cos(theta) * np.cos(phi),
        np.cos(theta) * np.sin(phi),
        np.sin(theta)
    ])
    
    origins = np.tile(origin, (n_rays, 1))
    
    return origins, directions
