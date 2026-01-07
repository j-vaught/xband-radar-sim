"""Mesh processing utilities for RCS computation."""
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import numpy as np


@dataclass
class MeshConfig:
    """Configuration for mesh refinement."""
    max_edge_length_m: float = 0.003  # λ/10 at 9.41 GHz
    min_edge_length_m: float = 0.001
    quality_threshold: float = 0.3    # Min triangle quality


def compute_mesh_quality(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Compute quality metric for each triangle.
    
    Quality = 4 * sqrt(3) * area / (a² + b² + c²)
    Range: 0 (degenerate) to 1 (equilateral)
    
    Args:
        vertices: (N, 3) vertex positions
        faces: (M, 3) face indices
        
    Returns:
        (M,) quality values
    """
    qualities = np.zeros(len(faces))
    sqrt3 = np.sqrt(3)
    
    for i, face in enumerate(faces):
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        
        # Edge lengths
        a = np.linalg.norm(v1 - v0)
        b = np.linalg.norm(v2 - v1)
        c = np.linalg.norm(v0 - v2)
        
        # Area via cross product
        area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
        
        # Quality metric
        denom = a*a + b*b + c*c
        if denom > 0:
            qualities[i] = 4 * sqrt3 * area / denom
    
    return qualities


def compute_electrical_size(
    vertices: np.ndarray,
    wavelength_m: float
) -> float:
    """
    Compute electrical size of object (max dimension / wavelength).
    
    Args:
        vertices: Vertex positions
        wavelength_m: Operating wavelength
        
    Returns:
        Electrical size L/λ
    """
    max_coords = np.max(vertices, axis=0)
    min_coords = np.min(vertices, axis=0)
    max_dim = np.max(max_coords - min_coords)
    return max_dim / wavelength_m


def compute_face_centers(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute centroid of each face."""
    centers = np.zeros((len(faces), 3))
    for i, face in enumerate(faces):
        centers[i] = (vertices[face[0]] + vertices[face[1]] + vertices[face[2]]) / 3
    return centers


def compute_face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute unit normal of each face."""
    normals = np.zeros((len(faces), 3))
    
    for i, face in enumerate(faces):
        v0 = vertices[face[0]]
        v1 = vertices[face[1]]
        v2 = vertices[face[2]]
        
        edge1 = v1 - v0
        edge2 = v2 - v0
        n = np.cross(edge1, edge2)
        norm = np.linalg.norm(n)
        
        if norm > 1e-12:
            normals[i] = n / norm
    
    return normals


def compute_face_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute area of each face."""
    areas = np.zeros(len(faces))
    
    for i, face in enumerate(faces):
        v0 = vertices[face[0]]
        v1 = vertices[face[1]]
        v2 = vertices[face[2]]
        
        edge1 = v1 - v0
        edge2 = v2 - v0
        areas[i] = 0.5 * np.linalg.norm(np.cross(edge1, edge2))
    
    return areas


def subdivide_mesh(
    vertices: np.ndarray,
    faces: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Subdivide mesh by splitting each triangle into 4.
    
    Args:
        vertices: (N, 3) vertex array
        faces: (M, 3) face array
        
    Returns:
        new_vertices, new_faces
    """
    vertex_list = vertices.tolist()
    new_faces = []
    edge_midpoints = {}
    
    for face in faces:
        midpoints = []
        for i in range(3):
            edge = tuple(sorted([face[i], face[(i+1) % 3]]))
            
            if edge not in edge_midpoints:
                v1 = np.array(vertex_list[edge[0]])
                v2 = np.array(vertex_list[edge[1]])
                mid = (v1 + v2) / 2
                edge_midpoints[edge] = len(vertex_list)
                vertex_list.append(mid.tolist())
            
            midpoints.append(edge_midpoints[edge])
        
        # Create 4 new triangles
        a, b, c = face
        ab, bc, ca = midpoints
        new_faces.extend([
            [a, ab, ca],
            [ab, b, bc],
            [ca, bc, c],
            [ab, bc, ca]
        ])
    
    return np.array(vertex_list), np.array(new_faces)


def refine_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    config: MeshConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Refine mesh to meet edge length requirements.
    
    Args:
        vertices: Input vertices
        faces: Input faces
        config: Mesh configuration
        
    Returns:
        Refined vertices, faces
    """
    max_iterations = 5
    
    for _ in range(max_iterations):
        # Check max edge length
        max_edge = 0.0
        for face in faces:
            for i in range(3):
                v1 = vertices[face[i]]
                v2 = vertices[face[(i+1) % 3]]
                edge_len = np.linalg.norm(v2 - v1)
                max_edge = max(max_edge, edge_len)
        
        if max_edge <= config.max_edge_length_m:
            break
        
        # Subdivide
        vertices, faces = subdivide_mesh(vertices, faces)
    
    return vertices, faces


def prepare_mesh_for_rcs(
    vertices: np.ndarray,
    faces: np.ndarray,
    wavelength_m: float
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Full mesh preparation pipeline.
    
    Args:
        vertices: Input vertices
        faces: Input faces
        wavelength_m: Operating wavelength
        
    Returns:
        vertices, faces, metadata dict
    """
    # Target edge length = λ/10
    config = MeshConfig(max_edge_length_m=wavelength_m / 10)
    
    # Refine
    vertices, faces = refine_mesh(vertices, faces, config)
    
    # Compute quality
    quality = compute_mesh_quality(vertices, faces)
    
    metadata = {
        'n_vertices': len(vertices),
        'n_faces': len(faces),
        'min_quality': float(np.min(quality)),
        'mean_quality': float(np.mean(quality)),
        'electrical_size': compute_electrical_size(vertices, wavelength_m)
    }
    
    return vertices, faces, metadata


def mesh_stats(vertices: np.ndarray, faces: np.ndarray) -> dict:
    """
    Compute mesh statistics.
    
    Returns:
        Dictionary with mesh properties
    """
    areas = compute_face_areas(vertices, faces)
    quality = compute_mesh_quality(vertices, faces)
    
    # Compute edge lengths
    edge_lengths = []
    for face in faces:
        for i in range(3):
            v1 = vertices[face[i]]
            v2 = vertices[face[(i+1) % 3]]
            edge_lengths.append(np.linalg.norm(v2 - v1))
    edge_lengths = np.array(edge_lengths)
    
    # Bounding box
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    
    return {
        'n_vertices': len(vertices),
        'n_faces': len(faces),
        'total_area': float(np.sum(areas)),
        'min_area': float(np.min(areas)),
        'max_area': float(np.max(areas)),
        'min_edge': float(np.min(edge_lengths)),
        'max_edge': float(np.max(edge_lengths)),
        'mean_edge': float(np.mean(edge_lengths)),
        'min_quality': float(np.min(quality)),
        'mean_quality': float(np.mean(quality)),
        'bbox_min': min_coords.tolist(),
        'bbox_max': max_coords.tolist()
    }
