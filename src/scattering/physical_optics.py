"""Physical Optics RCS computation (fallback)."""
import numpy as np
from typing import Tuple


def compute_po_rcs(
    vertices: np.ndarray,
    faces: np.ndarray,
    k: float,
    theta_inc: float = 0.0,
    phi_inc: float = 0.0
) -> float:
    """
    Physical Optics RCS approximation.
    
    Integrates over illuminated facets only.
    Valid for electrically large targets (ka > 5).
    
    Args:
        vertices: (N, 3) mesh vertices in meters
        faces: (M, 3) face vertex indices
        k: Wavenumber (2π/λ)
        theta_inc: Incidence elevation angle (radians)
        phi_inc: Incidence azimuth angle (radians)
        
    Returns:
        Monostatic RCS in m²
    """
    # Incident wave direction (unit vector pointing toward source)
    k_hat = np.array([
        np.sin(theta_inc) * np.cos(phi_inc),
        np.sin(theta_inc) * np.sin(phi_inc),
        np.cos(theta_inc)
    ])
    
    # Sum scattered field contributions
    E_scat = 0.0 + 0.0j
    
    for face in faces:
        v0 = vertices[face[0]]
        v1 = vertices[face[1]]
        v2 = vertices[face[2]]
        
        # Face center and normal
        center = (v0 + v1 + v2) / 3
        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = np.cross(edge1, edge2)
        area = 0.5 * np.linalg.norm(normal)
        
        if area < 1e-12:
            continue
            
        normal = normal / (2 * area)  # Unit normal
        
        # Check if illuminated (normal facing incident wave)
        cos_inc = -np.dot(normal, k_hat)
        if cos_inc <= 0:
            continue  # Shadowed facet
        
        # PO contribution (monostatic)
        # E_scat ∝ ∫∫ n̂ exp(2jk·r) dA
        phase = 2 * k * np.dot(k_hat, center)
        E_scat += cos_inc * area * np.exp(1j * phase)
    
    # RCS = (4π/λ²) |E_scat|²
    wavelength = 2 * np.pi / k
    rcs = (4 * np.pi / wavelength ** 2) * np.abs(E_scat) ** 2
    
    return rcs


def compute_po_bistatic(
    vertices: np.ndarray,
    faces: np.ndarray,
    k: float,
    theta_inc: float,
    phi_inc: float,
    theta_scat: float,
    phi_scat: float
) -> float:
    """
    Bistatic Physical Optics RCS.
    
    Args:
        vertices, faces: Mesh geometry
        k: Wavenumber
        theta_inc, phi_inc: Incident wave direction (radians)
        theta_scat, phi_scat: Scatter direction (radians)
        
    Returns:
        Bistatic RCS in m²
    """
    k_inc = np.array([
        np.sin(theta_inc) * np.cos(phi_inc),
        np.sin(theta_inc) * np.sin(phi_inc),
        np.cos(theta_inc)
    ])
    
    k_scat = np.array([
        np.sin(theta_scat) * np.cos(phi_scat),
        np.sin(theta_scat) * np.sin(phi_scat),
        np.cos(theta_scat)
    ])
    
    E_scat = 0.0 + 0.0j
    
    for face in faces:
        v0 = vertices[face[0]]
        v1 = vertices[face[1]]
        v2 = vertices[face[2]]
        
        center = (v0 + v1 + v2) / 3
        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = np.cross(edge1, edge2)
        area = 0.5 * np.linalg.norm(normal)
        
        if area < 1e-12:
            continue
            
        normal = normal / (2 * area)
        
        cos_inc = -np.dot(normal, k_inc)
        if cos_inc <= 0:
            continue
        
        # Bistatic phase: exp(jk(k_inc + k_scat)·r)
        phase = k * np.dot(k_inc + k_scat, center)
        E_scat += cos_inc * area * np.exp(1j * phase)
    
    wavelength = 2 * np.pi / k
    return (4 * np.pi / wavelength ** 2) * np.abs(E_scat) ** 2


def compute_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Compute face normals for mesh.
    
    Args:
        vertices: (N, 3) vertex positions
        faces: (M, 3) face vertex indices
        
    Returns:
        normals: (M, 3) unit normal vectors
    """
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
    """
    Compute face areas for mesh.
    
    Args:
        vertices: (N, 3) vertex positions
        faces: (M, 3) face vertex indices
        
    Returns:
        areas: (M,) face areas
    """
    areas = np.zeros(len(faces))
    
    for i, face in enumerate(faces):
        v0 = vertices[face[0]]
        v1 = vertices[face[1]]
        v2 = vertices[face[2]]
        
        edge1 = v1 - v0
        edge2 = v2 - v0
        areas[i] = 0.5 * np.linalg.norm(np.cross(edge1, edge2))
    
    return areas
