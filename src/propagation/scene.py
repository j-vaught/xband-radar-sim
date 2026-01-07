"""Scene loading and management for ray tracing."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import numpy as np


@dataclass
class SceneConfig:
    """Configuration for ray tracing scene."""
    geometry_file: Optional[str] = None
    ground_plane: bool = True
    ground_height_m: float = 0.0
    ground_material: str = "concrete"
    
    # Scene bounds
    x_range_m: Tuple[float, float] = (-100, 100)
    y_range_m: Tuple[float, float] = (-100, 100)
    z_range_m: Tuple[float, float] = (0, 50)


@dataclass
class TargetObject:
    """Target object in scene."""
    name: str
    position: Tuple[float, float, float]  # (x, y, z) in meters
    vertices: np.ndarray = field(default_factory=lambda: np.array([]))
    faces: np.ndarray = field(default_factory=lambda: np.array([]))
    rotation_deg: Tuple[float, float, float] = (0, 0, 0)
    material: str = "metal"


@dataclass
class Scene:
    """Ray tracing scene containing geometry and materials."""
    targets: List[TargetObject] = field(default_factory=list)
    ground_vertices: Optional[np.ndarray] = None
    ground_faces: Optional[np.ndarray] = None
    config: SceneConfig = field(default_factory=SceneConfig)
    
    def add_target(self, target: TargetObject) -> None:
        """Add target to scene."""
        self.targets.append(target)
    
    def get_all_triangles(self) -> np.ndarray:
        """Get all triangles in scene as (N, 3, 3) array."""
        triangles = []
        
        # Add ground
        if self.ground_vertices is not None and self.ground_faces is not None:
            for face in self.ground_faces:
                tri = self.ground_vertices[face]
                triangles.append(tri)
        
        # Add targets
        for target in self.targets:
            if len(target.vertices) > 0 and len(target.faces) > 0:
                for face in target.faces:
                    tri = target.vertices[face] + np.array(target.position)
                    triangles.append(tri)
        
        if triangles:
            return np.array(triangles)
        return np.empty((0, 3, 3))


def create_ground_plane(
    config: SceneConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create ground plane geometry.
    
    Args:
        config: Scene configuration
        
    Returns:
        vertices: (4, 3) corner vertices
        faces: (2, 3) triangle face indices
    """
    x_min, x_max = config.x_range_m
    y_min, y_max = config.y_range_m
    z = config.ground_height_m
    
    vertices = np.array([
        [x_min, y_min, z],
        [x_max, y_min, z],
        [x_max, y_max, z],
        [x_min, y_max, z]
    ], dtype=np.float64)
    
    # Two triangles for plane
    faces = np.array([
        [0, 1, 2],
        [0, 2, 3]
    ], dtype=np.int32)
    
    return vertices, faces


def create_corner_reflector(
    edge_length_m: float = 0.0762
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create trihedral corner reflector geometry.
    
    Args:
        edge_length_m: Edge length (default 3 inches)
        
    Returns:
        vertices, faces
    """
    L = edge_length_m
    
    # Trihedral corner has 3 triangular faces meeting at origin
    vertices = np.array([
        [0, 0, 0],  # Origin (apex)
        [L, 0, 0],  # X-axis vertex
        [0, L, 0],  # Y-axis vertex
        [0, 0, L],  # Z-axis vertex
    ], dtype=np.float64)
    
    # Three faces
    faces = np.array([
        [0, 1, 2],  # XY plane
        [0, 2, 3],  # YZ plane
        [0, 3, 1],  # XZ plane
    ], dtype=np.int32)
    
    return vertices, faces


def create_flat_plate(
    width_m: float = 1.0,
    height_m: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Create flat rectangular plate geometry."""
    w, h = width_m / 2, height_m / 2
    
    vertices = np.array([
        [-w, -h, 0],
        [w, -h, 0],
        [w, h, 0],
        [-w, h, 0],
    ], dtype=np.float64)
    
    faces = np.array([
        [0, 1, 2],
        [0, 2, 3]
    ], dtype=np.int32)
    
    return vertices, faces


def create_sphere(
    radius_m: float = 0.1,
    n_segments: int = 16
) -> Tuple[np.ndarray, np.ndarray]:
    """Create sphere geometry using icosphere subdivision."""
    # Start with octahedron
    vertices = [
        [0, 0, radius_m],    # Top
        [radius_m, 0, 0],    # +X
        [0, radius_m, 0],    # +Y
        [-radius_m, 0, 0],   # -X
        [0, -radius_m, 0],   # -Y
        [0, 0, -radius_m],   # Bottom
    ]
    
    faces = [
        [0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1],  # Top pyramid
        [5, 2, 1], [5, 3, 2], [5, 4, 3], [5, 1, 4],  # Bottom pyramid
    ]
    
    # Subdivide for smoother sphere
    for _ in range(2):
        new_faces = []
        edge_midpoints = {}
        
        for face in faces:
            midpoints = []
            for i in range(3):
                edge = tuple(sorted([face[i], face[(i+1) % 3]]))
                if edge not in edge_midpoints:
                    v1 = np.array(vertices[edge[0]])
                    v2 = np.array(vertices[edge[1]])
                    mid = (v1 + v2) / 2
                    mid = mid / np.linalg.norm(mid) * radius_m  # Project to sphere
                    edge_midpoints[edge] = len(vertices)
                    vertices.append(mid.tolist())
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
        
        faces = new_faces
    
    return np.array(vertices, dtype=np.float64), np.array(faces, dtype=np.int32)


def create_cylinder(
    radius_m: float = 0.127,  # 5 inches = 0.127m
    height_m: float = 0.76,   # 2.5 ft = 0.76m
    n_segments: int = 16
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create cylinder geometry for buoy or similar object.
    
    Cylinder is vertical (Z-axis), centered at origin, extending from 0 to height.
    
    Args:
        radius_m: Cylinder radius
        height_m: Cylinder height
        n_segments: Number of segments around circumference
    """
    vertices = []
    faces = []
    
    # Bottom center (index 0)
    vertices.append([0, 0, 0])
    
    # Bottom ring (indices 1 to n_segments)
    for i in range(n_segments):
        angle = 2 * np.pi * i / n_segments
        vertices.append([radius_m * np.cos(angle), radius_m * np.sin(angle), 0])
    
    # Top center (index n_segments + 1)
    vertices.append([0, 0, height_m])
    
    # Top ring (indices n_segments + 2 to 2*n_segments + 1)
    for i in range(n_segments):
        angle = 2 * np.pi * i / n_segments
        vertices.append([radius_m * np.cos(angle), radius_m * np.sin(angle), height_m])
    
    # Bottom cap faces
    for i in range(n_segments):
        next_i = (i + 1) % n_segments
        faces.append([0, next_i + 1, i + 1])  # CCW for outward normal
    
    # Top cap faces
    top_center = n_segments + 1
    top_ring_start = n_segments + 2
    for i in range(n_segments):
        next_i = (i + 1) % n_segments
        faces.append([top_center, top_ring_start + i, top_ring_start + next_i])
    
    # Side faces (quads as two triangles)
    for i in range(n_segments):
        next_i = (i + 1) % n_segments
        bottom_curr = i + 1
        bottom_next = next_i + 1
        top_curr = top_ring_start + i
        top_next = top_ring_start + next_i
        
        faces.append([bottom_curr, bottom_next, top_next])
        faces.append([bottom_curr, top_next, top_curr])
    
    return np.array(vertices, dtype=np.float64), np.array(faces, dtype=np.int32)


def create_test_scene(
    target_range_m: float = 2000.0,
    target_type: str = "corner"
) -> Scene:
    """
    Create standard test scene.
    
    Args:
        target_range_m: Distance to target
        target_type: "corner", "plate", or "sphere"
        
    Returns:
        Configured Scene object
    """
    config = SceneConfig(
        ground_plane=True,
        x_range_m=(-100, target_range_m + 100),
        y_range_m=(-100, 100)
    )
    
    scene = Scene(config=config)
    
    # Add ground
    ground_v, ground_f = create_ground_plane(config)
    scene.ground_vertices = ground_v
    scene.ground_faces = ground_f
    
    # Add target
    if target_type == "corner":
        v, f = create_corner_reflector(0.0762)  # 3 inch
    elif target_type == "plate":
        v, f = create_flat_plate(1.0, 1.0)
    elif target_type == "sphere":
        v, f = create_sphere(0.1)
    else:
        v, f = create_corner_reflector(0.0762)
    
    target = TargetObject(
        name=f"{target_type}_target",
        position=(target_range_m, 0, 1.0),  # 1m above ground
        vertices=v,
        faces=f
    )
    scene.add_target(target)
    
    return scene


def load_stl(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load mesh from STL file.
    
    Args:
        filepath: Path to STL file
        
    Returns:
        vertices, faces
    """
    import struct
    
    with open(filepath, 'rb') as f:
        # Read header
        header = f.read(80)
        
        # Read number of triangles
        n_triangles = struct.unpack('<I', f.read(4))[0]
        
        vertices = []
        faces = []
        vertex_map = {}
        
        for i in range(n_triangles):
            # Normal (ignored, will recompute)
            f.read(12)
            
            # Three vertices
            face_indices = []
            for _ in range(3):
                v = struct.unpack('<fff', f.read(12))
                v_key = (round(v[0], 6), round(v[1], 6), round(v[2], 6))
                
                if v_key not in vertex_map:
                    vertex_map[v_key] = len(vertices)
                    vertices.append(v)
                
                face_indices.append(vertex_map[v_key])
            
            faces.append(face_indices)
            
            # Attribute byte count
            f.read(2)
    
    return np.array(vertices, dtype=np.float64), np.array(faces, dtype=np.int32)


def save_stl(
    filepath: str,
    vertices: np.ndarray,
    faces: np.ndarray
) -> None:
    """Save mesh to binary STL file."""
    import struct
    
    with open(filepath, 'wb') as f:
        # Header (80 bytes)
        f.write(b'\x00' * 80)
        
        # Number of triangles
        f.write(struct.pack('<I', len(faces)))
        
        for face in faces:
            v0 = vertices[face[0]]
            v1 = vertices[face[1]]
            v2 = vertices[face[2]]
            
            # Compute normal
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            norm = np.linalg.norm(normal)
            if norm > 0:
                normal = normal / norm
            
            # Write normal
            f.write(struct.pack('<fff', *normal))
            
            # Write vertices
            f.write(struct.pack('<fff', *v0))
            f.write(struct.pack('<fff', *v1))
            f.write(struct.pack('<fff', *v2))
            
            # Attribute byte count
            f.write(struct.pack('<H', 0))
