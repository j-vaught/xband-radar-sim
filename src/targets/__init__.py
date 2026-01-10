"""Maritime target models for radar simulation."""
from .maritime import (
    BoatTarget,
    BuoyTarget,
    TowerTarget,
    compute_boat_rcs,
    compute_buoy_rcs,
    compute_tower_rcs,
)

__all__ = [
    'BoatTarget',
    'BuoyTarget',
    'TowerTarget',
    'compute_boat_rcs',
    'compute_buoy_rcs',
    'compute_tower_rcs',
]
