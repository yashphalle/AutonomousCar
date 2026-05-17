"""
SceneState — the ONLY interface between perception and planning.
Perception writes it. Planner reads it. No carla imports.
All speeds in m/s, distances in metres, angles in radians.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum


class TrafficLightState(Enum):
    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"
    OFF = "off"
    UNKNOWN = "unknown"


@dataclass
class EgoPose:
    x: float
    y: float
    z: float
    yaw_rad: float      # radians
    speed_mps: float    # m/s


@dataclass
class WaypointInfo:
    x: float
    y: float
    yaw_rad: float              # road heading, radians
    speed_limit_mps: float      # m/s (converted from CARLA km/h)
    is_junction: bool
    distance_from_ego_m: float


@dataclass
class TrafficLightInfo:
    id: int
    state: TrafficLightState
    distance_m: float
    red_duration_s: float
    yellow_duration_s: float
    green_duration_s: float


@dataclass
class StopSignInfo:
    id: int
    distance_m: float
    location_x: float
    location_y: float


@dataclass
class DetectedObject:
    id: int
    object_class: str       # "vehicle" | "pedestrian"
    x: float
    y: float
    yaw_rad: float
    vx_mps: float
    vy_mps: float
    speed_mps: float        # magnitude
    distance_m: float


@dataclass
class OccupancyGrid:
    """Stage 3e BEV grid — deferred."""
    pass


@dataclass
class SceneState:
    timestamp_s: float
    ego_pose: EgoPose
    route_waypoints: list[WaypointInfo]    # next ~50 m, ~1 m spacing
    route_start_idx: int                   # absolute index into full route
    traffic_lights: list[TrafficLightInfo] = field(default_factory=list)
    stop_signs: list[StopSignInfo] = field(default_factory=list)
    dynamic_objects: list[DetectedObject] = field(default_factory=list)
    lead_vehicle_id: int | None = None
    lead_vehicle_distance_m: float = -1.0    # -1.0 means no lead vehicle detected
    lead_vehicle_speed_mps: float = -1.0     # -1.0 means no lead vehicle detected
    # Phase 4 — Pedestrian + Crosswalk
    pedestrian_in_path: bool = False           # ped in ±30° forward cone, within 20 m
    crosswalk_ahead: bool = False              # crosswalk on ego's route within 25 m
    crosswalk_distance_m: float = -1.0         # distance to nearest crosswalk ahead (-1 = none)
    # Phase 5 — Stop sign
    stop_sign_ahead: bool = False              # stop sign ahead within 30 m
    stop_sign_distance_m: float = -1.0         # distance to nearest stop sign ahead (-1 = none)
    local_map: OccupancyGrid | None = None
