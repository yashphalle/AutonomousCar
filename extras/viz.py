"""
Rerun-based visualizer for the CARLA autonomous vehicle stack.

Runs standalone alongside a live main.py session:
  Terminal 1: ./CarlaUE4.sh
  Terminal 2: python3.10 main.py
  Terminal 3: python3.10 extras/viz.py [--host localhost] [--port 2000] [--save out.rrd]

"""

from __future__ import annotations

import argparse
import math
import os
import queue
import sys
import time
from typing import List, Optional, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils import carla_bootstrap  # noqa: F401

import carla
import numpy as np
import rerun as rr

# CARLA traffic light state → our enum (mirrors gt_perception.py, avoids cross-import)
_CARLA_TL_STATE_MAP: dict = {}   # populated after TrafficLightState import below

from config import (
    END_SPAWN_IDX,
    ROUTE_SAMPLING_RESOLUTION_M,
    TARGET_SPEED,
)
from perception.gt_perception import GTPerception
from perception.scene_state import (
    DetectedObject,
    EgoPose,
    SceneState,
    TrafficLightState,
    WaypointInfo,
)

_CARLA_TL_STATE_MAP.update({
    carla.TrafficLightState.Red:     TrafficLightState.RED,
    carla.TrafficLightState.Yellow:  TrafficLightState.YELLOW,
    carla.TrafficLightState.Green:   TrafficLightState.GREEN,
    carla.TrafficLightState.Off:     TrafficLightState.OFF,
    carla.TrafficLightState.Unknown: TrafficLightState.UNKNOWN,
})
from planning.behaviour_planner import BehaviourPlanner, BehaviourConfig
from planning.planner import PlannerConfig, RuleBasedPlanner, SpeedProfile
from planning.planner_output import PlannerOutput
from planning.route_planner import RoutePlanner

try:
    import rerun.blueprint as rrb
    _HAS_BLUEPRINT = True
except ImportError:
    _HAS_BLUEPRINT = False

# Planner reason → integer for the scalar time-series panel
_REASON_INT = {
    "cruise":           0,
    "speed_limit":      1,
    "junction_speed_cap": 2,
    "lead_vehicle":     3,
    "yellow_stop":      4,
    "yellow_continue":  5,
    "red_light_stop":   6,
    "emergency_brake":  7,
}

_TL_COLORS = {
    TrafficLightState.RED:     [255,  30,  30, 255],
    TrafficLightState.YELLOW:  [255, 200,   0, 255],
    TrafficLightState.GREEN:   [ 40, 220,  60, 255],
    TrafficLightState.OFF:     [120, 120, 120, 255],
    TrafficLightState.UNKNOWN: [120, 120, 120, 255],
}

# Rerun Boxes3D takes half-sizes
_VEHICLE_HALF     = [2.3, 0.9, 0.75]   # length/2, width/2, height/2
_PEDESTRIAN_HALF  = [0.3, 0.3, 0.9]
_EGO_HALF = [2.25, 1.0, 0.75]          # 4.5 × 2.0 × 1.5 m
_TL_HALF = 0.4
_TOPO_STEP_M = 2.0

# Do NOT call world.tick() — main.py owns the synchronous clock.
_VIZ_DT = 0.05   # 20 Hz matches main.py


def _c2r(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """Convert CARLA (left-hand Z-up) to Rerun (right-hand Z-up)."""
    return (x, -y, z)


def _yaw_to_quat(yaw_rad: float) -> List[float]:
    """
    CARLA yaw is clockwise (left-handed). Negate to get CCW (right-handed).
    Returns [w, x, y, z] quaternion for a pure Z-axis rotation.
    """
    half = -yaw_rad / 2.0
    return [math.cos(half), 0.0, 0.0, math.sin(half)]


def _world_to_bev(wx: float, wy: float, ego: "EgoPose") -> Tuple[float, float]:
    """
    CARLA world (wx, wy) → ego-local Rerun BEV.
    Result: X = ego-forward, Y = ego-left (Rerun right-hand).
    """
    dx = wx - ego.x
    dy = wy - ego.y
    cos_y = math.cos(ego.yaw_rad)
    sin_y = math.sin(ego.yaw_rad)
    local_x =  dx * cos_y + dy * sin_y
    local_y = -dx * sin_y + dy * cos_y
    return (local_x, -local_y)   # negate Y → Rerun right-hand


def _circle_strip(cx: float, cy: float, z: float, r: float, n: int = 18) -> List:
    step = 2 * math.pi / n
    return [[cx + r * math.cos(i * step), cy + r * math.sin(i * step), z]
            for i in range(n + 1)]


def _rect_strip(cx: float, cy: float, z: float, hw: float, hh: float) -> List:
    """Closed rectangle outline for LineStrips3D (hw=half-width, hh=half-height)."""
    return [
        [cx - hw, cy - hh, z],
        [cx + hw, cy - hh, z],
        [cx + hw, cy + hh, z],
        [cx - hw, cy + hh, z],
        [cx - hw, cy - hh, z],
    ]


def _octagon_strip(cx: float, cy: float, z: float, r: float) -> List:
    """Closed octagon in XY plane at height z (horizontal)."""
    return [[cx + r * math.cos(i * math.pi / 4),
             cy + r * math.sin(i * math.pi / 4), z]
            for i in range(9)]


def _circle_xz(cx: float, y: float, cz: float, r: float, n: int = 18) -> List:
    """Vertical circle in XZ plane at y — for standing sign rings."""
    step = 2 * math.pi / n
    return [[cx + r * math.cos(i * step), y, cz + r * math.sin(i * step)]
            for i in range(n + 1)]


def _octagon_xz(cx: float, y: float, cz: float, r: float) -> List:
    """Vertical octagon in XZ plane at y — for standing stop sign borders."""
    return [[cx + r * math.cos(i * math.pi / 4), y, cz + r * math.sin(i * math.pi / 4)]
            for i in range(9)]


def _height_to_color(z: float, z_min: float = -2.0, z_max: float = 4.0) -> List[int]:
    """Map a height value to an RGB colour (blue → green → red gradient)."""
    t = max(0.0, min(1.0, (z - z_min) / max(z_max - z_min, 1e-6)))
    r = int(255 * t)
    g = int(255 * (1.0 - abs(2 * t - 1.0)))
    b = int(255 * (1.0 - t))
    return [r, g, b, 200]


class CARLAVisualizer:
    """
    Rerun visualizer for the CARLA autonomous vehicle stack.

    Usage:
        viz = CARLAVisualizer(world, carla_map, ego_vehicle)
        viz.setup()
        # main loop:
        viz.update(scene_state, speed_profile)
        viz.close()
    """

    def __init__(
        self,
        world: "carla.World",
        carla_map: "carla.Map",
        ego_vehicle: "carla.Vehicle",
        recording_path: Optional[str] = None,
    ) -> None:
        self._world = world
        self._map = carla_map
        self._ego = ego_vehicle
        self._recording_path = recording_path

        self._lidar_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=2)
        self._lidar_sensor: Optional["carla.Sensor"] = None

        # Cache: actor_id → Rerun-space (x, y, z) for all TL actors
        self._tl_pos: dict[int, Tuple[float, float, float]] = {}
        # Cache: actor_id → CARLA yaw (radians) for housing orientation
        self._tl_yaw: dict[int, float] = {}
        # Cache: actor_id → carla.Actor ref (for live state queries)
        self._tl_actors: dict[int, "carla.Actor"] = {}

        # CARLA-world positions of static signs for BEV transform each tick
        # (cx, cy, cz, kmh) — speed limit signs
        self._speed_sign_cache: List[Tuple[float, float, float, int]] = []
        # (cx, cy, cz) — stop signs from world actors
        self._stop_sign_world_cache: List[Tuple[float, float, float]] = []

        self._frame: int = 0

    def setup(self) -> None:
        """Initialise Rerun, log static map geometry, hook LiDAR callback."""
        rr.init("carla_viz", spawn=True)

        if self._recording_path:
            rr.save(self._recording_path)
            print(f"[viz] Saving recording to {self._recording_path}")

        rr.log(
            "world",
            rr.ViewCoordinates.RIGHT_HAND_Z_UP,
            static=True,
        )

        self._send_blueprint()

        # Clear stale entity paths from previous design iterations
        for _stale in [
            "bev/traffic_lights/poles", "bev/traffic_lights/housings",
            "bev/traffic_lights/bulbs", "bev/traffic_lights/indicator",
            "bev/traffic_lights/glow",
            "bev/traffic_lights/active/fill",
            "bev/traffic_lights/bg/fill", "bev/traffic_lights/bg/ring",
            "bev/signs/speed/border_outer", "bev/signs/speed/border_inner",
            "bev/signs/stop/fill",
            "bev/signs/stop/border_outer", "bev/signs/stop/border_inner",
        ]:
            rr.log(_stale, rr.Clear(recursive=True))

        # Pre-compute traffic light world positions (used each tick)
        self._cache_traffic_light_positions()

        # Log static map geometry (expensive once, free thereafter)
        print("[viz] Building static map geometry …")
        self._log_static_map()
        print("[viz] Static map logged.")

        self._log_static_signs()
        self._attach_lidar()

        print("[viz] Setup complete — Rerun viewer should open.")

    def update(
        self,
        scene_state: SceneState,
        speed_profile: SpeedProfile,
        planner_out: "PlannerOutput | None" = None,
    ) -> None:
        """Log one frame of dynamic data to Rerun."""
        ts = scene_state.timestamp_s
        rr.set_time("sim_time", duration=ts)

        self._log_ego(scene_state)
        self._log_route(scene_state)
        self._log_traffic_lights(scene_state)
        self._log_stop_signs(scene_state)
        self._log_objects(scene_state)
        self._log_lidar(scene_state.ego_pose)
        self._log_bev_frame(scene_state)
        self._log_tl_debug(scene_state)
        self._log_scalars(scene_state, speed_profile, planner_out)
        self._log_planner_text(scene_state, speed_profile, planner_out)

        self._frame += 1

    def close(self) -> None:
        """Detach sensors and flush Rerun."""
        if self._lidar_sensor is not None and self._lidar_sensor.is_alive:
            self._lidar_sensor.stop()
            self._lidar_sensor.destroy()
            self._lidar_sensor = None
        print(f"[viz] Closed after {self._frame} frames.")

    # ------------------------------------------------------------------
    # Phase 2 stubs
    # ------------------------------------------------------------------

    def log_camera_frame(self, name: str, img_array: "np.ndarray") -> None:
        # Phase 2 — log RGB camera frames to Rerun image panels
        pass

    def log_voxel_grid(
        self,
        grid: "np.ndarray",
        origin: Tuple[float, float, float],
        voxel_size: float,
    ) -> None:
        # Phase 2 — log occupancy / semantic voxel grid as Points3D or Volume
        pass

    def _send_blueprint(self) -> None:
        if not _HAS_BLUEPRINT:
            print("[viz] rerun.blueprint unavailable — using default layout.")
            return
        try:
            blueprint = rrb.Blueprint(
                rrb.Horizontal(
                    rrb.Vertical(
                        rrb.Spatial3DView(name="World Model", origin="world"),
                        rrb.Spatial3DView(name="BEV (top-down)", origin="bev"),
                        row_shares=[3, 2],
                    ),
                    rrb.Vertical(
                        rrb.TimeSeriesView(
                            name="Speed",
                            contents=[
                                "scalars/ego_speed_mps",
                                "scalars/target_speed_mps",
                            ],
                        ),
                        rrb.TimeSeriesView(
                            name="Planner Rule",
                            contents=["scalars/planner_reason"],
                        ),
                        rrb.TextLogView(
                            name="Events",
                            contents=["logs/planner"],
                        ),
                        rrb.TextLogView(
                            name="Traffic Lights",
                            contents=["logs/traffic_lights"],
                        ),
                        row_shares=[2, 1, 1, 1],
                    ),
                    column_shares=[3, 1],
                )
            )
            rr.send_blueprint(blueprint)
        except Exception as exc:
            print(f"[viz] Blueprint failed ({exc}), using default layout.")

    def _cache_traffic_light_positions(self) -> None:
        """Build actor_id → position, CARLA yaw, and actor-ref maps for all TL actors."""
        for actor in self._world.get_actors().filter("traffic.traffic_light"):
            loc = actor.get_location()
            rot = actor.get_transform().rotation
            self._tl_pos[actor.id]   = _c2r(loc.x, loc.y, loc.z + 1.0)
            self._tl_yaw[actor.id]   = math.radians(rot.yaw)   # CARLA CW yaw
            self._tl_actors[actor.id] = actor
        print(f"[viz] Cached {len(self._tl_pos)} traffic light positions.")

    def _log_static_map(self) -> None:
        """
        Walk map topology edges to produce dense polyline strips.
        Lane segments → blue. Junction segments → orange.
        All strips batched into two rr.log calls with static=True.
        """
        topology = self._map.get_topology()

        lane_strips: List[List[Tuple[float, float, float]]] = []
        junction_strips: List[List[Tuple[float, float, float]]] = []

        seen_road_pairs: set[Tuple[int, int, int, int]] = set()

        for start_wp, end_wp in topology:
            # Deduplicate by full segment identity (road+section+lane at both ends)
            key = (
                start_wp.road_id, start_wp.section_id, start_wp.lane_id,
                end_wp.road_id,   end_wp.section_id,   end_wp.lane_id,
            )
            if key in seen_road_pairs:
                continue
            seen_road_pairs.add(key)

            strip: List[Tuple[float, float, float]] = []
            current = start_wp
            max_steps = 500
            steps = 0
            while steps < max_steps:
                loc = current.transform.location
                strip.append(_c2r(loc.x, loc.y, loc.z + 0.2))

                if (
                    current.road_id == end_wp.road_id
                    and current.lane_id == end_wp.lane_id
                ):
                    loc_end = end_wp.transform.location
                    strip.append(_c2r(loc_end.x, loc_end.y, loc_end.z + 0.2))
                    break

                nexts = current.next(_TOPO_STEP_M)
                if not nexts:
                    break

                # Follow the next waypoint that stays on the same road/lane if possible
                same = [
                    wp for wp in nexts
                    if wp.road_id == current.road_id and wp.lane_id == current.lane_id
                ]
                current = same[0] if same else nexts[0]
                steps += 1

            if len(strip) < 2:
                continue

            if start_wp.is_junction or end_wp.is_junction:
                junction_strips.append(strip)
            else:
                lane_strips.append(strip)

        if lane_strips:
            lane_colors = [[40, 90, 180, 200]] * len(lane_strips)
            rr.log(
                "world/map/lanes",
                rr.LineStrips3D(lane_strips, colors=lane_colors),
                static=True,
            )

        if junction_strips:
            junction_colors = [[255, 140, 0, 200]] * len(junction_strips)
            rr.log(
                "world/map/junctions",
                rr.LineStrips3D(junction_strips, colors=junction_colors),
                static=True,
            )

        print(
            f"[viz] Map: {len(lane_strips)} lane strips, "
            f"{len(junction_strips)} junction strips"
        )

    def _attach_lidar(self) -> None:
        """Find the LiDAR sensor attached to ego and register the callback."""
        lidar_actors = self._world.get_actors().filter("sensor.lidar.ray_cast")
        for sensor in lidar_actors:
            parent = sensor.parent
            if parent is not None and parent.id == self._ego.id:
                self._lidar_sensor = sensor
                sensor.listen(self._lidar_cb)
                print(f"[viz] LiDAR listener attached (actor id={sensor.id}).")
                return
        print("[viz] Warning: no LiDAR sensor found attached to ego.")

    def _log_bev_frame(self, state: SceneState) -> None:
        """
        Log all dynamic data in ego-local Rerun coordinates under bev/.
        Coordinate frame: X = ego-forward, Y = ego-left, Z = up.
        """
        ego = state.ego_pose
        z_ground = 0.3   # slight lift so lines float above the grid

        rr.log("bev/ego/body", rr.Boxes3D(
            centers=[[0.0, 0.0, 0.65]],
            half_sizes=[_EGO_HALF],          # 4.5 × 2.0 × 1.3 m
            colors=[[0, 210, 210, 235]],
            labels=[f"{ego.speed_mps*3.6:.0f} km/h"],
            radii=0.03,
        ))

        rr.log("bev/ego/roof", rr.Boxes3D(
            centers=[[0.25, 0.0, 1.60]],
            half_sizes=[[1.10, 0.72, 0.24]],
            colors=[[0, 180, 190, 220]],
            radii=0.02,
        ))

        _W_H = [0.35, 0.13, 0.30]           # half-sizes: length, width, height
        rr.log("bev/ego/wheels", rr.Boxes3D(
            centers=[
                [ 1.45,  1.02, 0.30],        # front-left
                [ 1.45, -1.02, 0.30],        # front-right
                [-1.25,  1.02, 0.30],        # rear-left
                [-1.25, -1.02, 0.30],        # rear-right
            ],
            half_sizes=[_W_H] * 4,
            colors=[[35, 35, 40, 255]] * 4,
            radii=0.01,
        ))

        # makes forward direction obvious
        rr.log("bev/ego/front_bumper", rr.LineStrips3D(
            [[[2.25,  1.05, 0.55], [2.25, -1.05, 0.55]]],
            colors=[[255, 255, 255, 240]],
            radii=0.10,
        ))

        # velocity-scaled heading arrow; length = max(4 m, speed_mps)
        arrow_len  = max(4.0, ego.speed_mps)
        shaft_x0   = 2.25
        shaft_x1   = shaft_x0 + arrow_len
        head_len   = min(1.2, arrow_len * 0.35)
        head_w     = 0.65
        arrow_z    = 1.85

        rr.log("bev/heading", rr.LineStrips3D(
            [
                # shaft
                [[shaft_x0, 0.0, arrow_z], [shaft_x1, 0.0, arrow_z]],
                # arrowhead left arm
                [[shaft_x1 - head_len,  head_w, arrow_z], [shaft_x1, 0.0, arrow_z]],
                # arrowhead right arm
                [[shaft_x1 - head_len, -head_w, arrow_z], [shaft_x1, 0.0, arrow_z]],
            ],
            colors=[[255, 240, 60, 230]],    # bright yellow
            radii=0.12,
        ))

        def _circle(r: float, n: int = 72) -> List:
            step = 2 * math.pi / n
            return [(r * math.cos(i * step), r * math.sin(i * step), 0.05)
                    for i in range(n + 1)]

        rr.log("bev/range/30m", rr.LineStrips3D(
            [_circle(30)], colors=[[70, 70, 70, 120]], radii=0.08,
        ))
        rr.log("bev/range/60m", rr.LineStrips3D(
            [_circle(60)], colors=[[50, 50, 50, 80]], radii=0.08,
        ))

        wps = state.route_waypoints
        if len(wps) >= 2:
            strips_bev: List[List] = []
            colors_bev: List[List[int]] = []
            seg_start = 0
            for i in range(1, len(wps)):
                zone_changed = (
                    self._speed_zone_color(wps[i].speed_limit_mps)
                    != self._speed_zone_color(wps[i - 1].speed_limit_mps)
                )
                at_end = (i == len(wps) - 1)
                if zone_changed or at_end:
                    end = i + 1 if at_end else i
                    seg = []
                    for j in range(seg_start, end):
                        bx, by = _world_to_bev(wps[j].x, wps[j].y, ego)
                        seg.append([bx, by, z_ground])
                    if len(seg) >= 2:
                        strips_bev.append(seg)
                        colors_bev.append(
                            self._speed_zone_color(wps[seg_start].speed_limit_mps)
                        )
                    seg_start = i
            if strips_bev:
                rr.log("bev/route", rr.LineStrips3D(
                    strips_bev, colors=colors_bev, radii=0.2,
                ))

        if state.dynamic_objects:
            centers_b, half_b, quats_b, colors_b, labels_b = [], [], [], [], []
            lead_id = state.lead_vehicle_id
            for obj in state.dynamic_objects:
                bx, by = _world_to_bev(obj.x, obj.y, ego)
                rel_yaw = obj.yaw_rad - ego.yaw_rad
                q = _yaw_to_quat(rel_yaw)
                rr_q = rr.Quaternion(xyzw=[q[1], q[2], q[3], q[0]])
                hs = _VEHICLE_HALF if obj.object_class == "vehicle" else _PEDESTRIAN_HALF
                is_lead = (obj.id == lead_id)
                centers_b.append([bx, by, hs[2]])
                half_b.append(hs)
                quats_b.append(rr_q)
                if is_lead:
                    colors_b.append([255, 140, 0, 240])
                elif obj.object_class == "vehicle":
                    colors_b.append([30, 100, 255, 220])
                else:
                    colors_b.append([255, 140, 0, 220])
                lead_tag = " [LEAD]" if is_lead else ""
                labels_b.append(f"{obj.object_class[:3]}{lead_tag} {obj.speed_mps:.0f}m/s")
            rr.log("bev/objects", rr.Boxes3D(
                centers=centers_b, half_sizes=half_b,
                quaternions=quats_b, colors=colors_b, labels=labels_b, radii=0.03,
            ))
        else:
            rr.log("bev/objects", rr.Clear(recursive=False))

        # Traffic lights — vertical housing on pole.
        # Route-relevant TLs: full housing + R/Y/G tiles + ring/label.
        # Background TLs (within 80 m): small dim cube only.
        _TL_POLE_TOP   = 3.50
        _TL_HOUSE_CZ   = 4.50
        _TL_HOUSE_HALF = [0.25, 0.15, 0.65]
        _TL_LIGHT_CZ   = [4.90, 4.50, 4.10]   # R, Y, G center heights
        _TL_LIGHT_HALF = [0.17, 0.07, 0.17]   # square face, thin depth
        _TL_TOP_CZ     = 5.25                  # flat — visible from top-down BEV
        _TL_TOP_HALF   = [0.40, 0.40, 0.03]
        _TL_RING_Z     = 5.35
        _TL_RING_R     = 0.55

        _LIGHT_LAYOUT = [
            (TrafficLightState.RED,    _TL_LIGHT_CZ[0], [255,  30,  30]),
            (TrafficLightState.YELLOW, _TL_LIGHT_CZ[1], [255, 200,   0]),
            (TrafficLightState.GREEN,  _TL_LIGHT_CZ[2], [ 40, 220,  60]),
        ]

        relevant_tl_ids = {tl.id: tl for tl in state.traffic_lights}

        # Map-based fallback for TLs missed by GTPerception's route filter
        # (viz route may differ from main.py route).
        ego_road_id: Optional[int] = None
        ego_lane_id: Optional[int] = None
        try:
            ego_loc = carla.Location(ego.x, ego.y, ego.z)
            ego_wp  = self._map.get_waypoint(
                ego_loc,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            if ego_wp is not None:
                ego_road_id = ego_wp.road_id
                ego_lane_id = ego_wp.lane_id
        except Exception:
            pass

        rel_poles:         List = []
        rel_house_centers: List = []
        rel_house_quats:   List = []
        rel_house_colors:  List = []
        rel_house_labels:  List = []
        rel_tile_centers:  List = []
        rel_tile_colors:   List = []
        rel_tile_quats:    List = []
        rel_top_centers:   List = []
        rel_top_colors:    List = []
        rel_top_labels:    List = []
        rel_rings:         List = []
        bg_centers: List = []
        bg_colors:  List = []

        for actor_id, actor in self._tl_actors.items():
            pos = self._tl_pos.get(actor_id)
            if pos is None:
                continue
            carla_wx, carla_wy = pos[0], -pos[1]
            bx, by = _world_to_bev(carla_wx, carla_wy, ego)
            dist_bev = math.hypot(bx, by)

            is_relevant = actor_id in relevant_tl_ids

            # promote to relevant if it affects ego's current road and is within 65 m
            if not is_relevant and ego_road_id is not None and dist_bev <= 65:
                try:
                    awps = actor.get_affected_lane_waypoints()
                    # Same road + same lane-direction sign (same side of road)
                    if any(
                        awp.road_id == ego_road_id
                        and (ego_lane_id is None or awp.lane_id * ego_lane_id > 0)
                        for awp in awps
                    ):
                        is_relevant = True
                except Exception:
                    pass

            # Always show route-relevant TLs; background TLs clipped at 80 m
            if dist_bev > 80 and not is_relevant:
                continue

            try:
                tl_state = _CARLA_TL_STATE_MAP.get(actor.get_state(), TrafficLightState.UNKNOWN)
            except Exception:
                tl_state = TrafficLightState.UNKNOWN

            tl_carla_yaw = self._tl_yaw.get(actor_id, 0.0)
            bev_yaw = tl_carla_yaw - ego.yaw_rad
            _q = _yaw_to_quat(bev_yaw)
            rr_q = rr.Quaternion(xyzw=[_q[1], _q[2], _q[3], _q[0]])

            sc = _TL_COLORS.get(tl_state, [120, 120, 120, 255])

            if is_relevant:
                tl_info = relevant_tl_ids.get(actor_id)
                dist_lbl = f"{tl_info.distance_m:.0f}m" if tl_info else f"{dist_bev:.0f}m*"
                lbl = f"{tl_state.value.upper()} {dist_lbl}"

                rel_poles.append([[bx, by, 0.0], [bx, by, _TL_POLE_TOP]])

                rel_house_centers.append([bx, by, _TL_HOUSE_CZ])
                rel_house_quats.append(rr_q)
                rel_house_colors.append([22, 22, 28, 235])
                rel_house_labels.append(lbl)

                # Lights stacked vertically — same BEV X/Y, only Z differs
                for ls, cz, base_rgb in _LIGHT_LAYOUT:
                    active = (tl_state == ls)
                    rel_tile_centers.append([bx, by, cz])
                    rel_tile_colors.append([*base_rgb, 255 if active else 28])
                    rel_tile_quats.append(rr_q)

                rel_top_centers.append([bx, by, _TL_TOP_CZ])
                rel_top_colors.append(sc)
                rel_top_labels.append(lbl)
                rel_rings.append(_circle_strip(bx, by, _TL_RING_Z, _TL_RING_R))
            else:
                bg_centers.append([bx, by, _TL_HOUSE_CZ])
                bg_colors.append([sc[0], sc[1], sc[2], 55])

        if rel_poles:
            n_r = len(rel_poles)
            rr.log("bev/traffic_lights/active/poles",
                   rr.LineStrips3D(rel_poles,
                                   colors=[[140, 140, 145, 220]] * n_r, radii=0.06))
            rr.log("bev/traffic_lights/active/housing",
                   rr.Boxes3D(
                       centers=rel_house_centers,
                       half_sizes=[_TL_HOUSE_HALF] * n_r,
                       quaternions=rel_house_quats,
                       colors=rel_house_colors,
                       radii=0.02,
                   ))
            n_t = len(rel_tile_centers)
            rr.log("bev/traffic_lights/active/lights",
                   rr.Boxes3D(
                       centers=rel_tile_centers,
                       half_sizes=[_TL_LIGHT_HALF] * n_t,
                       quaternions=rel_tile_quats,
                       colors=rel_tile_colors,
                       radii=0.01,
                   ))
            rr.log("bev/traffic_lights/active/topdot",
                   rr.Boxes3D(
                       centers=rel_top_centers,
                       half_sizes=[_TL_TOP_HALF] * n_r,
                       colors=rel_top_colors,
                       labels=rel_top_labels,
                       radii=0.01,
                   ))
            rr.log("bev/traffic_lights/active/ring",
                   rr.LineStrips3D(rel_rings,
                                   colors=[[255, 255, 255, 200]] * n_r, radii=0.07))
        else:
            rr.log("bev/traffic_lights/active", rr.Clear(recursive=True))

        if bg_centers:
            nb = len(bg_centers)
            rr.log("bev/traffic_lights/bg",
                   rr.Boxes3D(
                       centers=bg_centers,
                       half_sizes=[[0.22, 0.22, 0.22]] * nb,
                       colors=bg_colors,
                   ))
        else:
            rr.log("bev/traffic_lights/bg", rr.Clear(recursive=True))

        # Speed limit signs — vertical panel on pole.
        _SP_POLE_TOP  = 2.50
        _SP_FACE_CZ   = 2.70
        _SP_FACE_HALF = [0.30, 0.02, 0.30]   # width, depth (thin), height
        _SP_RING_R    = 0.33
        _SP_TOP_CZ    = 3.07
        _SP_TOP_HALF  = [0.30, 0.30, 0.03]   # flat for top-down visibility

        sp_poles:      List = []
        sp_faces:      List = []
        sp_rings:      List = []
        sp_ring_colors: List = []
        sp_tops:       List = []
        sp_top_labels: List = []

        for cx, cy, cz, kmh in self._speed_sign_cache:
            bx, by = _world_to_bev(cx, cy, ego)
            if math.hypot(bx, by) > 75:
                continue
            sp_poles.append([[bx, by, 0.0], [bx, by, _SP_POLE_TOP]])
            sp_faces.append([bx, by, _SP_FACE_CZ])
            sp_rings.append(_circle_xz(bx, by, _SP_FACE_CZ, _SP_RING_R))
            sp_ring_colors.append(self._speed_limit_sign_color(kmh))
            sp_tops.append([bx, by, _SP_TOP_CZ])
            sp_top_labels.append(str(kmh))

        if sp_poles:
            n = len(sp_poles)
            rr.log("bev/signs/speed/poles",
                   rr.LineStrips3D(sp_poles,
                                   colors=[[140, 140, 145, 210]] * n, radii=0.045))
            rr.log("bev/signs/speed/face",
                   rr.Boxes3D(
                       centers=sp_faces,
                       half_sizes=[_SP_FACE_HALF] * n,
                       colors=[[240, 240, 240, 235]] * n,
                       radii=0.01,
                   ))
            rr.log("bev/signs/speed/ring",
                   rr.LineStrips3D(sp_rings, colors=sp_ring_colors, radii=0.07))
            rr.log("bev/signs/speed/topdot",
                   rr.Boxes3D(
                       centers=sp_tops,
                       half_sizes=[_SP_TOP_HALF] * n,
                       colors=[[240, 240, 240, 220]] * n,
                       labels=sp_top_labels,
                       radii=0.01,
                   ))
        else:
            rr.log("bev/signs/speed", rr.Clear(recursive=True))

        # Stop signs — vertical panel on pole.
        _ST_POLE_TOP  = 2.30
        _ST_FACE_CZ   = 2.50
        _ST_FACE_HALF = [0.28, 0.02, 0.28]   # width, depth (thin), height
        _ST_OCT_R     = 0.30
        _ST_TOP_CZ    = 2.87
        _ST_TOP_HALF  = [0.28, 0.28, 0.03]

        st_poles:      List = []
        st_faces:      List = []
        st_octs:       List = []
        st_tops:       List = []
        st_top_labels: List = []

        for ss in state.stop_signs:
            bx, by = _world_to_bev(ss.location_x, ss.location_y, ego)
            st_poles.append([[bx, by, 0.0], [bx, by, _ST_POLE_TOP]])
            st_faces.append([bx, by, _ST_FACE_CZ])
            st_octs.append(_octagon_xz(bx, by, _ST_FACE_CZ, _ST_OCT_R))
            st_tops.append([bx, by, _ST_TOP_CZ])
            st_top_labels.append(f"STOP {ss.distance_m:.0f}m")

        if st_poles:
            n = len(st_poles)
            rr.log("bev/signs/stop/poles",
                   rr.LineStrips3D(st_poles,
                                   colors=[[140, 140, 145, 210]] * n, radii=0.045))
            rr.log("bev/signs/stop/face",
                   rr.Boxes3D(
                       centers=st_faces,
                       half_sizes=[_ST_FACE_HALF] * n,
                       colors=[[200, 20, 20, 235]] * n,
                       radii=0.01,
                   ))
            rr.log("bev/signs/stop/border",
                   rr.LineStrips3D(st_octs,
                                   colors=[[255, 255, 255, 230]] * n, radii=0.09))
            rr.log("bev/signs/stop/topdot",
                   rr.Boxes3D(
                       centers=st_tops,
                       half_sizes=[_ST_TOP_HALF] * n,
                       colors=[[200, 20, 20, 220]] * n,
                       labels=st_top_labels,
                       radii=0.01,
                   ))
        else:
            rr.log("bev/signs/stop", rr.Clear(recursive=True))

    def _log_static_signs(self) -> None:
        """
        Cache all sign actor positions for per-tick BEV transform,
        and log their world-frame 3D structure once as static geometry.
        """
        speed_pole_strips, speed_face_centers, speed_face_colors, speed_labels = [], [], [], []
        stop_pole_strips, stop_octagons, stop_face_centers = [], [], []

        for actor in self._world.get_actors():
            tid = actor.type_id
            loc = actor.get_location()

            if tid.startswith("traffic.speed_limit."):
                try:
                    kmh = int(tid.split(".")[-1])
                except ValueError:
                    kmh = 0

                # Cache CARLA world coords for BEV transform each tick
                self._speed_sign_cache.append((loc.x, loc.y, loc.z, kmh))

                rx, ry, rz = _c2r(loc.x, loc.y, loc.z)
                sign_z = rz + 2.5

                speed_pole_strips.append([[rx, ry, rz], [rx, ry, sign_z]])
                speed_face_centers.append([rx, ry, sign_z])
                speed_face_colors.append(self._speed_limit_sign_color(kmh))
                speed_labels.append(f"{kmh}")

            elif tid == "traffic.stop":
                self._stop_sign_world_cache.append((loc.x, loc.y, loc.z))

                rx, ry, rz = _c2r(loc.x, loc.y, loc.z)
                sign_z = rz + 2.2

                stop_pole_strips.append([[rx, ry, rz], [rx, ry, sign_z]])
                stop_octagons.append(_octagon_strip(rx, ry, sign_z, 0.4))
                stop_face_centers.append([rx, ry, sign_z])

        n_speed = len(speed_face_centers)
        n_stop  = len(stop_face_centers)
        print(f"[viz] Signs: {n_speed} speed limits, {n_stop} stops")

        if speed_pole_strips:
            rr.log("world/signs/speed/poles",
                   rr.LineStrips3D(speed_pole_strips,
                                   colors=[[160, 160, 160, 200]] * n_speed,
                                   radii=0.04),
                   static=True)
            rr.log("world/signs/speed/faces",
                   rr.Points3D(speed_face_centers,
                                colors=speed_face_colors,
                                radii=0.35,
                                labels=speed_labels),
                   static=True)

        if stop_pole_strips:
            rr.log("world/signs/stop/poles",
                   rr.LineStrips3D(stop_pole_strips,
                                   colors=[[160, 160, 160, 200]] * n_stop,
                                   radii=0.04),
                   static=True)
            rr.log("world/signs/stop/faces",
                   rr.LineStrips3D(stop_octagons,
                                   colors=[[220, 30, 30, 230]] * n_stop,
                                   radii=0.08),
                   static=True)
            rr.log("world/signs/stop/fill",
                   rr.Points3D(stop_face_centers,
                                colors=[[220, 30, 30, 160]] * n_stop,
                                radii=0.38,
                                labels=["STOP"] * n_stop),
                   static=True)

    @staticmethod
    def _speed_limit_sign_color(kmh: int) -> List[int]:
        """Encode speed danger via border colour: green → yellow-green → orange → red."""
        if kmh <= 30:
            return [60, 200, 60, 255]
        elif kmh <= 50:
            return [200, 210, 30, 255]
        elif kmh <= 70:
            return [255, 165, 0, 255]
        elif kmh <= 90:
            return [240, 100, 0, 255]
        else:
            return [210, 30, 30, 255]

    def _log_stop_signs(self, state: SceneState) -> None:
        """Highlight stop signs that are currently in the SceneState (within ~60 m)."""
        if not state.stop_signs:
            rr.log("world/stop_signs_active", rr.Clear(recursive=False))
            return

        centers, labels = [], []
        for ss in state.stop_signs:
            x, y, z = _c2r(ss.location_x, ss.location_y, state.ego_pose.z + 1.8)
            centers.append([x, y, z])
            labels.append(f"STOP {ss.distance_m:.0f}m")

        rr.log(
            "world/stop_signs_active",
            rr.Boxes3D(
                centers=centers,
                half_sizes=[[0.4, 0.1, 0.4]] * len(centers),
                colors=[[255, 30, 30, 255]] * len(centers),
                labels=labels,
                radii=0.04,
            ),
        )

    def _lidar_cb(self, point_cloud: "carla.LidarMeasurement") -> None:
        """CARLA sensor callback — runs in CARLA's thread, keep it fast."""
        pts = np.frombuffer(
            point_cloud.raw_data, dtype=np.float32
        ).reshape(-1, 4).copy()   # copy so we don't hold a reference to CARLA memory
        if not self._lidar_q.full():
            self._lidar_q.put_nowait(pts)

    def _log_ego(self, state: SceneState) -> None:
        pose = state.ego_pose
        quat   = _yaw_to_quat(pose.yaw_rad)
        rr_q   = rr.Quaternion(xyzw=[quat[1], quat[2], quat[3], quat[0]])
        cos_y  = math.cos(pose.yaw_rad)
        sin_y  = math.sin(pose.yaw_rad)

        def _world_offset(dx: float, dy: float, dz: float):
            """Apply ego heading rotation to a local offset, then convert to Rerun."""
            wx = pose.x + dx * cos_y - dy * sin_y
            wy = pose.y + dx * sin_y + dy * cos_y
            wz = pose.z + dz
            return list(_c2r(wx, wy, wz))

        rr.log("world/ego/body", rr.Boxes3D(
            centers=[_world_offset(0.0, 0.0, 0.65)],
            half_sizes=[_EGO_HALF],
            quaternions=[rr_q],
            colors=[[0, 210, 210, 235]],
            labels=[f"ego {pose.speed_mps*3.6:.0f}km/h"],
            radii=0.03,
        ))

        rr.log("world/ego/roof", rr.Boxes3D(
            centers=[_world_offset(0.25, 0.0, 1.60)],
            half_sizes=[[1.10, 0.72, 0.24]],
            quaternions=[rr_q],
            colors=[[0, 180, 190, 220]],
            radii=0.02,
        ))

    @staticmethod
    def _speed_zone_color(speed_mps: float) -> List[int]:
        """Color route segment by speed limit zone."""
        kmh = speed_mps * 3.6
        if kmh <= 30:
            return [40, 220, 60, 220]
        elif kmh <= 50:
            return [180, 220, 0, 220]
        elif kmh <= 90:
            return [255, 140, 0, 220]
        else:
            return [220, 40, 40, 220]

    def _log_route(self, state: SceneState) -> None:
        """Log route as speed-zone colour-coded segments."""
        wps: List[WaypointInfo] = state.route_waypoints
        if len(wps) < 2:
            return

        z = state.ego_pose.z + 0.3
        strips: List[List] = []
        colors: List[List[int]] = []

        # Group consecutive waypoints that share the same speed limit zone
        seg_start = 0
        for i in range(1, len(wps)):
            zone_changed = (
                self._speed_zone_color(wps[i].speed_limit_mps)
                != self._speed_zone_color(wps[i - 1].speed_limit_mps)
            )
            at_end = (i == len(wps) - 1)
            if zone_changed or at_end:
                end = i + 1 if at_end else i
                seg = [_c2r(wps[j].x, wps[j].y, z) for j in range(seg_start, end)]
                if len(seg) >= 2:
                    strips.append(seg)
                    colors.append(self._speed_zone_color(wps[seg_start].speed_limit_mps))
                seg_start = i

        if strips:
            rr.log("world/route", rr.LineStrips3D(strips, colors=colors, radii=0.07))

    def _log_traffic_lights(self, state: SceneState) -> None:
        """Log each visible traffic light as a coloured small cube."""
        for tl_info in state.traffic_lights:
            color = _TL_COLORS.get(tl_info.state, [120, 120, 120, 255])
            pos = self._tl_pos.get(tl_info.id)
            if pos is None:
                # Actor appeared after setup — look up now and cache
                actors = self._world.get_actors().filter("traffic.traffic_light")
                for actor in actors:
                    if actor.id == tl_info.id:
                        loc = actor.get_location()
                        pos = _c2r(loc.x, loc.y, loc.z + 1.0)
                        self._tl_pos[tl_info.id] = pos
                        break
            if pos is None:
                continue

            rr.log(
                f"world/traffic_lights/{tl_info.id}",
                rr.Boxes3D(
                    centers=[list(pos)],
                    half_sizes=[[_TL_HALF, _TL_HALF, _TL_HALF]],
                    colors=[color],
                    labels=[f"{tl_info.state.value} {tl_info.distance_m:.0f}m"],
                    radii=0.02,
                ),
            )

    def _log_objects(self, state: SceneState) -> None:
        """Batch-log all dynamic objects (vehicles + pedestrians). Lead vehicle shown in orange."""
        objects: List[DetectedObject] = state.dynamic_objects
        if not objects:
            rr.log("world/objects", rr.Clear(recursive=False))
            return

        centers = []
        half_sizes = []
        quaternions = []
        colors = []
        labels = []

        lead_id = state.lead_vehicle_id

        for obj in objects:
            cx, cy, cz = _c2r(obj.x, obj.y, state.ego_pose.z + 0.75)
            centers.append([cx, cy, cz])

            is_lead = (obj.id == lead_id)
            if obj.object_class == "vehicle":
                half_sizes.append(_VEHICLE_HALF)
                colors.append([255, 140, 0, 240] if is_lead else [30, 100, 255, 200])
            else:
                half_sizes.append(_PEDESTRIAN_HALF)
                colors.append([255, 80, 0, 240] if is_lead else [255, 140, 0, 220])

            quat = _yaw_to_quat(obj.yaw_rad)
            quaternions.append(
                rr.Quaternion(xyzw=[quat[1], quat[2], quat[3], quat[0]])
            )
            lead_tag = " [LEAD]" if is_lead else ""
            labels.append(f"{obj.object_class}{lead_tag} {obj.speed_mps:.1f}m/s")

        rr.log(
            "world/objects",
            rr.Boxes3D(
                centers=centers,
                half_sizes=half_sizes,
                quaternions=quaternions,
                colors=colors,
                labels=labels,
                radii=0.03,
            ),
        )

    def _log_lidar(self, ego_pose: "EgoPose") -> None:
        """Drain LiDAR queue, transform to world frame, and log."""
        pts: Optional[np.ndarray] = None
        try:
            while True:
                pts = self._lidar_q.get_nowait()
        except queue.Empty:
            pass

        if pts is None or len(pts) == 0:
            return

        # pts columns: x, y, z, intensity — sensor-local CARLA frame
        # transform: rotate by ego yaw → translate to world → negate Y
        x_s = pts[:, 0]
        y_s = pts[:, 1]
        z_s = pts[:, 2]

        cos_y = math.cos(ego_pose.yaw_rad)
        sin_y = math.sin(ego_pose.yaw_rad)

        x_w = x_s * cos_y - y_s * sin_y + ego_pose.x
        y_w = x_s * sin_y + y_s * cos_y + ego_pose.y
        z_w = z_s + ego_pose.z + 1.8   # 1.8 m = LiDAR mount height

        # CARLA → Rerun: negate Y
        xyz = np.stack([x_w, -y_w, z_w], axis=1)

        z_vals = xyz[:, 2]
        z_min, z_max = float(z_vals.min()), float(z_vals.max())
        if z_max == z_min:
            z_max = z_min + 1.0

        t = np.clip((z_vals - z_min) / (z_max - z_min), 0.0, 1.0)
        r_ch = (255 * t).astype(np.uint8)
        g_ch = (255 * (1.0 - np.abs(2 * t - 1.0))).astype(np.uint8)
        b_ch = (255 * (1.0 - t)).astype(np.uint8)
        a_ch = np.full(len(t), 200, dtype=np.uint8)
        colors = np.stack([r_ch, g_ch, b_ch, a_ch], axis=1)

        rr.log(
            "world/lidar",
            rr.Points3D(
                xyz,          # numpy array passed directly — avoid .tolist() on 70k pts
                colors=colors,
                radii=0.05,
            ),
        )

        # BEV: sensor-local x=forward, y=right(CARLA) → ego-local x=forward, y=left(Rerun)
        bev_xyz = np.stack([x_s, -y_s, z_s + 1.8], axis=1)
        rr.log(
            "bev/lidar",
            rr.Points3D(bev_xyz, colors=colors, radii=0.05),
        )

    def _log_tl_debug(self, state: SceneState) -> None:
        """
        Log TL diagnostics each tick. Columns: cache (all Town01 TLs),
        range (within 80 m), scene (route-relevant, full housing).
        """
        ego = state.ego_pose
        n_world = len(self._tl_actors)

        n_in_range = 0
        for actor_id in self._tl_actors:
            pos = self._tl_pos.get(actor_id)
            if pos is None:
                continue
            bx, by = _world_to_bev(pos[0], -pos[1], ego)
            if math.hypot(bx, by) <= 80:
                n_in_range += 1

        n_scene = len(state.traffic_lights)

        lines = [
            f"[TL] cache={n_world}  in_range(80m)={n_in_range}  "
            f"route-relevant={n_scene}"
        ]
        for tl in state.traffic_lights:
            try:
                actor = self._tl_actors.get(tl.id)
                pos = self._tl_pos.get(tl.id)
                bev_str = ""
                if pos is not None:
                    bx, by = _world_to_bev(pos[0], -pos[1], ego)
                    bev_str = f"  bev=({bx:+.1f},{by:+.1f})"
                lines.append(
                    f"  id={tl.id:5d}  {tl.state.value.upper():<8s}"
                    f"  dist={tl.distance_m:5.1f}m{bev_str}"
                )
            except Exception as exc:
                lines.append(f"  id={tl.id}  ERROR: {exc}")

        msg = "\n".join(lines)
        rr.log("logs/traffic_lights", rr.TextLog(msg, level=rr.TextLogLevel.INFO))

        if self._frame % 20 == 0:
            print(msg)

    # FSM state → integer for the scalar time-series panel
    _FSM_INT = {
        "lane_follow":          0,
        "follow_vehicle":       1,
        "prepare_lane_change":  2,
        "lane_change":          3,
        "emergency_stop":       4,
        "slow_for_crosswalk":   5,
        "stop_and_go":          6,
    }

    def _log_scalars(
        self,
        state: SceneState,
        speed_profile: SpeedProfile,
        planner_out: "PlannerOutput | None" = None,
    ) -> None:
        ego_speed = state.ego_pose.speed_mps
        target_speed = (
            speed_profile.target_speeds_mps[0]
            if speed_profile.target_speeds_mps
            else 0.0
        )
        reason_int = _REASON_INT.get(speed_profile.reason, 0)

        rr.log("scalars/ego_speed_mps",    rr.Scalars(ego_speed))
        rr.log("scalars/target_speed_mps", rr.Scalars(target_speed))
        rr.log("scalars/planner_reason",   rr.Scalars(float(reason_int)))

        if planner_out is not None:
            fsm_int = self._FSM_INT.get(planner_out.fsm_state, -1)
            rr.log("scalars/fsm_state", rr.Scalars(float(fsm_int)))
            rr.log("scalars/acc_speed_mps", rr.Scalars(planner_out.target_speed_mps))
            if state.lead_vehicle_distance_m >= 0:
                rr.log("scalars/lead_dist_m", rr.Scalars(state.lead_vehicle_distance_m))

    def _log_planner_text(
        self,
        state: SceneState,
        speed_profile: SpeedProfile,
        planner_out: "PlannerOutput | None" = None,
    ) -> None:
        nearest_tl = (
            min(state.traffic_lights, key=lambda t: t.distance_m)
            if state.traffic_lights else None
        )
        nearest_obj = (
            min(state.dynamic_objects, key=lambda o: o.distance_m)
            if state.dynamic_objects else None
        )

        tl_str = (
            f"TL {nearest_tl.state.value.upper()} {nearest_tl.distance_m:.1f}m"
            if nearest_tl else "TL none"
        )
        obj_str = (
            f"OBJ {nearest_obj.object_class} {nearest_obj.distance_m:.1f}m"
            if nearest_obj else "OBJ none"
        )
        spd_kmh = state.ego_pose.speed_mps * 3.6
        tgt_kmh = (
            speed_profile.target_speeds_mps[0] * 3.6
            if speed_profile.target_speeds_mps else 0.0
        )
        fsm_str = f"fsm={planner_out.fsm_state}" if planner_out else ""
        lead_str = (
            f"LEAD {state.lead_vehicle_distance_m:.1f}m {state.lead_vehicle_speed_mps*3.6:.1f}km/h"
            if state.lead_vehicle_id is not None else "LEAD none"
        )

        msg = (
            f"[{self._frame:5d}] {speed_profile.reason:<22s} "
            f"{fsm_str:<22s} "
            f"spd={spd_kmh:5.1f}km/h tgt={tgt_kmh:5.1f}km/h  "
            f"{tl_str}  {obj_str}  {lead_str}"
        )
        rr.log("logs/planner", rr.TextLog(msg, level=rr.TextLogLevel.INFO))


def _find_ego_vehicle(world: "carla.World") -> Optional["carla.Vehicle"]:
    """
    Locate the ego vehicle in the running CARLA world.
    Priority:
      1. Any vehicle with role_name == "hero"
      2. First vehicle matching VEHICLE_MODEL from config
      3. Any vehicle (fallback)
    Returns None if no vehicle found at all.
    """
    vehicles = world.get_actors().filter("vehicle.*")
    if not vehicles:
        return None

    for v in vehicles:
        if v.attributes.get("role_name") == "hero":
            return v

    from config import VEHICLE_MODEL
    for v in vehicles:
        if VEHICLE_MODEL in v.type_id:
            return v

    return list(vehicles)[0]


def _nearest_spawn_to(
    spawn_points: List["carla.Transform"],
    location: "carla.Location",
) -> int:
    """Return the index of the spawn point closest to `location`."""
    best_idx = 0
    best_dist = float("inf")
    for i, sp in enumerate(spawn_points):
        d = math.hypot(
            sp.location.x - location.x,
            sp.location.y - location.y,
        )
        if d < best_dist:
            best_dist = d
            best_idx = i
    return best_idx


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rerun visualizer for the CARLA AV stack."
    )
    parser.add_argument(
        "--host", default="localhost",
        help="CARLA server host (default: localhost)",
    )
    parser.add_argument(
        "--port", type=int, default=2000,
        help="CARLA server port (default: 2000)",
    )
    parser.add_argument(
        "--save", default=None, metavar="PATH",
        help="Optional .rrd path to save the Rerun recording.",
    )
    args = parser.parse_args()

    print(f"[viz] Connecting to CARLA at {args.host}:{args.port} …")
    client = carla.Client(args.host, args.port)
    client.set_timeout(15.0)
    world = client.get_world()
    carla_map = world.get_map()
    print(f"[viz] Connected. Map: {carla_map.name}")

    ego_vehicle = None
    print("[viz] Waiting for ego vehicle …", end="", flush=True)
    while ego_vehicle is None:
        ego_vehicle = _find_ego_vehicle(world)
        if ego_vehicle is None:
            print(".", end="", flush=True)
            time.sleep(1.0)
    print(f"\n[viz] Ego found: {ego_vehicle.type_id} (id={ego_vehicle.id})")

    spawn_points = carla_map.get_spawn_points()
    route: list = []
    current_idx: int = 0

    try:
        ego_loc = ego_vehicle.get_location()
        start_idx = _nearest_spawn_to(spawn_points, ego_loc)
        end_idx = min(END_SPAWN_IDX, len(spawn_points) - 1)
        if start_idx == end_idx:
            end_idx = (end_idx + 1) % len(spawn_points)

        start_loc = spawn_points[start_idx].location
        end_loc = spawn_points[end_idx].location

        route_planner = RoutePlanner(carla_map, ROUTE_SAMPLING_RESOLUTION_M)
        route = route_planner.compute(start_loc, end_loc)
        print(
            f"[viz] Route built: {len(route)} waypoints "
            f"(spawn[{start_idx}] → spawn[{end_idx}])"
        )
    except Exception as exc:
        print(f"[viz] Warning: route planning failed ({exc}). Ego and map still shown.")
        route = []

    gt_perception = GTPerception(world, ego_vehicle)
    planner = RuleBasedPlanner(PlannerConfig(cruise_speed_mps=TARGET_SPEED))
    behaviour_planner = BehaviourPlanner(BehaviourConfig(cruise_speed_mps=TARGET_SPEED))
    viz = CARLAVisualizer(
        world=world,
        carla_map=carla_map,
        ego_vehicle=ego_vehicle,
        recording_path=args.save,
    )
    viz.setup()

    def _advance_idx(ego_x: float, ego_y: float, idx: int, window: int = 10) -> int:
        """Slide current_idx forward when we are past a waypoint."""
        if not route:
            return 0
        best = idx
        best_dist = float("inf")
        end = min(idx + window, len(route))
        for i in range(max(0, idx - 2), end):
            wp, _ = route[i]
            loc = wp.transform.location
            d = math.hypot(ego_x - loc.x, ego_y - loc.y)
            if d < best_dist:
                best_dist = d
                best = i
        return best

    print("[viz] Starting visualizer loop. Press Ctrl+C to stop.")
    try:
        while True:
            loop_start = time.monotonic()

            # Re-acquire ego in case it was respawned by main.py
            new_ego = _find_ego_vehicle(world)
            if new_ego is not None and new_ego.id != ego_vehicle.id:
                print(
                    f"[viz] Ego changed: {ego_vehicle.id} → {new_ego.id}. "
                    "Re-attaching LiDAR …"
                )
                ego_vehicle = new_ego
                gt_perception = GTPerception(world, ego_vehicle)
                viz._ego = ego_vehicle
                viz.close()
                viz.setup()

            ego_loc = ego_vehicle.get_location()
            current_idx = _advance_idx(ego_loc.x, ego_loc.y, current_idx)

            # Perception → planner → visualizer
            try:
                scene_state = gt_perception.update(
                    full_route=route, current_idx=current_idx
                )
            except Exception as exc:
                print(f"[viz] GTPerception.update failed: {exc}")
                time.sleep(_VIZ_DT)
                continue

            planner_out = None
            try:
                planner_out = behaviour_planner.plan(scene_state)
                planner.config.cruise_speed_mps = planner_out.target_speed_mps
            except Exception as exc:
                print(f"[viz] BehaviourPlanner.plan failed: {exc}")

            try:
                speed_profile = planner.plan(scene_state)
            except Exception as exc:
                print(f"[viz] Planner.plan failed: {exc}")
                from planning.planner import SpeedProfile as _SP
                speed_profile = _SP(target_speeds_mps=[], reason="cruise")

            viz.update(scene_state, speed_profile, planner_out)

            # Honour the target DT (don't drift faster than main.py)
            elapsed = time.monotonic() - loop_start
            sleep_s = max(0.0, _VIZ_DT - elapsed)
            time.sleep(sleep_s)

    except KeyboardInterrupt:
        print("\n[viz] KeyboardInterrupt — shutting down.")
    finally:
        viz.close()


if __name__ == "__main__":
    main()
