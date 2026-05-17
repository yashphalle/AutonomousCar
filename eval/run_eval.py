"""
Stage 2 evaluation harness.

Usage:
    python -m eval.run_eval --frames 500 --npcs 10
    python -m eval.run_eval --seed 42 --routes all
    python -m eval.run_eval --seed 42 --routes 1,3,5
"""

from __future__ import annotations

import argparse
import collections
import csv
import logging
import math
import os
import random
import sys
import time
import threading
from typing import Optional

# Bootstrap CARLA sys.path before importing carla or any module that imports it.
from utils import carla_bootstrap  # noqa: F401

import carla
import numpy as np

import config
from control.pid_controller import PIDController
from eval.routes import EVAL_ROUTES, EvalRoute
from perception.gt_perception import GTPerception
from perception.scene_state import TrafficLightState
from planning.behaviour_planner import BehaviourPlanner, BehaviourConfig
from planning.planner import RuleBasedPlanner, PlannerConfig
from planning.planner_output import PlannerOutput
from planning.route_planner import RoutePlanner
from planning.waypoint_manager import WaypointManager
from planning.lane_aware_waypoint_manager import LaneAwareWaypointManager
from utils.carla_utils import get_ego_state, normalize_angle, output_to_throttle_brake
from utils.run_logger import RunLogger

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("eval")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_ROUTE_TIME_S = 300.0        # hard timeout per route (5 min — accounts for TL wait cycles)
NPC_COUNT = 10                  # default NPC vehicles
PEDESTRIAN_COUNT = 3            # pedestrians per route
COLLISION_WINDOW_TICKS = 20     # 1 s at 20 Hz — rolling window for fault check
EGO_SPEED_FAULT_THRESHOLD = 0.5 # m/s — ego must be moving for "planner fault"
RED_LIGHT_STOP_DIST_M = 3.0    # ego past stop line if TL dist < this


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2 CARLA Eval Harness")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--routes",
        default="all",
        help="'all' or comma-separated route IDs, e.g. '1,3,5'",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Maximum frames per route (overrides 120 s timeout when set)",
    )
    parser.add_argument("--npcs", type=int, default=NPC_COUNT)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def make_output_dir() -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = os.path.join("eval", "results", f"baseline_gt_{ts}")
    os.makedirs(out, exist_ok=True)
    return out


# ---------------------------------------------------------------------------
# Collision sensor wrapper
# ---------------------------------------------------------------------------
class CollisionTracker:
    """Attaches a collision sensor to the ego and tracks collision events."""

    def __init__(self, world: carla.World, vehicle: carla.Actor):
        self._vehicle = vehicle
        self._lock = threading.Lock()
        self.total_collisions = 0
        self.planner_fault_collisions = 0

        # Rolling window: {actor_id -> count of ticks seen} — updated each sim tick
        # Populated from outside via update_dynamic_window().
        self._dynamic_window: dict[int, int] = {}

        # Speed samples from last COLLISION_WINDOW_TICKS ticks
        self._speed_window: collections.deque[float] = collections.deque(
            maxlen=COLLISION_WINDOW_TICKS
        )

        bp = world.get_blueprint_library().find("sensor.other.collision")
        transform = carla.Transform()
        self._sensor = world.spawn_actor(bp, transform, attach_to=vehicle)
        self._sensor.listen(self._on_collision)

    def _on_collision(self, event: carla.CollisionEvent) -> None:
        with self._lock:
            self.total_collisions += 1
            other_id = event.other_actor.id

            # Planner-fault check:
            # 1. Ego was moving (mean speed > threshold) in the 1-s pre-impact window
            # 2. The other actor was seen in dynamic_objects for >=1 tick pre-impact
            speed_ok = (
                len(self._speed_window) > 0
                and (sum(self._speed_window) / len(self._speed_window))
                > EGO_SPEED_FAULT_THRESHOLD
            )
            seen_ok = self._dynamic_window.get(other_id, 0) >= 1

            if speed_ok and seen_ok:
                self.planner_fault_collisions += 1

    def update_tick(self, ego_speed: float, dynamic_ids: set[int]) -> None:
        """Call once per sim tick to advance the rolling windows."""
        with self._lock:
            self._speed_window.append(ego_speed)
            # Increment seen-count for currently visible actors, drop unseen ones.
            new_window: dict[int, int] = {}
            for actor_id in dynamic_ids:
                new_window[actor_id] = self._dynamic_window.get(actor_id, 0) + 1
            self._dynamic_window = new_window

    @property
    def sensor(self) -> carla.Actor:
        return self._sensor

    def destroy(self) -> None:
        if self._sensor is not None and self._sensor.is_alive:
            self._sensor.stop()
            self._sensor.destroy()
            self._sensor = None


# ---------------------------------------------------------------------------
# Spawn helpers
# ---------------------------------------------------------------------------
def _try_spawn_ego(
    world: carla.World,
    spawn_points: list,
    preferred_idx: int,
    model: str = config.VEHICLE_MODEL,
) -> Optional[carla.Actor]:
    """Try preferred_idx, then ±5 neighbours on collision."""
    bp = world.get_blueprint_library().find(model)
    if bp is None:
        raise ValueError(f"Vehicle blueprint '{model}' not found.")

    indices_to_try = [preferred_idx]
    for delta in range(1, 6):
        for sign in (+1, -1):
            idx = preferred_idx + sign * delta
            if 0 <= idx < len(spawn_points):
                indices_to_try.append(idx)

    for idx in indices_to_try:
        vehicle = world.try_spawn_actor(bp, spawn_points[idx])
        if vehicle is not None:
            if idx != preferred_idx:
                log.warning(
                    "Preferred spawn %d was blocked; used spawn %d instead.",
                    preferred_idx,
                    idx,
                )
            return vehicle

    return None


def _spawn_npcs(
    world: carla.World,
    traffic_manager: carla.TrafficManager,
    spawn_points: list,
    n: int,
    seed: int,
    rng: random.Random,
) -> list[carla.Actor]:
    """Spawn n NPC vehicles under traffic manager autopilot."""
    bp_lib = world.get_blueprint_library()
    vehicle_bps = bp_lib.filter("vehicle.*")

    # Exclude bikes/bicycles for reliability
    vehicle_bps = [
        bp for bp in vehicle_bps
        if int(bp.get_attribute("number_of_wheels")) >= 4
    ]

    available = [sp for sp in spawn_points]
    rng.shuffle(available)
    spawned: list[carla.Actor] = []

    for i, sp in enumerate(available):
        if len(spawned) >= n:
            break
        bp = rng.choice(vehicle_bps)
        if bp.has_attribute("color"):
            bp.set_attribute("color", rng.choice(bp.get_attribute("color").recommended_values))
        npc = world.try_spawn_actor(bp, sp)
        if npc is not None:
            npc.set_autopilot(True, traffic_manager.get_port())
            spawned.append(npc)

    log.info("Spawned %d / %d NPC vehicles.", len(spawned), n)
    return spawned


def _spawn_pedestrians(
    world: carla.World,
    n: int,
    rng: random.Random,
) -> list[carla.Actor]:
    """Spawn n pedestrians at random locations."""
    bp_lib = world.get_blueprint_library()
    ped_bps = bp_lib.filter("walker.pedestrian.*")
    spawned: list[carla.Actor] = []

    for _ in range(n * 3):  # try extra times in case of collision
        if len(spawned) >= n:
            break
        bp = rng.choice(list(ped_bps))
        # Get a random nav point from the world
        spawn_loc = world.get_random_location_from_navigation()
        if spawn_loc is None:
            continue
        transform = carla.Transform(spawn_loc)
        ped = world.try_spawn_actor(bp, transform)
        if ped is not None:
            spawned.append(ped)

    log.info("Spawned %d / %d pedestrians.", len(spawned), n)
    return spawned


# ---------------------------------------------------------------------------
# Red-light violation check (per tick)
# ---------------------------------------------------------------------------
def _check_red_light_violation(
    scene_state,
    ego_speed: float,
    violation_counter: list,
) -> None:
    for tl in scene_state.traffic_lights:
        if (
            tl.state == TrafficLightState.RED
            and tl.distance_m < RED_LIGHT_STOP_DIST_M
            and ego_speed > 0.1
        ):
            violation_counter[0] += 1
            break  # count at most once per tick


# ---------------------------------------------------------------------------
# Per-route evaluation
# ---------------------------------------------------------------------------
def run_route(
    client: carla.Client,
    world: carla.World,
    traffic_manager: carla.TrafficManager,
    carla_map: carla.Map,
    route: EvalRoute,
    args: argparse.Namespace,
    frame_logger: RunLogger,
    rng: random.Random,
) -> dict:
    """Run one evaluation route. Returns a metrics dict."""
    log.info("--- Route %d: %s ---", route.route_id, route.label)

    spawn_points = carla_map.get_spawn_points()

    # ------------------------------------------------------------------
    # Spawn ego
    # ------------------------------------------------------------------
    vehicle = _try_spawn_ego(world, spawn_points, route.start_spawn_idx)
    if vehicle is None:
        log.error("Could not spawn ego for route %d; skipping.", route.route_id)
        return {
            "route_id": route.route_id,
            "label": route.label,
            "completion_pct": 0.0,
            "collisions": 0,
            "planner_fault_collisions": 0,
            "red_light_violations": 0,
            "mean_speed_mps": 0.0,
            "mean_jerk": 0.0,
            "time_s": 0.0,
            "timeout": False,
            "skipped": True,
        }

    spawned_actors: list[carla.Actor] = [vehicle]

    # ------------------------------------------------------------------
    # Collision sensor
    # ------------------------------------------------------------------
    collision_tracker = CollisionTracker(world, vehicle)
    spawned_actors.append(collision_tracker.sensor)

    # ------------------------------------------------------------------
    # NPCs and pedestrians
    # ------------------------------------------------------------------
    npcs = _spawn_npcs(world, traffic_manager, spawn_points, args.npcs, args.seed, rng)
    spawned_actors.extend(npcs)

    pedestrians = _spawn_pedestrians(world, PEDESTRIAN_COUNT, rng)
    spawned_actors.extend(pedestrians)

    # Tick once to let actors settle before building route
    world.tick()

    # ------------------------------------------------------------------
    # Route computation
    # ------------------------------------------------------------------
    start_loc = spawn_points[route.start_spawn_idx].location
    end_loc = spawn_points[route.end_spawn_idx].location

    route_planner = RoutePlanner(carla_map, config.ROUTE_SAMPLING_RESOLUTION_M)
    computed_route = route_planner.compute(start_loc, end_loc)
    log.info(
        "Route %d: %d waypoints (spawn[%d] -> spawn[%d])",
        route.route_id,
        len(computed_route),
        route.start_spawn_idx,
        route.end_spawn_idx,
    )

    if len(computed_route) == 0:
        log.error("Empty route for route_id=%d; skipping.", route.route_id)
        _batch_destroy(client, spawned_actors)
        return _empty_result(route, skipped=True)

    # ------------------------------------------------------------------
    # Stack components
    # ------------------------------------------------------------------
    _wm = WaypointManager(
        computed_route,
        target_speed=config.TARGET_SPEED,
        search_window=config.WAYPOINT_SEARCH_WINDOW,
        slow_down_dist_m=config.SLOW_DOWN_DIST_M,
        slow_down_floor_speed=config.SLOW_DOWN_FLOOR_SPEED,
    )
    waypoint_manager = LaneAwareWaypointManager(_wm)

    gt_perception = GTPerception(world, vehicle)
    planner = RuleBasedPlanner(PlannerConfig(cruise_speed_mps=config.TARGET_SPEED))
    behaviour_planner = BehaviourPlanner(BehaviourConfig(cruise_speed_mps=config.TARGET_SPEED))

    lateral_pid = PIDController(
        *config.LATERAL_PID, dt=config.DT, integral_limit=config.INTEGRAL_LIMIT
    )
    longitudinal_pid = PIDController(
        *config.LONGITUDINAL_PID, dt=config.DT, integral_limit=config.INTEGRAL_LIMIT
    )

    # ------------------------------------------------------------------
    # Metric accumulators
    # ------------------------------------------------------------------
    speed_samples: list[float] = []
    jerk_samples: list[float] = []
    reason_counter: collections.Counter = collections.Counter()
    red_light_violations = [0]  # mutable container for nested function
    distance_traveled = 0.0
    prev_speed = 0.0
    prev_accel = 0.0
    frame_count = 0
    timed_out = False

    # Frame limit (if --frames supplied; else use MAX_ROUTE_TIME_S)
    max_frames = args.frames if args.frames is not None else int(MAX_ROUTE_TIME_S / config.DT)

    route_start_wall = time.time()

    # ------------------------------------------------------------------
    # Main tick loop
    # ------------------------------------------------------------------
    try:
        while True:
            world.tick()
            frame_count += 1

            ego = get_ego_state(vehicle)
            waypoint_manager.update(ego.x, ego.y)

            # Goal reached?
            if waypoint_manager.goal_distance(ego.x, ego.y) < config.GOAL_REACHED_THRESHOLD_M:
                log.info("Route %d: goal reached after %d frames.", route.route_id, frame_count)
                break

            # Timeout / frame cap?
            elapsed_wall = time.time() - route_start_wall
            if args.frames is not None and frame_count >= max_frames:
                log.info("Route %d: frame cap (%d) reached.", route.route_id, args.frames)
                break
            if args.frames is None and elapsed_wall >= MAX_ROUTE_TIME_S:
                log.warning(
                    "Route %d: timeout after %.1f s.", route.route_id, elapsed_wall
                )
                timed_out = True
                break

            # Perception
            scene_state = gt_perception.update(
                full_route=computed_route,
                current_idx=waypoint_manager.current_idx,
            )

            # Behaviour planning (FSM + ACC + lane change)
            planner_out = behaviour_planner.plan(scene_state)
            waypoint_manager.set_lane_offset(planner_out.target_lane_offset)

            # Safety override (red light, emergency brake — always runs)
            speed_profile = planner.plan(scene_state)

            # Pass speed profile back to waypoint manager so get_lookahead
            # can use the planner-adjusted target speed.
            waypoint_manager.set_speed_profile(speed_profile)

            # ------------------------------------------------------------------
            # Metric collection — per-frame
            # ------------------------------------------------------------------
            reason_counter[speed_profile.reason] += 1
            ego_speed = scene_state.ego_pose.speed_mps
            speed_samples.append(ego_speed)

            accel = (ego_speed - prev_speed) / config.DT
            if frame_count > 1:  # skip first tick (no meaningful prev_speed)
                jerk = abs(accel - prev_accel) / config.DT
                jerk_samples.append(jerk)
            prev_accel = accel
            prev_speed = ego_speed

            distance_traveled += ego_speed * config.DT

            # Red-light violation
            _check_red_light_violation(scene_state, ego_speed, red_light_violations)

            # Update collision tracker rolling windows
            dynamic_ids = {obj.id for obj in scene_state.dynamic_objects}
            collision_tracker.update_tick(ego_speed, dynamic_ids)

            # Nearest TL distance and nearest lead vehicle distance for CSV
            nearest_tl_dist = (
                scene_state.traffic_lights[0].distance_m
                if scene_state.traffic_lights
                else float("nan")
            )
            nearest_lead_dist = (
                scene_state.dynamic_objects[0].distance_m
                if scene_state.dynamic_objects
                else float("nan")
            )

            # ------------------------------------------------------------------
            # PID control (identical to main.py)
            # ------------------------------------------------------------------
            lookahead_dist = max(
                config.LOOKAHEAD_MIN_M,
                ego.speed * config.LOOKAHEAD_TIME_S,
            )
            lx, ly, target_speed, _ = waypoint_manager.get_lookahead(
                ego.x, ego.y, lookahead_dist
            )

            heading_error = normalize_angle(
                math.atan2(ly - ego.y, lx - ego.x) - ego.yaw_rad
            )
            steer = max(-1.0, min(1.0, lateral_pid.step(heading_error)))

            speed_error = target_speed - ego.speed
            throttle, brake = output_to_throttle_brake(
                longitudinal_pid.step(speed_error)
            )

            vehicle.apply_control(
                carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
            )

            # Follow ego with spectator camera
            vt = vehicle.get_transform()
            fwd = vt.get_forward_vector()
            world.get_spectator().set_transform(carla.Transform(
                carla.Location(
                    x=vt.location.x - 8.0 * fwd.x,
                    y=vt.location.y - 8.0 * fwd.y,
                    z=vt.location.z + 4.0,
                ),
                carla.Rotation(pitch=-15, yaw=vt.rotation.yaw),
            ))

            # ------------------------------------------------------------------
            # Per-frame CSV log (extended RunLogger header)
            # ------------------------------------------------------------------
            completion_pct = min(1.0, distance_traveled / route.expected_distance_m) * 100.0
            frame_logger.log(
                ego_x=ego.x,
                ego_y=ego.y,
                ego_yaw_rad=ego.yaw_rad,
                ego_speed=ego_speed,
                target_x=lx,
                target_y=ly,
                target_speed=target_speed,
                heading_error=heading_error,
                speed_error=speed_error,
                steer=steer,
                throttle=throttle,
                brake=brake,
                current_idx=waypoint_manager.current_idx,
                # Extended fields
                route_id=route.route_id,
                planner_reason=speed_profile.reason,
                nearest_tl_dist_m=nearest_tl_dist,
                nearest_lead_dist_m=nearest_lead_dist,
                completion_pct=completion_pct,
                fsm_state=planner_out.fsm_state,
                lane_offset_m=planner_out.target_lane_offset,
            )

    except Exception as exc:
        log.exception("Exception during route %d tick loop: %s", route.route_id, exc)
    finally:
        # Apply full brake before destroying
        try:
            vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
        except Exception:
            pass

        collision_tracker.destroy()
        # _batch_destroy filters with is_alive, so already-destroyed sensor is safely skipped.
        _batch_destroy(client, spawned_actors)

    # ------------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------------
    elapsed_wall = time.time() - route_start_wall
    completion_pct = min(1.0, distance_traveled / route.expected_distance_m) * 100.0

    return {
        "route_id": route.route_id,
        "label": route.label,
        "completion_pct": round(completion_pct, 2),
        "collisions": collision_tracker.total_collisions,
        "planner_fault_collisions": collision_tracker.planner_fault_collisions,
        "red_light_violations": red_light_violations[0],
        "mean_speed_mps": round(sum(speed_samples) / max(len(speed_samples), 1), 3),
        "mean_jerk": round(sum(jerk_samples) / max(len(jerk_samples), 1), 3),
        "time_s": round(elapsed_wall, 2),
        "timeout": timed_out,
        "skipped": False,
        "_reason_counter": reason_counter,
    }


# ---------------------------------------------------------------------------
# Batch actor cleanup
# ---------------------------------------------------------------------------
def _batch_destroy(client: carla.Client, actors: list[carla.Actor]) -> None:
    valid = [a for a in actors if a is not None and a.is_alive]
    if not valid:
        return
    client.apply_batch([carla.command.DestroyActor(a.id) for a in valid])


def _empty_result(route: EvalRoute, skipped: bool = False) -> dict:
    return {
        "route_id": route.route_id,
        "label": route.label,
        "completion_pct": 0.0,
        "collisions": 0,
        "planner_fault_collisions": 0,
        "red_light_violations": 0,
        "mean_speed_mps": 0.0,
        "mean_jerk": 0.0,
        "time_s": 0.0,
        "timeout": False,
        "skipped": skipped,
        "_reason_counter": collections.Counter(),
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def write_per_route_csv(out_dir: str, results: list[dict]) -> str:
    path = os.path.join(out_dir, "per_route.csv")
    fieldnames = [
        "route_id", "label", "completion_pct", "collisions",
        "planner_fault_collisions", "red_light_violations",
        "mean_speed_mps", "mean_jerk", "time_s", "timeout",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    return path


def write_summary_txt(
    out_dir: str,
    results: list[dict],
    all_reason_counters: list[collections.Counter],
    args: argparse.Namespace,
    date_str: str,
) -> str:
    path = os.path.join(out_dir, "summary.txt")

    n_routes = len(results)
    n_completed = sum(1 for r in results if not r.get("timeout") and not r.get("skipped"))
    total_collisions = sum(r["collisions"] for r in results)
    total_fault = sum(r["planner_fault_collisions"] for r in results)
    total_rl_violations = sum(r["red_light_violations"] for r in results)
    valid_results = [r for r in results if not r.get("skipped")]
    mean_speed = (
        sum(r["mean_speed_mps"] for r in valid_results) / len(valid_results)
        if valid_results else 0.0
    )
    mean_jerk = (
        sum(r["mean_jerk"] for r in valid_results) / len(valid_results)
        if valid_results else 0.0
    )

    # Aggregate reason histogram
    global_reasons: collections.Counter = collections.Counter()
    for ctr in all_reason_counters:
        global_reasons.update(ctr)
    total_ticks = sum(global_reasons.values()) or 1

    reason_keys = [
        "cruise", "speed_limit", "junction_speed_cap",
        "lead_vehicle", "yellow_stop", "yellow_continue",
        "red_light_stop", "emergency_brake", "lane_follow",
    ]

    lines = [
        "=== Stage 2 Baseline — GT Perception ===",
        f"Seed: {args.seed}   Routes: {n_routes}   Date: {date_str}",
        "",
        "Aggregate:",
        f"  Routes completed (no timeout): {n_completed}/{n_routes}",
        f"  Total collisions: {total_collisions}",
        f"  Planner-fault collisions: {total_fault}",
        f"  Red-light violations: {total_rl_violations}",
        f"  Mean speed (m/s): {mean_speed:.2f}",
        f"  Mean jerk: {mean_jerk:.2f}",
        "",
        "Reason histogram (% of ticks):",
    ]
    for key in reason_keys:
        pct = global_reasons.get(key, 0) / total_ticks * 100.0
        lines.append(f"  {key:<22} {pct:5.1f}%")

    # Any reasons seen that aren't in the canonical list
    extra = set(global_reasons.keys()) - set(reason_keys)
    for key in sorted(extra):
        pct = global_reasons[key] / total_ticks * 100.0
        lines.append(f"  {key:<22} {pct:5.1f}%  (unrecognised reason)")

    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))

    return path


# ---------------------------------------------------------------------------
# Extended RunLogger that writes extra eval columns
# ---------------------------------------------------------------------------
class EvalRunLogger(RunLogger):
    """Extends RunLogger header with eval-specific columns."""

    EXTRA_HEADER = [
        "route_id",
        "planner_reason",
        "nearest_tl_dist_m",
        "nearest_lead_dist_m",
        "completion_pct",
        "fsm_state",        # NEW
        "lane_offset_m",    # NEW
    ]

    def __init__(self, log_dir: str = "logs"):
        # Call parent __init__ which writes the base header and opens the file.
        super().__init__(log_dir=log_dir)
        # The parent already wrote the header; we need to redo it with extras.
        # Re-open in overwrite mode.
        self._file.close()
        self._file = open(self.path, "w", newline="")
        self._writer = __import__("csv").writer(self._file)
        self._writer.writerow(self.HEADER + self.EXTRA_HEADER)
        self._frame = 0
        self._t0 = time.time()

    def log(
        self,
        ego_x: float,
        ego_y: float,
        ego_yaw_rad: float,
        ego_speed: float,
        target_x: float,
        target_y: float,
        target_speed: float,
        heading_error: float,
        speed_error: float,
        steer: float,
        throttle: float,
        brake: float,
        current_idx: int,
        # Extended fields
        route_id: int = 0,
        planner_reason: str = "",
        nearest_tl_dist_m: float = float("nan"),
        nearest_lead_dist_m: float = float("nan"),
        completion_pct: float = 0.0,
        fsm_state: str = "",
        lane_offset_m: float = 0.0,
    ) -> None:
        self._frame += 1
        nan_str = lambda v: "" if math.isnan(v) else f"{v:.3f}"
        self._writer.writerow([
            self._frame,
            f"{time.time() - self._t0:.3f}",
            f"{ego_x:.3f}",
            f"{ego_y:.3f}",
            f"{ego_yaw_rad:.4f}",
            f"{ego_speed:.3f}",
            f"{target_x:.3f}",
            f"{target_y:.3f}",
            f"{target_speed:.3f}",
            f"{heading_error:.4f}",
            f"{speed_error:.3f}",
            f"{steer:.4f}",
            f"{throttle:.4f}",
            f"{brake:.4f}",
            current_idx,
            # Extended
            route_id,
            planner_reason,
            nan_str(nearest_tl_dist_m),
            nan_str(nearest_lead_dist_m),
            f"{completion_pct:.2f}",
            fsm_state,
            f"{lane_offset_m:.4f}",
        ])
        self._file.flush()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    # Resolve which routes to run
    if args.routes.strip().lower() == "all":
        selected_routes = EVAL_ROUTES
    else:
        wanted_ids = {int(x.strip()) for x in args.routes.split(",")}
        selected_routes = [r for r in EVAL_ROUTES if r.route_id in wanted_ids]
        if not selected_routes:
            log.error("No matching routes found for --routes '%s'.", args.routes)
            sys.exit(1)

    # Seeded RNG for reproducibility
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    # Output directory
    out_dir = make_output_dir()
    log.info("Results will be written to: %s", os.path.abspath(out_dir))

    # Per-frame CSV logger (written into logs/ as per project convention)
    frame_logger = EvalRunLogger(log_dir="logs")
    log.info("Per-frame log: %s", frame_logger.path)

    # CARLA client
    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)

    log.info("Loading Town01...")
    world = client.load_world("Town01")

    # Synchronous mode
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = config.DT
    world.apply_settings(settings)

    # Traffic Manager
    traffic_manager = client.get_trafficmanager(args.tm_port)
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.set_random_device_seed(args.seed)

    carla_map = world.get_map()

    all_results: list[dict] = []
    all_reason_counters: list[collections.Counter] = []

    try:
        for route in selected_routes:
            try:
                result = run_route(
                    client=client,
                    world=world,
                    traffic_manager=traffic_manager,
                    carla_map=carla_map,
                    route=route,
                    args=args,
                    frame_logger=frame_logger,
                    rng=rng,
                )
            except Exception as exc:
                log.exception(
                    "Unhandled exception on route %d: %s", route.route_id, exc
                )
                result = _empty_result(route, skipped=True)

            all_results.append(result)
            all_reason_counters.append(result.pop("_reason_counter", collections.Counter()))

            log.info(
                "Route %d done | completion=%.1f%% | collisions=%d | "
                "red_light_violations=%d | timeout=%s",
                result["route_id"],
                result["completion_pct"],
                result["collisions"],
                result["red_light_violations"],
                result["timeout"],
            )

            # Brief pause between routes to let CARLA settle
            world.tick()
            world.tick()

    finally:
        # Restore async mode
        settings = world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)

        frame_logger.close()

    # Write outputs
    date_str = time.strftime("%Y-%m-%d")
    csv_path = write_per_route_csv(out_dir, all_results)
    txt_path = write_summary_txt(out_dir, all_results, all_reason_counters, args, date_str)

    log.info("per_route.csv written: %s", os.path.abspath(csv_path))
    log.info("summary.txt written:   %s", os.path.abspath(txt_path))

    # Print summary to stdout
    with open(txt_path) as f:
        print(f.read())


if __name__ == "__main__":
    main()
