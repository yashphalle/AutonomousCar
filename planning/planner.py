"""
RuleBasedPlanner — reads SceneState, outputs SpeedProfile.
No carla imports. All units: m/s, metres, radians.

Priority order (highest wins):
  0. Emergency brake   — object within emergency_brake_radius_m
  1. Red light stop    — ramp to 0 before stop line
  2. Yellow light      — stop if safely stoppable, else continue
  3. Lead vehicle      — follow closest same-heading vehicle ahead
  4. Junction cap      — cap speed near unsignalized junctions
  5. Speed limit       — per-waypoint cap from WaypointInfo
  6. Cruise            — target_speed from PlannerConfig
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from perception.scene_state import SceneState, TrafficLightState


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------

@dataclass
class SpeedProfile:
    target_speeds_mps: list[float]   # parallel to SceneState.route_waypoints
    reason: str                       # primary rule that fired this tick


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PlannerConfig:
    cruise_speed_mps: float = 8.0
    junction_speed_mps: float = 5.0
    safe_following_distance_m: float = 10.0
    comfortable_following_distance_m: float = 12.0
    min_decel_distance_m: float = 5.0
    max_decel_mps2: float = 3.0
    emergency_brake_radius_m: float = 5.0
    lead_vehicle_heading_tolerance_rad: float = 0.349   # ±20° (40° cone)
    lead_vehicle_search_distance_m: float = 10.0
    stop_line_buffer_m: float = 4.0  # stop this far before the TL pole


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_angle(a: float) -> float:
    """Wrap angle to [-pi, pi]."""
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class RuleBasedPlanner:
    def __init__(self, config: PlannerConfig = None):
        self.config = config or PlannerConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(self, state: SceneState) -> SpeedProfile:
        cfg = self.config
        n = len(state.route_waypoints)

        # With no waypoints ahead, hold cruise (no-op profile).
        if n == 0:
            return SpeedProfile(target_speeds_mps=[], reason="cruise")

        ego_speed = state.ego_pose.speed_mps
        ego_yaw = state.ego_pose.yaw_rad

        # --- Build base profile starting from cruise (lowest priority) -------
        speeds = [cfg.cruise_speed_mps] * n
        reason = "cruise"

        # Priority 6 — speed limit (applied per waypoint, overrides cruise)
        for i, wp in enumerate(state.route_waypoints):
            if wp.speed_limit_mps > 0.0:
                speeds[i] = min(speeds[i], wp.speed_limit_mps)
        # If any waypoint was capped by speed limit and it differs from pure cruise,
        # mark reason — but only as default; higher-priority rules will overwrite.
        if any(
            wp.speed_limit_mps > 0.0 and wp.speed_limit_mps < cfg.cruise_speed_mps
            for wp in state.route_waypoints
        ):
            reason = "speed_limit"

        # Priority 5 — junction speed cap (per waypoint, min with existing)
        junction_fired = False
        for i, wp in enumerate(state.route_waypoints):
            if wp.is_junction and wp.distance_from_ego_m <= 15.0:
                junction_fired = True
            if junction_fired:
                speeds[i] = min(speeds[i], cfg.junction_speed_mps)
        if junction_fired:
            reason = "junction_speed_cap"

        # Priority 4 — lead vehicle following
        lead = self._find_lead_vehicle(state, ego_yaw)
        if lead is not None:
            gap = lead.distance_m
            lead_speed = lead.speed_mps
            if gap < cfg.safe_following_distance_m:
                follow_speed = max(0.0, lead_speed - 1.0)
                ramp_n = min(10, n)
                ramp = self._decel_ramp(ramp_n, ego_speed, follow_speed)
                for i in range(ramp_n):
                    speeds[i] = min(speeds[i], ramp[i])
                for i in range(ramp_n, n):
                    speeds[i] = min(speeds[i], follow_speed)
                reason = "lead_vehicle"
            elif gap <= cfg.comfortable_following_distance_m:
                # Hold current speed — clamp but don't accelerate past what's there
                for i in range(n):
                    speeds[i] = min(speeds[i], ego_speed)
                # reason stays as whatever it was (not overriding higher priorities)

        # Priority 2 — yellow light
        yellow_lights = [
            tl for tl in state.traffic_lights
            if tl.state == TrafficLightState.YELLOW and tl.distance_m > 0
        ]
        if yellow_lights:
            nearest_y = min(yellow_lights, key=lambda tl: tl.distance_m)
            d_stop = (ego_speed ** 2) / (2.0 * cfg.max_decel_mps2) if cfg.max_decel_mps2 > 0 else float("inf")
            if d_stop < nearest_y.distance_m:
                # Safely stoppable — physics-based ramp to 0
                stop_dist_y = max(cfg.min_decel_distance_m,
                                  nearest_y.distance_m - cfg.stop_line_buffer_m)
                for i, wp in enumerate(state.route_waypoints):
                    d_rem = max(0.0, stop_dist_y - wp.distance_from_ego_m)
                    v_max_y = math.sqrt(2.0 * cfg.max_decel_mps2 * d_rem) if d_rem > 0 else 0.0
                    speeds[i] = min(speeds[i], v_max_y)
                reason = "yellow_stop"
            else:
                # Not safely stoppable — continue (don't override speeds)
                reason = "yellow_continue"

        # Priority 1 — red light stop
        red_lights = [
            tl for tl in state.traffic_lights
            if tl.state == TrafficLightState.RED and tl.distance_m > 0
        ]
        if red_lights:
            nearest_r = min(red_lights, key=lambda tl: tl.distance_m)
            stop_dist = max(cfg.min_decel_distance_m,
                            nearest_r.distance_m - cfg.stop_line_buffer_m)
            for i, wp in enumerate(state.route_waypoints):
                d_remaining = max(0.0, stop_dist - wp.distance_from_ego_m)
                v_max = math.sqrt(2.0 * cfg.max_decel_mps2 * d_remaining) if d_remaining > 0 else 0.0
                speeds[i] = min(speeds[i], v_max)
            reason = "red_light_stop"

        # Priority 0 — emergency brake (forward cone only, ignores beside/behind)
        for obj in state.dynamic_objects:
            if obj.distance_m <= cfg.emergency_brake_radius_m:
                bearing = math.atan2(obj.y - state.ego_pose.y, obj.x - state.ego_pose.x)
                heading_diff = abs(_normalize_angle(bearing - state.ego_pose.yaw_rad))
                if heading_diff <= cfg.lead_vehicle_heading_tolerance_rad:
                    print(f"[emergency_brake] id={obj.id} dist={obj.distance_m:.1f}m angle={math.degrees(heading_diff):.1f}°")
                    return SpeedProfile(
                        target_speeds_mps=[0.0] * n,
                        reason="emergency_brake",
                    )

        return SpeedProfile(target_speeds_mps=speeds, reason=reason)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_profile(self, n: int, speed: float, reason: str) -> SpeedProfile:
        """Return a uniform SpeedProfile with all n waypoints at `speed`."""
        return SpeedProfile(target_speeds_mps=[speed] * n, reason=reason)

    def _decel_ramp(self, n: int, from_speed: float, to_speed: float) -> list[float]:
        """Linearly interpolate n speeds from from_speed to to_speed, clamped >= 0."""
        if n <= 1:
            return [max(0.0, to_speed)]
        result = []
        for i in range(n):
            t = i / (n - 1)
            s = from_speed + t * (to_speed - from_speed)
            result.append(max(0.0, s))
        return result

    def _find_lead_vehicle(self, state: SceneState, ego_yaw: float):
        """
        Return the nearest DetectedObject that is:
        - object_class == "vehicle"
        - within lead_vehicle_search_distance_m
        - heading (bearing from ego) within lead_vehicle_heading_tolerance_rad of ego_yaw
        Returns None if no such vehicle found.
        """
        cfg = self.config
        ego_x = state.ego_pose.x
        ego_y = state.ego_pose.y
        best = None
        best_dist = float("inf")

        for obj in state.dynamic_objects:
            if obj.object_class != "vehicle":
                continue
            if obj.distance_m > cfg.lead_vehicle_search_distance_m:
                continue
            bearing = math.atan2(obj.y - ego_y, obj.x - ego_x)
            heading_diff = abs(_normalize_angle(bearing - ego_yaw))
            if heading_diff <= cfg.lead_vehicle_heading_tolerance_rad:
                if obj.distance_m < best_dist:
                    best_dist = obj.distance_m
                    best = obj

        return best
