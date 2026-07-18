"""
GTPerception — reads CARLA ground truth and produces SceneState each tick.

This is the only file that imports carla and bridges it to the SceneState
contract. Planner and everything downstream see only SceneState.
All speeds in m/s, distances in metres, angles in radians.
"""

from __future__ import annotations

import math

import carla

from perception.scene_state import (
    DetectedObject,
    EgoPose,
    OccupancyGrid,
    SceneState,
    StopSignInfo,
    TrafficLightInfo,
    TrafficLightState,
    WaypointInfo,
)

_CARLA_TL_STATE = {
    carla.TrafficLightState.Red:     TrafficLightState.RED,
    carla.TrafficLightState.Yellow:  TrafficLightState.YELLOW,
    carla.TrafficLightState.Green:   TrafficLightState.GREEN,
    carla.TrafficLightState.Off:     TrafficLightState.OFF,
    carla.TrafficLightState.Unknown: TrafficLightState.UNKNOWN,
}

_TL_LOOK_AHEAD_M = 60.0
_OBJECT_RADIUS_M = 60.0
_STOP_SIGN_RADIUS_M = 60.0
_ROUTE_WINDOW = 50   # number of waypoints to include (~50 m at 1 m spacing)
_LEAD_VEHICLE_SEARCH_M = 40.0

# Phase 4 / Phase 5 constants
_PED_CONE_HALF_ANGLE_COS = 0.9397  # cos(20°) — pedestrian forward cone half-angle
_PED_SEARCH_M = 20.0
_CROSSWALK_SEARCH_M = 30.0
_STOP_SIGN_AHEAD_M = 30.0

# Phase 3 — lane change clear-zone constants
_LC_LOOK_AHEAD_M  = 40.0   # clear zone ahead for lane change
_LC_LOOK_BEHIND_M =  5.0   # clear zone behind for lane change


class GTPerception:
    def __init__(self, world, vehicle):
        self._world = world
        self._ego = vehicle
        self._map = world.get_map()

    def update(self, full_route: list, current_idx: int) -> SceneState:
        snapshot = self._world.get_snapshot()

        t = self._ego.get_transform()
        v = self._ego.get_velocity()
        ego_x, ego_y, ego_z = t.location.x, t.location.y, t.location.z
        yaw_rad = math.radians(t.rotation.yaw)
        speed_mps = math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)

        ego_pose = EgoPose(
            x=ego_x,
            y=ego_y,
            z=ego_z,
            yaw_rad=yaw_rad,
            speed_mps=speed_mps,
        )

        route_waypoints = self._build_route_waypoints(
            full_route, current_idx, ego_x, ego_y
        )
        traffic_lights = self._build_traffic_lights(ego_x, ego_y, full_route, current_idx)
        stop_signs = self._build_stop_signs(ego_x, ego_y)
        dynamic_objects = self._build_dynamic_objects(ego_x, ego_y)

        # Lead vehicle detection (Phase 2)
        lead_actor, lead_dist = self._find_lead_vehicle(ego_x, ego_y, yaw_rad)
        if lead_actor is not None:
            lead_vel = lead_actor.get_velocity()
            lead_speed_mps = math.sqrt(lead_vel.x**2 + lead_vel.y**2 + lead_vel.z**2)
            lead_id = lead_actor.id
            print(f"[gt_perception] lead_vehicle id={lead_id} dist={lead_dist:.1f}m speed={lead_speed_mps*3.6:.1f}km/h")
        else:
            lead_speed_mps = -1.0
            lead_id = None
            lead_dist = -1.0

        # Phase 4 — pedestrian + crosswalk
        ped_in_path = self._pedestrian_in_path(ego_x, ego_y, yaw_rad)
        crosswalk_ahead, crosswalk_dist = self._find_nearest_crosswalk_ahead(ego_x, ego_y, yaw_rad)

        # Phase 5 — stop sign ahead (uses already-built stop_signs list)
        ss_ahead, ss_dist = self._stop_sign_ahead(ego_x, ego_y, yaw_rad, stop_signs)

        # Phase 3 — adjacent lane clearance for lane change
        left_avail, left_clear, right_avail, right_clear = self._check_adjacent_lanes(
            ego_x, ego_y, yaw_rad
        )

        return SceneState(
            timestamp_s=snapshot.timestamp.elapsed_seconds,
            ego_pose=ego_pose,
            route_waypoints=route_waypoints,
            route_start_idx=current_idx,
            traffic_lights=traffic_lights,
            stop_signs=stop_signs,
            dynamic_objects=dynamic_objects,
            lead_vehicle_id=lead_id,
            lead_vehicle_distance_m=lead_dist,
            lead_vehicle_speed_mps=lead_speed_mps,
            pedestrian_in_path=ped_in_path,
            crosswalk_ahead=crosswalk_ahead,
            crosswalk_distance_m=crosswalk_dist,
            stop_sign_ahead=ss_ahead,
            stop_sign_distance_m=ss_dist,
            left_lane_available=left_avail,
            left_lane_clear=left_clear,
            right_lane_available=right_avail,
            right_lane_clear=right_clear,
            local_map=None,
        )

    # ------------------------------------------------------------------
    def _build_route_waypoints(
        self,
        full_route: list,
        current_idx: int,
        ego_x: float,
        ego_y: float,
    ) -> list[WaypointInfo]:
        end_idx = min(current_idx + _ROUTE_WINDOW, len(full_route))
        results: list[WaypointInfo] = []

        for i in range(current_idx, end_idx):
            carla_wp, _ = full_route[i]
            loc = carla_wp.transform.location
            wp_x, wp_y = loc.x, loc.y
            yaw_rad = math.radians(carla_wp.transform.rotation.yaw)

            # Speed limit: try waypoint attribute first, fall back to ego
            try:
                speed_limit_mps = carla_wp.get_speed_limit() / 3.6
            except Exception:
                try:
                    if hasattr(self._ego, "get_speed_limit"):
                        speed_limit_mps = self._ego.get_speed_limit() / 3.6
                    else:
                        speed_limit_mps = 0.0
                except Exception:
                    speed_limit_mps = 0.0

            results.append(
                WaypointInfo(
                    x=wp_x,
                    y=wp_y,
                    yaw_rad=yaw_rad,
                    speed_limit_mps=speed_limit_mps,
                    is_junction=carla_wp.is_junction,
                    distance_from_ego_m=math.hypot(wp_x - ego_x, wp_y - ego_y),
                )
            )

        return results

    def _build_traffic_lights(
        self, ego_x: float, ego_y: float, full_route: list, current_idx: int
    ) -> list[TrafficLightInfo]:
        results: list[TrafficLightInfo] = []
        actors = self._world.get_actors().filter("traffic.traffic_light")

        # Build a set of (road_id, lane_id) pairs from upcoming route waypoints
        # covering the next ~65 m so the lane filter matches approach roads,
        # not just the ego's exact current waypoint.
        upcoming_lanes: set[tuple[int, int]] = set()
        for carla_wp, _ in full_route[current_idx: current_idx + 70]:
            upcoming_lanes.add((carla_wp.road_id, carla_wp.lane_id))

        for tl in actors:
            loc = tl.get_location()
            dist = math.hypot(ego_x - loc.x, ego_y - loc.y)
            if dist > _TL_LOOK_AHEAD_M:
                continue

            # Lane filter: light must affect a lane the car will travel through
            relevant = any(
                (awp.road_id, awp.lane_id) in upcoming_lanes
                for awp in tl.get_affected_lane_waypoints()
            )
            if not relevant:
                continue

            results.append(
                TrafficLightInfo(
                    id=tl.id,
                    state=_CARLA_TL_STATE.get(tl.get_state(), TrafficLightState.UNKNOWN),
                    distance_m=dist,
                    red_duration_s=tl.get_red_time(),
                    yellow_duration_s=tl.get_yellow_time(),
                    green_duration_s=tl.get_green_time(),
                )
            )

        results.sort(key=lambda x: x.distance_m)
        return results

    def _build_stop_signs(self, ego_x: float, ego_y: float) -> list[StopSignInfo]:
        results: list[StopSignInfo] = []
        for sign in self._world.get_actors().filter("traffic.stop"):
            loc = sign.get_location()
            dist = math.hypot(ego_x - loc.x, ego_y - loc.y)
            if dist <= _STOP_SIGN_RADIUS_M:
                results.append(
                    StopSignInfo(
                        id=sign.id,
                        distance_m=dist,
                        location_x=loc.x,
                        location_y=loc.y,
                    )
                )
        results.sort(key=lambda x: x.distance_m)
        return results

    def _find_lead_vehicle(
        self, ego_x: float, ego_y: float, ego_yaw_rad: float
    ) -> tuple:
        """
        Find the closest vehicle that is:
          1. In the same lane as ego (road_id + lane_id exact match)
          2. Ahead of ego (dot product with forward vector > 0)
          3. Within _LEAD_VEHICLE_SEARCH_M

        Returns (actor, distance_m) or (None, -1.0).

        CARLA coordinate notes:
          - Left-handed Z-up. X=forward, Y=right, Z=up.
          - Forward vector: (cos(yaw_rad), sin(yaw_rad)) — no sign flip.
          - Yaw clockwise: yaw=0→+X, yaw=90°→+Y(right).
          - lane_id sign encodes direction: same value = same lane + same direction.
        """
        ego_loc = carla.Location(x=ego_x, y=ego_y)
        try:
            ego_wp = self._map.get_waypoint(
                ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving
            )
        except Exception:
            return None, -1.0
        if ego_wp is None:
            return None, -1.0

        ego_road_id = ego_wp.road_id
        ego_lane_id = ego_wp.lane_id

        # Forward vector in CARLA left-handed space — no Y negation
        fwd_x = math.cos(ego_yaw_rad)
        fwd_y = math.sin(ego_yaw_rad)

        best_actor = None
        best_dist = float("inf")

        for actor in self._world.get_actors().filter("vehicle.*"):
            if actor.id == self._ego.id:
                continue

            loc = actor.get_location()
            dist = math.hypot(ego_x - loc.x, ego_y - loc.y)
            if dist > _LEAD_VEHICLE_SEARCH_M:
                continue

            # Lane membership: same road + same lane_id (sign encodes direction)
            try:
                actor_wp = self._map.get_waypoint(
                    loc, project_to_road=True, lane_type=carla.LaneType.Driving
                )
            except Exception:
                continue
            if actor_wp is None:
                continue
            if actor_wp.road_id != ego_road_id or actor_wp.lane_id != ego_lane_id:
                continue

            # Ahead check via dot product in CARLA space (no Y negation)
            dx = loc.x - ego_x
            dy = loc.y - ego_y
            dot = dx * fwd_x + dy * fwd_y
            if dot <= 0.0:
                continue  # behind or at same position

            if dist < best_dist:
                best_dist = dist
                best_actor = actor

        if best_actor is None:
            return None, -1.0
        return best_actor, best_dist

    def _pedestrian_in_path(
        self, ego_x: float, ego_y: float, ego_yaw_rad: float
    ) -> bool:
        """
        Phase 4: Return True if any pedestrian is within _PED_SEARCH_M and inside
        a ±30° forward cone.

        CARLA left-handed Z-up: forward = (cos(yaw), sin(yaw)), no Y negation.
        """
        fwd_x = math.cos(ego_yaw_rad)
        fwd_y = math.sin(ego_yaw_rad)

        for actor in self._world.get_actors().filter("walker.pedestrian.*"):
            loc = actor.get_location()
            dx = loc.x - ego_x
            dy = loc.y - ego_y
            dist = math.hypot(dx, dy)
            if dist > _PED_SEARCH_M or dist < 1e-3:
                continue
            dot = dx * fwd_x + dy * fwd_y
            # Ahead AND within cone: dot/dist > cos(30°)
            if dot > 0.0 and (dot / dist) > _PED_CONE_HALF_ANGLE_COS:
                print(f"[gt_perception] pedestrian_in_path id={actor.id} dist={dist:.1f}m")
                return True
        return False

    def _find_nearest_crosswalk_ahead(
        self, ego_x: float, ego_y: float, ego_yaw_rad: float
    ) -> tuple[bool, float]:
        """
        Phase 4: Scan crosswalk polygon vertices from map.get_crosswalks().
        Returns (True, min_dist_m) if any vertex is ahead and within _CROSSWALK_SEARCH_M,
        else (False, -1.0).

        CARLA left-handed Z-up: forward = (cos(yaw), sin(yaw)), no Y negation.
        """
        fwd_x = math.cos(ego_yaw_rad)
        fwd_y = math.sin(ego_yaw_rad)

        try:
            vertices = self._map.get_crosswalks()
        except Exception:
            return False, -1.0

        min_dist = float("inf")
        found = False

        for loc in vertices:
            dx = loc.x - ego_x
            dy = loc.y - ego_y
            dist = math.hypot(dx, dy)
            if dist > _CROSSWALK_SEARCH_M or dist < 1e-3:
                continue
            dot = dx * fwd_x + dy * fwd_y
            if dot > 0.0 and dist < min_dist:
                min_dist = dist
                found = True

        if found:
            print(f"[gt_perception] crosswalk_ahead dist={min_dist:.1f}m")
            return True, min_dist
        return False, -1.0

    def _stop_sign_ahead(
        self,
        ego_x: float,
        ego_y: float,
        ego_yaw_rad: float,
        stop_signs: list[StopSignInfo],
    ) -> tuple[bool, float]:
        """
        Phase 5: From the already-built stop_signs list, find the nearest sign
        that is ahead of ego and within _STOP_SIGN_AHEAD_M.

        CARLA left-handed Z-up: forward = (cos(yaw), sin(yaw)), no Y negation.
        """
        fwd_x = math.cos(ego_yaw_rad)
        fwd_y = math.sin(ego_yaw_rad)

        best_dist = float("inf")
        found = False

        for sign in stop_signs:
            if sign.distance_m > _STOP_SIGN_AHEAD_M:
                continue
            dx = sign.location_x - ego_x
            dy = sign.location_y - ego_y
            dot = dx * fwd_x + dy * fwd_y
            if dot > 0.0 and sign.distance_m < best_dist:
                best_dist = sign.distance_m
                found = True

        if found:
            return True, best_dist
        return False, -1.0

    def _check_adjacent_lanes(
        self, ego_x: float, ego_y: float, ego_yaw_rad: float
    ) -> tuple[bool, bool, bool, bool]:
        """
        Phase 3: check left and right adjacent lanes for availability and clearance.
        Returns (left_available, left_clear, right_available, right_clear).

        'available' = a drivable same-direction lane exists AND map permits the lane change.
        'clear'     = no vehicles within _LC_LOOK_AHEAD_M ahead or _LC_LOOK_BEHIND_M behind.
        """
        try:
            ego_wp = self._map.get_waypoint(
                carla.Location(x=ego_x, y=ego_y),
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
        except Exception:
            return False, False, False, False
        if ego_wp is None:
            return False, False, False, False

        lc_flag = ego_wp.lane_change
        left_permitted  = lc_flag in (carla.LaneChange.Left,  carla.LaneChange.Both)
        right_permitted = lc_flag in (carla.LaneChange.Right, carla.LaneChange.Both)

        left_avail, left_clear = (
            self._check_one_lane(ego_wp, ego_wp.get_left_lane(), ego_x, ego_y, ego_yaw_rad)
            if left_permitted else (False, False)
        )
        right_avail, right_clear = (
            self._check_one_lane(ego_wp, ego_wp.get_right_lane(), ego_x, ego_y, ego_yaw_rad)
            if right_permitted else (False, False)
        )

        if left_avail or right_avail:
            print(
                f"[gt_perception] adj_lanes  left={'avail' if left_avail else 'none'}"
                f"{'(clear)' if left_clear else '(blocked)' if left_avail else ''}"
                f"  right={'avail' if right_avail else 'none'}"
                f"{'(clear)' if right_clear else '(blocked)' if right_avail else ''}"
            )

        return left_avail, left_clear, right_avail, right_clear

    def _check_one_lane(
        self,
        ego_wp,
        adj_wp,
        ego_x: float,
        ego_y: float,
        ego_yaw_rad: float,
    ) -> tuple[bool, bool]:
        """
        Returns (available, clear) for one adjacent lane.

        Guards (all must pass for 'available'):
          1. adj_wp exists
          2. lane_type == Driving
          3. Same direction as ego (lane_id sign matches — opposite sign = oncoming traffic)
          4. Lane change already confirmed permitted by caller

        'clear' = no vehicles in that lane within _LC_LOOK_AHEAD_M ahead
                  or _LC_LOOK_BEHIND_M behind ego.

        CARLA coordinate system: left-handed Z-up.
        Forward vector: (cos(yaw_rad), sin(yaw_rad)) — NO Y negation.
        dot > 0 means ahead, dot < 0 means behind.
        """
        if adj_wp is None:
            return False, False
        if adj_wp.lane_type != carla.LaneType.Driving:
            return False, False
        # Same sign of lane_id = same traffic direction. Opposite sign = oncoming.
        if adj_wp.lane_id * ego_wp.lane_id < 0:
            return False, False

        adj_road_id = adj_wp.road_id
        adj_lane_id = adj_wp.lane_id
        fwd_x = math.cos(ego_yaw_rad)
        fwd_y = math.sin(ego_yaw_rad)

        for actor in self._world.get_actors().filter("vehicle.*"):
            if actor.id == self._ego.id:
                continue
            loc = actor.get_location()
            dist = math.hypot(ego_x - loc.x, ego_y - loc.y)
            if dist > _LC_LOOK_AHEAD_M + _LC_LOOK_BEHIND_M:
                continue
            try:
                actor_wp = self._map.get_waypoint(
                    loc, project_to_road=True, lane_type=carla.LaneType.Driving
                )
            except Exception:
                continue
            if actor_wp is None:
                continue
            if actor_wp.road_id != adj_road_id or actor_wp.lane_id != adj_lane_id:
                continue
            # Vehicle is in adjacent lane — check longitudinal window
            dx = loc.x - ego_x
            dy = loc.y - ego_y
            dot = dx * fwd_x + dy * fwd_y
            ahead  = dot > 0 and dot < _LC_LOOK_AHEAD_M
            behind = dot <= 0 and abs(dot) < _LC_LOOK_BEHIND_M
            if ahead or behind:
                return True, False   # lane exists but is occupied

        return True, True   # lane exists and is clear

    def _build_dynamic_objects(
        self, ego_x: float, ego_y: float
    ) -> list[DetectedObject]:
        results: list[DetectedObject] = []
        ego_id = self._ego.id

        for filt, label in [
            ("vehicle.*", "vehicle"),
            ("walker.pedestrian.*", "pedestrian"),
        ]:
            for actor in self._world.get_actors().filter(filt):
                if actor.id == ego_id:
                    continue
                loc = actor.get_location()
                dist = math.hypot(ego_x - loc.x, ego_y - loc.y)
                if dist > _OBJECT_RADIUS_M:
                    continue
                vel = actor.get_velocity()
                speed_mps = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
                yaw_rad = math.radians(actor.get_transform().rotation.yaw)
                results.append(
                    DetectedObject(
                        id=actor.id,
                        object_class=label,
                        x=loc.x,
                        y=loc.y,
                        yaw_rad=yaw_rad,
                        vx_mps=vel.x,
                        vy_mps=vel.y,
                        speed_mps=speed_mps,
                        distance_m=dist,
                    )
                )

        results.sort(key=lambda x: x.distance_m)
        return results
