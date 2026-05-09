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

        return SceneState(
            timestamp_s=snapshot.timestamp.elapsed_seconds,
            ego_pose=ego_pose,
            route_waypoints=route_waypoints,
            route_start_idx=current_idx,
            traffic_lights=traffic_lights,
            stop_signs=stop_signs,
            dynamic_objects=dynamic_objects,
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
