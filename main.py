import math

from utils import carla_bootstrap  # noqa: F401  (sys.path side effect for `agents.*`)

import carla

import config
from control.pid_controller import PIDController
from planning.route_planner import RoutePlanner
from planning.waypoint_manager import WaypointManager
from utils.carla_utils import (
    draw_next_waypoints,
    get_ego_state,
    normalize_angle,
    output_to_throttle_brake,
    synchronous_mode,
)
from utils.run_logger import RunLogger
from vehicle.autonomous_vehicle import AutonomousVehicle
from perception.gt_perception import GTPerception
from planning.planner import RuleBasedPlanner, PlannerConfig


def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)

    av = AutonomousVehicle(client)
    world, carla_map = av.spawn_world(config.TOWN)
    vehicle = av.spawn_vehicle(config.VEHICLE_MODEL)
    av.equip_sensors()

    spawn_points = carla_map.get_spawn_points()
    start_loc = spawn_points[config.START_SPAWN_IDX].location
    end_loc = spawn_points[config.END_SPAWN_IDX].location

    route_planner = RoutePlanner(carla_map, config.ROUTE_SAMPLING_RESOLUTION_M)
    route = route_planner.compute(start_loc, end_loc)
    print(
        f"Route: {len(route)} waypoints "
        f"(spawn[{config.START_SPAWN_IDX}] -> spawn[{config.END_SPAWN_IDX}])"
    )

    waypoint_manager = WaypointManager(
        route,
        target_speed=config.TARGET_SPEED,
        search_window=config.WAYPOINT_SEARCH_WINDOW,
        slow_down_dist_m=config.SLOW_DOWN_DIST_M,
        slow_down_floor_speed=config.SLOW_DOWN_FLOOR_SPEED,
    )

    gt_perception = GTPerception(world, vehicle)
    planner = RuleBasedPlanner(PlannerConfig(cruise_speed_mps=config.TARGET_SPEED))

    lateral_pid = PIDController(
        *config.LATERAL_PID, dt=config.DT, integral_limit=config.INTEGRAL_LIMIT
    )
    longitudinal_pid = PIDController(
        *config.LONGITUDINAL_PID, dt=config.DT, integral_limit=config.INTEGRAL_LIMIT
    )

    spectator = world.get_spectator()

    logger = RunLogger()
    print(f"Logging to {logger.path}")

    try:
        with synchronous_mode(world, fixed_dt=config.DT):
            while True:
                world.tick()

                ego = get_ego_state(vehicle)
                waypoint_manager.update(ego.x, ego.y)
                scene_state = gt_perception.update(full_route=route, current_idx=waypoint_manager.current_idx)
                speed_profile = planner.plan(scene_state)
                waypoint_manager.set_speed_profile(speed_profile)

                if waypoint_manager.goal_distance(ego.x, ego.y) < config.GOAL_REACHED_THRESHOLD_M:
                    print("Goal reached.")
                    break

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

                # --- Status print every 1 s (20 ticks at 20 Hz) ---------------
                nearest_tl = min(scene_state.traffic_lights, key=lambda t: t.distance_m, default=None)
                nearest_obj = min(scene_state.dynamic_objects, key=lambda o: o.distance_m, default=None)
                if logger._frame % 20 == 0:
                    tl_str = (
                        f"TL {nearest_tl.state.value.upper()} {nearest_tl.distance_m:.1f}m"
                        if nearest_tl else "TL none"
                    )
                    obj_str = (
                        f"OBJ {nearest_obj.object_class} {nearest_obj.distance_m:.1f}m"
                        if nearest_obj else "OBJ none"
                    )
                    print(
                        f"[{logger._frame:5d}] {speed_profile.reason:<22s} "
                        f"spd={ego.speed*3.6:5.1f}km/h tgt={target_speed*3.6:5.1f}km/h "
                        f"wp={waypoint_manager.current_idx:4d}  {tl_str}  {obj_str}"
                    )

                draw_next_waypoints(
                    world, route, waypoint_manager.current_idx,
                    n=20, life_time=config.DT * 2,
                )
                logger.log(
                    ego_x=ego.x, ego_y=ego.y,
                    ego_yaw_rad=ego.yaw_rad, ego_speed=ego.speed,
                    target_x=lx, target_y=ly, target_speed=target_speed,
                    heading_error=heading_error, speed_error=speed_error,
                    steer=steer, throttle=throttle, brake=brake,
                    current_idx=waypoint_manager.current_idx,
                    planner_reason=speed_profile.reason,
                    nearest_tl_state=nearest_tl.state.value if nearest_tl else "",
                    nearest_tl_dist_m=nearest_tl.distance_m if nearest_tl else -1.0,
                    nearest_obj_dist_m=nearest_obj.distance_m if nearest_obj else -1.0,
                )

                vt = vehicle.get_transform()
                fwd = vt.get_forward_vector()
                spectator.set_transform(
                    carla.Transform(
                        carla.Location(
                            x=vt.location.x - 8.0 * fwd.x,
                            y=vt.location.y - 8.0 * fwd.y,
                            z=vt.location.z + 4.0,
                        ),
                        carla.Rotation(pitch=-15, yaw=vt.rotation.yaw),
                    )
                )

            vehicle.apply_control(
                carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)
            )
    finally:
        logger.close()
        for sensor in av.sensors.values():
            if sensor is not None and sensor.is_alive:
                sensor.destroy()
        if vehicle is not None and vehicle.is_alive:
            vehicle.destroy()


if __name__ == "__main__":
    main()
