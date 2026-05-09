"""
Spawns an ego vehicle and dumps every queryable piece of CARLA state at each
log tick — ego pose + control + dynamics, current waypoint context, nearby
traffic lights with full attributes including timing, stop signs, nearby
vehicles and pedestrians, and weather — to a JSONL log file with periodic
readable summaries on stdout.

The ego can be driven three ways:
  --mode teleop    pygame keyboard control (default)
  --mode pid       our Stage 1 PID stack drives a route between spawn points,
                   picking a new random destination each time it arrives
  --mode autopilot CARLA's Traffic Manager (known to be flaky in async mode
                   with many actors; kept for completeness with a warning)

Usage:
  Terminal 1: ./CarlaUE4.sh
  Terminal 2 (optional, road overlay): python extras/map_overlay.py
  Terminal 3: python extras/explore_carla_state.py
              python extras/explore_carla_state.py --mode pid --pid-target-speed 8
              python extras/explore_carla_state.py --npcs 0 --peds 0   # quiet world

Teleop controls (focus the pygame window):
  W / Up | S / Down | A / Left | D / Right | Space (handbrake) | R (reverse) | Q / Esc

Logs go to logs/carla_state_<timestamp>.jsonl
"""

import argparse
import json
import math
import os
import random
import sys
import time

# Allow imports from project root when running as `python extras/explore_carla_state.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import carla_bootstrap  # noqa: F401  (sys.path side effect for `agents.*`)

import carla
import pygame

from control.pid_controller import PIDController
from planning.route_planner import RoutePlanner
from planning.waypoint_manager import WaypointManager
from utils.carla_utils import normalize_angle, output_to_throttle_brake


NEARBY_RADIUS_M = 60.0
LOOP_HZ = 20
LOG_INTERVAL_S = 0.25
PRINT_INTERVAL_S = 2.0


def transform_to_dict(t):
    return {
        "location": {"x": t.location.x, "y": t.location.y, "z": t.location.z},
        "rotation": {"pitch": t.rotation.pitch, "yaw": t.rotation.yaw, "roll": t.rotation.roll},
    }


def vector_to_dict(v):
    return {"x": v.x, "y": v.y, "z": v.z}


def bbox_to_dict(b):
    return {"extent": vector_to_dict(b.extent), "location": vector_to_dict(b.location)}


def collect_ego_state(vehicle):
    transform = vehicle.get_transform()
    velocity = vehicle.get_velocity()
    ang_vel = vehicle.get_angular_velocity()
    accel = vehicle.get_acceleration()
    control = vehicle.get_control()
    light_state = vehicle.get_light_state()
    approaching_light = vehicle.get_traffic_light()

    return {
        "id": vehicle.id,
        "type_id": vehicle.type_id,
        "is_alive": vehicle.is_alive,
        "attributes": dict(vehicle.attributes),
        "semantic_tags": list(vehicle.semantic_tags),
        "transform": transform_to_dict(transform),
        "velocity": vector_to_dict(velocity),
        "speed_mps": math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2),
        "angular_velocity": vector_to_dict(ang_vel),
        "acceleration": vector_to_dict(accel),
        "control": {
            "throttle": control.throttle,
            "steer": control.steer,
            "brake": control.brake,
            "hand_brake": control.hand_brake,
            "reverse": control.reverse,
            "manual_gear_shift": control.manual_gear_shift,
            "gear": control.gear,
        },
        "bounding_box": bbox_to_dict(vehicle.bounding_box),
        "vehicle_light_state_bits": int(light_state),
        "speed_limit_kmh": vehicle.get_speed_limit(),
        "is_at_traffic_light": vehicle.is_at_traffic_light(),
        "approaching_traffic_light_state": str(vehicle.get_traffic_light_state()),
        "approaching_traffic_light_id": (approaching_light.id if approaching_light else None),
    }


def collect_waypoint_context(carla_map, ego_loc):
    wp = carla_map.get_waypoint(ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
    if wp is None:
        return None
    left = wp.get_left_lane()
    right = wp.get_right_lane()
    landmarks = wp.get_landmarks(50.0, stop_at_junction=False)

    return {
        "id": wp.id,
        "road_id": wp.road_id,
        "section_id": wp.section_id,
        "lane_id": wp.lane_id,
        "lane_type": str(wp.lane_type),
        "lane_width": wp.lane_width,
        "lane_change": str(wp.lane_change),
        "s_along_road": wp.s,
        "is_junction": wp.is_junction,
        "junction_id": wp.junction_id,
        "transform": transform_to_dict(wp.transform),
        "left_lane": (
            {"lane_id": left.lane_id, "lane_type": str(left.lane_type), "lane_width": left.lane_width}
            if left else None
        ),
        "right_lane": (
            {"lane_id": right.lane_id, "lane_type": str(right.lane_type), "lane_width": right.lane_width}
            if right else None
        ),
        "landmarks_within_50m": [
            {
                "id": lm.id,
                "name": lm.name,
                "type": lm.type,
                "sub_type": lm.sub_type,
                "value": lm.value,
                "unit": lm.unit,
                "text": lm.text,
                "distance": lm.distance,
                "s": lm.s,
                "t": lm.t,
                "is_dynamic": lm.is_dynamic,
                "transform": transform_to_dict(lm.transform),
            }
            for lm in landmarks
        ],
    }


def collect_nearby_traffic_lights(world, ego_loc, radius=NEARBY_RADIUS_M):
    lights = world.get_actors().filter("traffic.traffic_light")
    out = []
    for light in lights:
        loc = light.get_location()
        d = loc.distance(ego_loc)
        if d > radius:
            continue
        affected = light.get_affected_lane_waypoints()
        group = light.get_group_traffic_lights()
        out.append({
            "id": light.id,
            "type_id": light.type_id,
            "is_alive": light.is_alive,
            "transform": transform_to_dict(light.get_transform()),
            "location": vector_to_dict(loc),
            "distance_to_ego_m": d,
            "state": str(light.get_state()),
            "elapsed_time_s": light.get_elapsed_time(),
            "red_time_s": light.get_red_time(),
            "yellow_time_s": light.get_yellow_time(),
            "green_time_s": light.get_green_time(),
            "is_frozen": light.is_frozen(),
            "pole_index": light.get_pole_index(),
            "group_traffic_light_ids": [tl.id for tl in group],
            "trigger_volume": bbox_to_dict(light.trigger_volume) if hasattr(light, "trigger_volume") else None,
            "affected_lane_waypoints": [
                {
                    "road_id": wp.road_id,
                    "lane_id": wp.lane_id,
                    "s": wp.s,
                    "transform": transform_to_dict(wp.transform),
                }
                for wp in affected
            ],
        })
    return out


def collect_nearby_stop_signs(world, ego_loc, radius=NEARBY_RADIUS_M):
    stops = world.get_actors().filter("traffic.stop")
    out = []
    for sign in stops:
        loc = sign.get_location()
        d = loc.distance(ego_loc)
        if d > radius:
            continue
        out.append({
            "id": sign.id,
            "type_id": sign.type_id,
            "is_alive": sign.is_alive,
            "transform": transform_to_dict(sign.get_transform()),
            "location": vector_to_dict(loc),
            "distance_to_ego_m": d,
            "trigger_volume": bbox_to_dict(sign.trigger_volume) if hasattr(sign, "trigger_volume") else None,
        })
    return out


def collect_nearby_actors(world, ego, type_filter, radius=NEARBY_RADIUS_M):
    ego_loc = ego.get_location()
    actors = world.get_actors().filter(type_filter)
    out = []
    for a in actors:
        if a.id == ego.id:
            continue
        loc = a.get_location()
        d = loc.distance(ego_loc)
        if d > radius:
            continue
        v = a.get_velocity()
        out.append({
            "id": a.id,
            "type_id": a.type_id,
            "is_alive": a.is_alive,
            "transform": transform_to_dict(a.get_transform()),
            "velocity": vector_to_dict(v),
            "speed_mps": math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2),
            "distance_to_ego_m": d,
            "bounding_box": bbox_to_dict(a.bounding_box) if hasattr(a, "bounding_box") else None,
        })
    return out


def collect_weather(world):
    w = world.get_weather()
    return {
        "cloudiness": w.cloudiness,
        "precipitation": w.precipitation,
        "precipitation_deposits": w.precipitation_deposits,
        "wind_intensity": w.wind_intensity,
        "sun_azimuth_angle": w.sun_azimuth_angle,
        "sun_altitude_angle": w.sun_altitude_angle,
        "fog_density": w.fog_density,
        "fog_distance": w.fog_distance,
        "fog_falloff": w.fog_falloff,
        "wetness": w.wetness,
        "scattering_intensity": w.scattering_intensity,
        "mie_scattering_scale": w.mie_scattering_scale,
        "rayleigh_scattering_scale": w.rayleigh_scattering_scale,
    }


def print_summary(state):
    ego = state["ego"]
    wp = state["ego_waypoint"] or {}
    tls = state["traffic_lights_nearby"]
    stops = state["stop_signs_nearby"]
    veh = state["vehicles_nearby"]
    peds = state["pedestrians_nearby"]

    loc = ego["transform"]["location"]
    print(f"\n--- t={state['t']:6.2f}s ---")
    print(f"  Ego @ ({loc['x']:7.1f}, {loc['y']:7.1f})  "
          f"yaw={ego['transform']['rotation']['yaw']:7.1f}°  "
          f"speed={ego['speed_mps']:5.2f} m/s  "
          f"limit={ego['speed_limit_kmh']:.0f} km/h")
    print(f"  Control: throttle={ego['control']['throttle']:.2f} "
          f"steer={ego['control']['steer']:+.2f} "
          f"brake={ego['control']['brake']:.2f}  gear={ego['control']['gear']}")
    print(f"  At traffic light: {ego['is_at_traffic_light']}  "
          f"approaching state={ego['approaching_traffic_light_state']}  "
          f"id={ego['approaching_traffic_light_id']}")
    if wp:
        print(f"  Waypoint: road={wp['road_id']} lane={wp['lane_id']} "
              f"type={wp['lane_type']} width={wp['lane_width']:.2f}m  "
              f"junction={wp['is_junction']} "
              f"({len(wp['landmarks_within_50m'])} landmarks within 50m)")
    print(f"  Nearby traffic lights ({len(tls)}):")
    for tl in tls[:4]:
        print(f"    TL {tl['id']:5d}: {tl['state']:>30s}  d={tl['distance_to_ego_m']:5.1f}m  "
              f"R={tl['red_time_s']:.1f}s Y={tl['yellow_time_s']:.1f}s G={tl['green_time_s']:.1f}s  "
              f"controls {len(tl['affected_lane_waypoints'])} lanes")
    print(f"  Nearby stop signs ({len(stops)}):")
    for s in stops[:2]:
        print(f"    Stop {s['id']:5d}: d={s['distance_to_ego_m']:5.1f}m")
    print(f"  Nearby vehicles ({len(veh)}):")
    for v in veh[:4]:
        print(f"    {v['type_id']:30s} d={v['distance_to_ego_m']:5.1f}m  speed={v['speed_mps']:5.2f}")
    if peds:
        print(f"  Nearby pedestrians ({len(peds)}):")
        for p in peds[:3]:
            print(f"    {p['type_id']:30s} d={p['distance_to_ego_m']:5.1f}m")


TELEOP_HELP = (
    "Teleop keys (focus the small pygame window):\n"
    "  W / Up    : throttle\n"
    "  S / Down  : brake\n"
    "  A / Left  : steer left\n"
    "  D / Right : steer right\n"
    "  Space     : handbrake (hold)\n"
    "  R         : toggle reverse\n"
    "  Q / Esc   : quit\n"
)


def init_teleop_window():
    pygame.init()
    display = pygame.display.set_mode((600, 320))
    pygame.display.set_caption("CARLA Teleop — CLICK HERE for keyboard control")
    font = pygame.font.SysFont("monospace", 18)
    big_font = pygame.font.SysFont("monospace", 22, bold=True)
    return display, font, big_font


def teleop_step(control, reverse_state):
    """Returns (control, reverse_state, quit_requested) after consuming key events."""
    quit_requested = False
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            quit_requested = True
        elif event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_q, pygame.K_ESCAPE):
                quit_requested = True
            elif event.key == pygame.K_r:
                reverse_state = not reverse_state

    keys = pygame.key.get_pressed()

    if keys[pygame.K_w] or keys[pygame.K_UP]:
        control.throttle = min(control.throttle + 0.05, 1.0)
    else:
        control.throttle = 0.0

    if keys[pygame.K_s] or keys[pygame.K_DOWN]:
        control.brake = min(control.brake + 0.1, 1.0)
    else:
        control.brake = 0.0

    steer_step = 0.05
    if keys[pygame.K_a] or keys[pygame.K_LEFT]:
        control.steer = max(control.steer - steer_step, -0.7)
    elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
        control.steer = min(control.steer + steer_step, 0.7)
    else:
        if control.steer > 0:
            control.steer = max(control.steer - steer_step, 0.0)
        elif control.steer < 0:
            control.steer = min(control.steer + steer_step, 0.0)

    control.hand_brake = bool(keys[pygame.K_SPACE])
    control.reverse = reverse_state

    return control, reverse_state, quit_requested


def render_teleop_hud(display, font, big_font, ego_speed_mps, control):
    focused = pygame.key.get_focused()
    bg = (10, 60, 20) if focused else (60, 10, 10)
    display.fill(bg)

    banner = "KEYBOARD ACTIVE" if focused else "CLICK THIS WINDOW FOR CONTROL"
    banner_color = (120, 255, 140) if focused else (255, 120, 120)
    surf = big_font.render(banner, True, banner_color)
    display.blit(surf, (10, 10))

    lines = [
        f"speed:     {ego_speed_mps:5.2f} m/s  ({ego_speed_mps * 3.6:.1f} km/h)",
        f"throttle:  {control.throttle:.2f}",
        f"brake:     {control.brake:.2f}",
        f"steer:     {control.steer:+.2f}",
        f"reverse:   {control.reverse}",
        f"handbrake: {control.hand_brake}",
        "",
        "WASD / arrows  | space = handbrake",
        "R = reverse    | Q / Esc = quit",
    ]
    for i, line in enumerate(lines):
        surf = font.render(line, True, (200, 220, 255))
        display.blit(surf, (10, 50 + i * 24))
    pygame.display.flip()


def spawn_npc_vehicles(client, world, n: int):
    if n <= 0:
        return []
    tm = client.get_trafficmanager()
    bps = world.get_blueprint_library().filter("vehicle.*")
    bps = [
        bp for bp in bps
        if bp.has_attribute("number_of_wheels")
        and int(bp.get_attribute("number_of_wheels")) == 4
    ]
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    npcs = []
    for sp in spawn_points:
        if len(npcs) >= n:
            break
        bp = random.choice(bps)
        if bp.has_attribute("color"):
            color = random.choice(bp.get_attribute("color").recommended_values)
            bp.set_attribute("color", color)
        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", "autopilot")
        v = world.try_spawn_actor(bp, sp)
        if v is not None:
            v.set_autopilot(True, tm.get_port())
            npcs.append(v)
    return npcs


def spawn_pedestrians(world, n: int):
    if n <= 0:
        return [], []
    walker_bps = world.get_blueprint_library().filter("walker.pedestrian.*")
    controller_bp = world.get_blueprint_library().find("controller.ai.walker")

    walkers = []
    controllers = []
    attempts = 0
    while len(walkers) < n and attempts < n * 4:
        attempts += 1
        loc = world.get_random_location_from_navigation()
        if loc is None:
            continue
        spawn_point = carla.Transform(location=loc)
        bp = random.choice(walker_bps)
        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible", "false")
        walker = world.try_spawn_actor(bp, spawn_point)
        if walker is None:
            continue
        ctrl = world.try_spawn_actor(controller_bp, carla.Transform(), attach_to=walker)
        if ctrl is None:
            walker.destroy()
            continue
        walkers.append(walker)
        controllers.append(ctrl)

    time.sleep(0.3)  # let physics settle
    for ctrl in controllers:
        ctrl.start()
        target = world.get_random_location_from_navigation()
        if target is not None:
            ctrl.go_to_location(target)
        ctrl.set_max_speed(1.4)
    return walkers, controllers


def cleanup_actors(npcs, walkers, walker_controllers):
    for ctrl in walker_controllers:
        try:
            ctrl.stop()
        except Exception:
            pass
    for actor in walker_controllers + walkers + npcs:
        try:
            if actor.is_alive:
                actor.destroy()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--town", default=None,
                        help="Optional town to load (e.g. Town01).")
    parser.add_argument("--vehicle", default="vehicle.mercedes.coupe")
    parser.add_argument("--spawn-idx", type=int, default=0)
    parser.add_argument("--mode", choices=["teleop", "pid", "autopilot"], default="teleop",
                        help="Drive mode for the ego: teleop (pygame keys), "
                             "pid (our stack drives a route), or autopilot (CARLA TM).")
    parser.add_argument("--end-spawn-idx", type=int, default=50,
                        help="Destination spawn-point index for --mode pid.")
    parser.add_argument("--pid-target-speed", type=float, default=8.0,
                        help="Cruise speed (m/s) for --mode pid.")
    parser.add_argument("--npcs", type=int, default=20,
                        help="Number of background NPC vehicles (default 20).")
    parser.add_argument("--peds", type=int, default=10,
                        help="Number of background pedestrians (default 10).")
    args = parser.parse_args()

    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.load_world(args.town) if args.town else client.get_world()
    carla_map = world.get_map()

    settings = world.get_settings()
    if settings.synchronous_mode:
        print("World was in synchronous mode; switching to async for this script.")
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)

    blueprint = world.get_blueprint_library().find(args.vehicle)
    spawn_points = carla_map.get_spawn_points()
    spawn_point = spawn_points[args.spawn_idx]
    vehicle = None
    for offset in range(len(spawn_points)):
        sp = spawn_points[(args.spawn_idx + offset) % len(spawn_points)]
        try:
            vehicle = world.spawn_actor(blueprint, sp)
            break
        except RuntimeError:
            continue
    if vehicle is None:
        raise RuntimeError("Could not spawn ego: every spawn point was occupied.")
    print(f"Spawned ego: id={vehicle.id} type={vehicle.type_id} at {spawn_point.location}")

    teleop_display = None
    teleop_font = None
    teleop_big_font = None
    route_planner = None
    waypoint_manager = None
    lateral_pid = None
    longitudinal_pid = None

    if args.mode == "autopilot":
        print("WARNING: --mode autopilot uses CARLA's Traffic Manager, which has")
        print("         known stability issues in async mode with many actors.")
        print("         If it crashes, switch to --mode pid (our PID stack).\n")
        vehicle.set_autopilot(True)
    elif args.mode == "pid":
        end_loc = spawn_points[args.end_spawn_idx % len(spawn_points)].location
        route_planner = RoutePlanner(carla_map, sampling_resolution=1.0)
        route = route_planner.compute(vehicle.get_location(), end_loc)
        if not route:
            raise RuntimeError("Could not compute initial route for PID mode.")
        waypoint_manager = WaypointManager(
            route,
            target_speed=args.pid_target_speed,
            search_window=10,
            slow_down_dist_m=15.0,
            slow_down_floor_speed=1.0,
        )
        lateral_pid = PIDController(1.2, 0.01, 0.3, dt=0.05, integral_limit=2.0)
        longitudinal_pid = PIDController(1.0, 0.05, 0.1, dt=0.05, integral_limit=2.0)
        print(f"PID mode: {len(route)} waypoints toward spawn[{args.end_spawn_idx}].")
        print("On goal reached, a new random destination is chosen automatically.\n")
    else:  # teleop
        teleop_display, teleop_font, teleop_big_font = init_teleop_window()
        print("Teleop mode enabled.")
        print(TELEOP_HELP)
        print("IMPORTANT: click the small pygame window for keyboard control.")
        print("           If you press WASD with the CARLA window focused,")
        print("           CARLA's own free-fly camera will move instead of the car.\n")

    print(f"Spawning {args.npcs} NPC vehicles and {args.peds} pedestrians...")
    npc_vehicles = spawn_npc_vehicles(client, world, args.npcs)
    walkers, walker_controllers = spawn_pedestrians(world, args.peds)
    print(f"  spawned {len(npc_vehicles)} NPC vehicles, {len(walkers)} pedestrians.\n")

    os.makedirs("logs", exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    jsonl_path = f"logs/carla_state_{ts}.jsonl"
    f_jsonl = open(jsonl_path, "w")

    print(f"=== Map: {carla_map.name} ===")
    print(f"  total traffic lights: {len(world.get_actors().filter('traffic.traffic_light'))}")
    print(f"  total stop signs:     {len(world.get_actors().filter('traffic.stop'))}")
    print(f"  total spawn points:   {len(spawn_points)}")
    print(f"\nLogging state to {jsonl_path} every {LOG_INTERVAL_S}s")
    print(f"Console summary every {PRINT_INTERVAL_S}s. Ctrl+C to quit.\n")

    spectator = world.get_spectator()
    clock = pygame.time.Clock()
    t0 = time.time()
    last_log = -LOG_INTERVAL_S
    last_print = -PRINT_INTERVAL_S
    control = carla.VehicleControl()
    reverse_state = False
    last_state = None
    quit_requested = False

    try:
        while not quit_requested:
            now = time.time() - t0
            ego_loc = vehicle.get_location()

            if args.mode == "teleop":
                control, reverse_state, quit_requested = teleop_step(control, reverse_state)
                vehicle.apply_control(control)
            elif args.mode == "pid":
                if waypoint_manager.goal_distance(ego_loc.x, ego_loc.y) < 5.0:
                    new_end = random.choice(spawn_points).location
                    new_route = route_planner.compute(ego_loc, new_end)
                    if new_route:
                        waypoint_manager = WaypointManager(
                            new_route,
                            target_speed=args.pid_target_speed,
                            search_window=10,
                            slow_down_dist_m=15.0,
                            slow_down_floor_speed=1.0,
                        )
                        lateral_pid.reset()
                        longitudinal_pid.reset()
                        print(f"  [pid] new destination: {len(new_route)} waypoints away.")

                ego_yaw_rad = math.radians(vehicle.get_transform().rotation.yaw)
                v = vehicle.get_velocity()
                ego_speed = math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)

                waypoint_manager.update(ego_loc.x, ego_loc.y)
                lookahead_dist = max(5.0, ego_speed * 0.7)
                lx, ly, target_speed, _ = waypoint_manager.get_lookahead(
                    ego_loc.x, ego_loc.y, lookahead_dist
                )
                heading_error = normalize_angle(
                    math.atan2(ly - ego_loc.y, lx - ego_loc.x) - ego_yaw_rad
                )
                steer = max(-1.0, min(1.0, lateral_pid.step(heading_error)))
                speed_error = target_speed - ego_speed
                throttle, brake = output_to_throttle_brake(longitudinal_pid.step(speed_error))
                vehicle.apply_control(carla.VehicleControl(
                    throttle=throttle, steer=steer, brake=brake
                ))

            if now - last_log >= LOG_INTERVAL_S:
                last_state = {
                    "t": now,
                    "ego": collect_ego_state(vehicle),
                    "ego_waypoint": collect_waypoint_context(carla_map, ego_loc),
                    "traffic_lights_nearby": collect_nearby_traffic_lights(world, ego_loc),
                    "stop_signs_nearby": collect_nearby_stop_signs(world, ego_loc),
                    "vehicles_nearby": collect_nearby_actors(world, vehicle, "vehicle.*"),
                    "pedestrians_nearby": collect_nearby_actors(world, vehicle, "walker.*"),
                    "weather": collect_weather(world),
                }
                f_jsonl.write(json.dumps(last_state) + "\n")
                f_jsonl.flush()
                last_log = now

                if now - last_print >= PRINT_INTERVAL_S:
                    print_summary(last_state)
                    last_print = now

            t = vehicle.get_transform()
            fwd = t.get_forward_vector()
            spectator.set_transform(carla.Transform(
                carla.Location(
                    x=t.location.x - 8.0 * fwd.x,
                    y=t.location.y - 8.0 * fwd.y,
                    z=t.location.z + 4.0,
                ),
                carla.Rotation(pitch=-15, yaw=t.rotation.yaw),
            ))

            if args.mode == "teleop" and last_state is not None:
                render_teleop_hud(teleop_display, teleop_font, teleop_big_font,
                                  last_state["ego"]["speed_mps"], control)

            clock.tick(LOOP_HZ)

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        f_jsonl.close()
        cleanup_actors(npc_vehicles, walkers, walker_controllers)
        if vehicle is not None and vehicle.is_alive:
            vehicle.destroy()
        if teleop_display is not None:
            pygame.quit()
        print(f"Cleanup done. Log saved to {jsonl_path}")


if __name__ == "__main__":
    main()
