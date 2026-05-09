"""
Draws CARLA's HD map (road centerlines, junctions, traffic lights, stop signs)
into the running simulator viewport using debug drawing.

Static elements are drawn once with persistent life_time. Traffic-light states
are refreshed every 0.5 s so you can watch them cycle live.

Usage:
  Terminal 1: ./CarlaUE4.sh
  Terminal 2: python extras/map_overlay.py [--town Town01]

Press Ctrl+C in the script terminal to quit. Debug drawings persist in the
simulator until CARLA is restarted.
"""

import argparse
import time

import carla


COLOR_ROAD = carla.Color(40, 90, 180)         # blue
COLOR_JUNCTION = carla.Color(255, 140, 0)     # orange
COLOR_STOP_SIGN = carla.Color(220, 30, 30)    # red

LIGHT_STATE_COLOR = {
    carla.TrafficLightState.Red:     carla.Color(255, 30, 30),
    carla.TrafficLightState.Yellow:  carla.Color(255, 200, 0),
    carla.TrafficLightState.Green:   carla.Color(40, 220, 60),
    carla.TrafficLightState.Off:     carla.Color(150, 150, 150),
    carla.TrafficLightState.Unknown: carla.Color(150, 150, 150),
}


def draw_road_network(world, carla_map, sampling: float = 2.0):
    waypoints = carla_map.generate_waypoints(sampling)
    print(f"  Drawing {len(waypoints)} lane waypoints...")
    for wp in waypoints:
        color = COLOR_JUNCTION if wp.is_junction else COLOR_ROAD
        a = wp.transform.location
        for next_wp in wp.next(sampling):
            b = next_wp.transform.location
            world.debug.draw_line(
                carla.Location(x=a.x, y=a.y, z=a.z + 0.3),
                carla.Location(x=b.x, y=b.y, z=b.z + 0.3),
                thickness=0.08,
                color=color,
                life_time=0.0,
            )


def draw_stop_signs(world) -> int:
    stops = world.get_actors().filter("traffic.stop")
    for sign in stops:
        loc = sign.get_location()
        world.debug.draw_point(
            carla.Location(x=loc.x, y=loc.y, z=loc.z + 0.8),
            size=0.22,
            color=COLOR_STOP_SIGN,
            life_time=0.0,
        )
        world.debug.draw_string(
            carla.Location(x=loc.x, y=loc.y, z=loc.z + 1.4),
            "STOP",
            color=COLOR_STOP_SIGN,
            life_time=0.0,
        )
    return len(stops)


def draw_traffic_lights(world, life_time: float):
    lights = world.get_actors().filter("traffic.traffic_light")
    for light in lights:
        loc = light.get_location()
        color = LIGHT_STATE_COLOR.get(
            light.get_state(), carla.Color(150, 150, 150)
        )
        world.debug.draw_point(
            carla.Location(x=loc.x, y=loc.y, z=loc.z + 1.0),
            size=0.30,
            color=color,
            life_time=life_time,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--town", default=None,
        help="Optional town to load (e.g. Town01). Defaults to the currently-loaded world.",
    )
    parser.add_argument(
        "--sampling", type=float, default=2.0,
        help="Road waypoint sampling distance in metres (default: 2.0).",
    )
    parser.add_argument(
        "--refresh", type=float, default=0.5,
        help="Traffic-light state refresh interval in seconds (default: 0.5).",
    )
    args = parser.parse_args()

    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.load_world(args.town) if args.town else client.get_world()
    carla_map = world.get_map()

    junction_ids = set()
    for a, b in carla_map.get_topology():
        if a.is_junction:
            junction_ids.add(a.junction_id)
        if b.is_junction:
            junction_ids.add(b.junction_id)

    print(f"=== HD Map: {carla_map.name} ===")
    print(f"  topology edges:   {len(carla_map.get_topology())}")
    print(f"  unique junctions: {len(junction_ids)}")
    print(f"  traffic lights:   {len(world.get_actors().filter('traffic.traffic_light'))}")
    print(f"  stop signs:       {len(world.get_actors().filter('traffic.stop'))}")
    print()

    print("Drawing static elements...")
    draw_road_network(world, carla_map, sampling=args.sampling)
    n_stops = draw_stop_signs(world)
    print(f"  stop signs drawn: {n_stops}")

    print("\nLegend:")
    print("  blue lines    : road centerlines (one per driving lane)")
    print("  orange lines  : junction segments")
    print("  red dots/STOP : stop signs")
    print("  R/Y/G dots    : traffic lights (live state, refreshes every 0.5 s)")
    print()
    print("Fly the spectator camera around to explore. Ctrl+C to quit.")
    print("(Drawings persist until CARLA is restarted.)")

    try:
        while True:
            draw_traffic_lights(world, life_time=args.refresh * 1.5)
            time.sleep(args.refresh)
    except KeyboardInterrupt:
        print("\nExiting.")


if __name__ == "__main__":
    main()
