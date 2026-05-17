"""
Scan a CARLA town for stop sign actors and print which spawn points are nearest.
Run this while CARLA is running to find routes that will hit stop_and_go FSM.

Usage:
    python3.10 extras/find_stop_signs.py
    python3.10 extras/find_stop_signs.py --town Town02
"""
import argparse
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import carla_bootstrap  # noqa: F401
import carla

parser = argparse.ArgumentParser()
parser.add_argument("--town", default=None, help="Load a specific town (e.g. Town02)")
args = parser.parse_args()

client = carla.Client("localhost", 2000)
client.set_timeout(30.0)

if args.town:
    print(f"Loading {args.town} …")
    world = client.load_world(args.town)
else:
    world = client.get_world()

carla_map = world.get_map()
print(f"Map: {carla_map.name}")
spawn_points = carla_map.get_spawn_points()

stop_signs = list(world.get_actors().filter("traffic.stop"))
print(f"Town01 stop signs found: {len(stop_signs)}")

if not stop_signs:
    print("No traffic.stop actors in this map — stop_and_go FSM cannot be tested here.")
    print("Consider using Town02 or Town03 which have stop signs.")
    sys.exit(0)

for ss in stop_signs:
    loc = ss.get_location()
    # Find 3 nearest spawn points
    dists = sorted(
        [(i, math.hypot(sp.location.x - loc.x, sp.location.y - loc.y))
         for i, sp in enumerate(spawn_points)],
        key=lambda x: x[1],
    )
    nearest = dists[:3]
    wp = carla_map.get_waypoint(loc, project_to_road=True)
    road_info = f"road={wp.road_id} lane={wp.lane_id}" if wp else "road=?"
    print(
        f"\nStop sign id={ss.id}  {road_info}  pos=({loc.x:.0f},{loc.y:.0f})"
    )
    for idx, dist in nearest:
        print(f"  nearest spawn[{idx:3d}]  {dist:6.1f}m")

print("\nTo test stop_and_go, pick a START and END spawn that bracket a stop sign.")
print("Example:  python3.10 main.py --start <near> --end <far> --scene test_lane_follow")
