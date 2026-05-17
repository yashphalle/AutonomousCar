"""
SceneSpawner — resolves NPCSpec entries to CARLA locations and spawns actors.

Usage in main.py:
    spawner = SceneSpawner(world, carla_map, route, start_idx=config.START_SPAWN_IDX)
    npc_actors = spawner.spawn(SCENES["follow_vehicle_close"])
    # ... run loop ...
    spawner.destroy()
"""
from __future__ import annotations

import math
import random

import carla

from extras.scenes.definitions import NPCSpec, SceneConfig


class SceneSpawner:
    def __init__(self, client, world, carla_map, route: list, start_idx: int = 0):
        self._client = client
        self._world = world
        self._map = carla_map
        self._route = route
        self._start_idx = start_idx
        self._actors: list = []
        self._bp_lib = world.get_blueprint_library()

    # ------------------------------------------------------------------
    def spawn(self, scene: SceneConfig) -> list:
        """Spawn all NPCs in the scene. Returns the list of spawned actors."""
        print(f"[scene_spawner] loading scene '{scene.name}': {scene.description}")
        for spec in scene.npcs:
            actor = self._spawn_npc(spec)
            if actor is not None:
                self._actors.append(actor)
        print(f"[scene_spawner] spawned {len(self._actors)} NPCs")
        return self._actors

    def destroy(self) -> None:
        for actor in self._actors:
            if actor.is_alive:
                actor.destroy()
        self._actors.clear()
        print("[scene_spawner] all NPCs destroyed")

    # ------------------------------------------------------------------
    def _spawn_npc(self, spec: NPCSpec):
        """Resolve offset → waypoint → spawn vehicle or pedestrian."""
        target_wp = self._resolve_route_offset(spec.route_offset_m, spec.lateral_offset_m)
        if target_wp is None:
            print(f"[scene_spawner] WARNING: could not resolve offset {spec.route_offset_m:.1f}m for '{spec.label}'")
            return None

        if spec.is_pedestrian:
            return self._spawn_pedestrian(spec, target_wp)
        return self._spawn_vehicle(spec, target_wp)

    def _spawn_vehicle(self, spec: NPCSpec, target_wp):
        bp = self._bp_lib.find(spec.blueprint)
        if bp is None:
            bp = random.choice(self._bp_lib.filter("vehicle.*"))

        if bp.has_attribute("color"):
            bp.set_attribute("color", "255,0,0")  # red = NPC

        spawn_transform = target_wp.transform
        spawn_transform.location.z += 0.3

        actor = self._world.try_spawn_actor(bp, spawn_transform)
        if actor is None:
            print(f"[scene_spawner] WARNING: vehicle spawn failed for '{spec.label}'")
            return None

        tm = self._client.get_trafficmanager()
        actor.set_autopilot(True, tm.get_port())
        desired_kmh = spec.target_speed_mps * 3.6
        speed_limit_kmh = 30.0
        perc_below = max(0.0, (speed_limit_kmh - desired_kmh) / speed_limit_kmh * 100.0)
        tm.vehicle_percentage_speed_difference(actor, perc_below)

        print(f"[scene_spawner] spawned vehicle '{spec.label}' id={actor.id} offset={spec.route_offset_m:.1f}m target={desired_kmh:.1f}km/h")
        return actor

    def _spawn_pedestrian(self, spec: NPCSpec, target_wp):
        walkers = self._bp_lib.filter("walker.pedestrian.*")
        if not walkers:
            print(f"[scene_spawner] WARNING: no pedestrian blueprints found")
            return None
        bp = random.choice(walkers)
        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible", "false")

        spawn_transform = target_wp.transform
        spawn_transform.location.z += 0.5  # lift above ground

        walker = self._world.try_spawn_actor(bp, spawn_transform)
        if walker is None:
            print(f"[scene_spawner] WARNING: pedestrian spawn failed for '{spec.label}'")
            return None

        # Spawn AI controller and start it
        ctrl_bp = self._bp_lib.find("controller.ai.walker")
        controller = self._world.try_spawn_actor(ctrl_bp, carla.Transform(), walker)
        if controller is not None:
            self._actors.append(controller)  # track controller separately for cleanup
            self._world.tick()               # controller needs a tick to initialise
            controller.start()
            if spec.target_speed_mps > 0:
                if spec.walk_to_lateral_m is not None:
                    # Directed crossing: walk to the opposite side of the road
                    dest_wp = self._resolve_route_offset(spec.route_offset_m, spec.walk_to_lateral_m)
                    dest = dest_wp.transform.location if dest_wp else self._world.get_random_location_from_navigation()
                else:
                    dest = self._world.get_random_location_from_navigation()
                controller.go_to_location(dest)
                controller.set_max_speed(spec.target_speed_mps)
            # speed=0 → walker stands still (no go_to_location call)

        print(f"[scene_spawner] spawned pedestrian '{spec.label}' id={walker.id} offset={spec.route_offset_m:.1f}m speed={spec.target_speed_mps:.1f}m/s")
        return walker

    def _resolve_route_offset(self, offset_m: float, lateral_offset_m: float = 0.0):
        """
        Walk the route from start_idx, accumulating arc length until offset_m is reached.
        Returns the CARLA Waypoint at that position (optionally shifted laterally).
        """
        if not self._route:
            return None

        accumulated = 0.0
        prev_loc = self._route[self._start_idx][0].transform.location

        for i in range(self._start_idx, len(self._route)):
            wp, _ = self._route[i]
            loc = wp.transform.location
            seg_len = math.hypot(loc.x - prev_loc.x, loc.y - prev_loc.y)
            accumulated += seg_len
            prev_loc = loc

            if accumulated >= abs(offset_m):
                if lateral_offset_m == 0.0:
                    return wp
                # Shift laterally: perpendicular to waypoint heading
                heading = math.radians(wp.transform.rotation.yaw)
                perp = heading + math.pi / 2.0
                shift_x = lateral_offset_m * math.cos(perp)
                shift_y = lateral_offset_m * math.sin(perp)
                # Return a new transform at the shifted location
                shifted_loc = carla.Location(
                    x=loc.x + shift_x,
                    y=loc.y + shift_y,
                    z=loc.z,
                )
                t = carla.Transform(shifted_loc, wp.transform.rotation)
                # We need a Waypoint-like object; return the nearest waypoint to shifted loc
                shifted_wp = self._map.get_waypoint(shifted_loc, project_to_road=False)
                return shifted_wp if shifted_wp else wp

        # offset beyond route end — return last waypoint
        return self._route[-1][0]
