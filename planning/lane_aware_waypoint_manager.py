"""
LaneAwareWaypointManager — wraps the frozen WaypointManager and adds lateral lane offset.

The offset shifts the lookahead target perpendicular to the route heading, allowing
BehaviourPlanner to steer the car toward an adjacent lane without touching frozen code.

Positive offset = left of route centreline.
Negative offset = right of route centreline.
"""
from __future__ import annotations

import math

from planning.waypoint_manager import WaypointManager


class LaneAwareWaypointManager:
    def __init__(self, waypoint_manager: WaypointManager):
        self._wm = waypoint_manager
        self._lane_offset: float = 0.0

    def set_lane_offset(self, offset_m: float) -> None:
        self._lane_offset = offset_m

    def set_speed_profile(self, profile) -> None:
        self._wm.set_speed_profile(profile)

    def update(self, ego_x: float, ego_y: float) -> None:
        self._wm.update(ego_x, ego_y)

    def goal_distance(self, ego_x: float, ego_y: float) -> float:
        return self._wm.goal_distance(ego_x, ego_y)

    @property
    def current_idx(self) -> int:
        return self._wm.current_idx

    def get_lookahead(
        self, ego_x: float, ego_y: float, lookahead_dist: float
    ) -> tuple:
        lx, ly, target_speed, road_option = self._wm.get_lookahead(
            ego_x, ego_y, lookahead_dist
        )
        if abs(self._lane_offset) > 1e-3:
            heading = math.atan2(ly - ego_y, lx - ego_x)
            perp = heading + math.pi / 2.0
            lx += self._lane_offset * math.cos(perp)
            ly += self._lane_offset * math.sin(perp)
        return lx, ly, target_speed, road_option
