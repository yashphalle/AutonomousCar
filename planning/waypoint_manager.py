import math


class WaypointManager:
    def __init__(
        self,
        route,
        target_speed: float,
        search_window: int = 10,
        slow_down_dist_m: float = 15.0,
        slow_down_floor_speed: float = 1.0,
    ):
        if not route:
            raise ValueError("Route is empty.")
        self.route = route
        self.target_speed = target_speed
        self.search_window = search_window
        self.slow_down_dist_m = slow_down_dist_m
        self.slow_down_floor = slow_down_floor_speed
        self.current_idx = 0
        self._speed_profile = None

    def set_speed_profile(self, profile) -> None:
        self._speed_profile = profile

    def update(self, ego_x: float, ego_y: float) -> None:
        # Window-search rather than "advance if next is closer": at junctions
        # the next sparse waypoint can be farther than the current, freezing a
        # naive pointer.
        best_idx = self.current_idx
        best_loc = self.route[best_idx][0].transform.location
        best_dist = math.hypot(ego_x - best_loc.x, ego_y - best_loc.y)

        end = min(self.current_idx + self.search_window + 1, len(self.route))
        for i in range(self.current_idx + 1, end):
            loc = self.route[i][0].transform.location
            d = math.hypot(ego_x - loc.x, ego_y - loc.y)
            if d < best_dist:
                best_dist = d
                best_idx = i
        self.current_idx = best_idx

    def get_lookahead(self, ego_x: float, ego_y: float, lookahead_dist: float):
        idx = self.current_idx
        cur = self.route[idx][0].transform.location
        accumulated = math.hypot(ego_x - cur.x, ego_y - cur.y)

        while idx < len(self.route) - 1 and accumulated < lookahead_dist:
            a = self.route[idx][0].transform.location
            b = self.route[idx + 1][0].transform.location
            accumulated += math.hypot(b.x - a.x, b.y - a.y)
            idx += 1

        wp, road_option = self.route[idx]
        loc = wp.transform.location

        # Taper near the goal so the car doesn't overshoot the final waypoint.
        goal_dist = self.goal_distance(ego_x, ego_y)
        if self._speed_profile is not None:
            offset = idx - self.current_idx
            speeds = self._speed_profile.target_speeds_mps
            profile_speed = speeds[min(offset, len(speeds) - 1)] if speeds else self.target_speed
            # still apply goal-approach taper on top of profile speed
            if goal_dist < self.slow_down_dist_m:
                ratio = max(0.0, goal_dist / self.slow_down_dist_m)
                target_speed = max(self.slow_down_floor, profile_speed * ratio)
            else:
                target_speed = profile_speed
        else:
            # original Stage 1 taper logic (unchanged)
            if goal_dist < self.slow_down_dist_m:
                ratio = max(0.0, goal_dist / self.slow_down_dist_m)
                target_speed = max(self.slow_down_floor, self.target_speed * ratio)
            else:
                target_speed = self.target_speed

        return loc.x, loc.y, target_speed, road_option

    def goal_distance(self, ego_x: float, ego_y: float) -> float:
        last = self.route[-1][0].transform.location
        return math.hypot(ego_x - last.x, ego_y - last.y)
