from agents.navigation.global_route_planner import GlobalRoutePlanner


class RoutePlanner:
    def __init__(self, world_map, sampling_resolution: float = 1.0):
        self._planner = GlobalRoutePlanner(world_map, sampling_resolution)

    def compute(self, start_location, end_location):
        return self._planner.trace_route(start_location, end_location)
