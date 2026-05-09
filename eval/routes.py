"""Evaluation routes for Stage 2 baseline. Town01."""
from dataclasses import dataclass


@dataclass
class EvalRoute:
    route_id: int
    label: str
    start_spawn_idx: int
    end_spawn_idx: int
    expected_distance_m: float  # rough estimate for completion %


EVAL_ROUTES: list[EvalRoute] = [
    EvalRoute(1,  "long_straight_low_traffic",      0,   50,  400.0),   # TODO: verify with map_overlay
    EvalRoute(2,  "long_straight_opposite",         50,   0,  400.0),   # TODO: verify with map_overlay
    EvalRoute(3,  "multi_junction",                 10,  80,  350.0),   # TODO: verify with map_overlay
    EvalRoute(4,  "single_traffic_light",           20,  30,  200.0),   # TODO: verify with map_overlay
    EvalRoute(5,  "cross_town_diagonal",             5,  90,  600.0),   # TODO: verify with map_overlay
    EvalRoute(6,  "dense_npc_cluster",              15,  40,  300.0),   # TODO: verify with map_overlay
    EvalRoute(7,  "bridge_speed_limit",             60,  70,  250.0),   # TODO: verify with map_overlay
    EvalRoute(8,  "left_turn_opposing_traffic",     25,  35,  200.0),   # TODO: verify with map_overlay
    EvalRoute(9,  "long_endurance",                  0, 100, 1000.0),   # TODO: verify with map_overlay
    EvalRoute(10, "short_loop",                     45,  48,  100.0),   # TODO: verify with map_overlay
]
