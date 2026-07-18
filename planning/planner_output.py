from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class PlannerOutput:
    fsm_state: str              # current FSM state name
    target_speed_mps: float     # requested cruise speed (safety layer may lower it)
    target_lane_offset: float   # metres lateral from route centreline (0.0 = stay in lane)
    reason: str                 # human-readable decision string for logs
    action_costs: dict = field(default_factory=dict)  # Phase 7 placeholder