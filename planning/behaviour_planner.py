"""
BehaviourPlanner — FSM-based behaviour layer that sits above RuleBasedPlanner.

Data flow:
  SceneState → BehaviourPlanner → PlannerOutput (fsm_state, lane_offset, target_speed)
  SceneState → RuleBasedPlanner → SpeedProfile  (safety overrides — always runs)

Phase 1: LANE_FOLLOW
Phase 2: FOLLOW_VEHICLE + ACC
Phase 4: EMERGENCY_STOP + SLOW_FOR_CROSSWALK
Phase 5: STOP_AND_GO (stop sign dwell)

No carla imports. Reads only SceneState.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from perception.scene_state import SceneState
from planning.planner_output import PlannerOutput


@dataclass
class BehaviourConfig:
    cruise_speed_mps: float = 8.0
    # Phase 2 — ACC
    follow_detect_distance_m: float = 40.0
    follow_gap_time_s: float = 2.0
    follow_min_gap_m: float = 6.0
    acc_proportional_gain: float = 0.5
    acc_min_speed_mps: float = 0.0
    # Phase 4 — Crosswalk
    crosswalk_slow_distance_m: float = 25.0
    crosswalk_slow_speed_mps: float = 3.0
    # Phase 4 — Emergency stop (pedestrian)
    emergency_clear_s: float = 3.0          # ped must be gone this long before resuming
    # Phase 5 — Stop sign
    stop_sign_approach_m: float = 15.0      # start stopping at this distance
    stop_sign_dwell_s: float = 2.0          # seconds to hold at stop line before going


class BehaviourPlanner:
    FSM_LANE_FOLLOW      = "lane_follow"
    FSM_FOLLOW_VEHICLE   = "follow_vehicle"
    FSM_EMERGENCY_STOP   = "emergency_stop"
    FSM_SLOW_CROSSWALK   = "slow_for_crosswalk"
    FSM_STOP_AND_GO      = "stop_and_go"

    def __init__(self, config: BehaviourConfig = None):
        self.config = config or BehaviourConfig()
        self._fsm_state = self.FSM_LANE_FOLLOW
        self._prev_fsm_state = self.FSM_LANE_FOLLOW
        self._prev_reason = ""

        # Emergency stop: timestamp when pedestrian last seen clear
        self._ped_clear_since: float | None = None

        # Stop-and-go: timestamp when ego reached the stop line (speed ≈ 0)
        self._stopped_at_sign_since: float | None = None
        self._stop_sign_served: bool = False   # True after dwell is satisfied

    def plan(self, scene_state: SceneState) -> PlannerOutput:
        now = time.monotonic()

        ego_speed  = scene_state.ego_pose.speed_mps
        lead_dist  = scene_state.lead_vehicle_distance_m
        lead_speed = scene_state.lead_vehicle_speed_mps
        has_lead   = (
            scene_state.lead_vehicle_id is not None
            and lead_dist >= 0.0
            and lead_dist <= self.config.follow_detect_distance_m
        )

        # ── Priority 1: EMERGENCY_STOP (pedestrian in path) ──────────────────
        if scene_state.pedestrian_in_path:
            self._ped_clear_since = None
            self._fsm_state = self.FSM_EMERGENCY_STOP
            reason = "emergency_stop: pedestrian_in_path"
            return self._emit(0.0, reason)

        if self._fsm_state == self.FSM_EMERGENCY_STOP:
            if self._ped_clear_since is None:
                self._ped_clear_since = now
            held = now - self._ped_clear_since
            if held < self.config.emergency_clear_s:
                reason = f"emergency_stop: clearing ({held:.1f}/{self.config.emergency_clear_s:.0f}s)"
                return self._emit(0.0, reason)
            self._ped_clear_since = None

        # ── Priority 2: STOP_AND_GO (stop sign) ──────────────────────────────
        if scene_state.stop_sign_ahead:
            dist = scene_state.stop_sign_distance_m
            if dist <= self.config.stop_sign_approach_m and not self._stop_sign_served:
                self._fsm_state = self.FSM_STOP_AND_GO
                if ego_speed < 0.3:
                    if self._stopped_at_sign_since is None:
                        self._stopped_at_sign_since = now
                    held = now - self._stopped_at_sign_since
                    if held >= self.config.stop_sign_dwell_s:
                        self._stop_sign_served = True
                    else:
                        reason = f"stop_and_go: dwell {held:.1f}/{self.config.stop_sign_dwell_s:.0f}s"
                        return self._emit(0.0, reason)
                else:
                    self._stopped_at_sign_since = None
                    ramp_speed = min(self.config.cruise_speed_mps, (2.0 * 1.5 * max(dist, 0.1)) ** 0.5)
                    reason = f"stop_and_go: approaching {dist:.1f}m"
                    return self._emit(ramp_speed, reason)
            else:
                if dist > self.config.stop_sign_approach_m + 5.0:
                    self._stop_sign_served = False
        else:
            self._stop_sign_served = False
            self._stopped_at_sign_since = None

        # ── Priority 3: SLOW_FOR_CROSSWALK ───────────────────────────────────
        if (scene_state.crosswalk_ahead
                and scene_state.crosswalk_distance_m >= 0.0
                and scene_state.crosswalk_distance_m < self.config.crosswalk_slow_distance_m):
            self._fsm_state = self.FSM_SLOW_CROSSWALK
            reason = f"slow_for_crosswalk dist={scene_state.crosswalk_distance_m:.1f}m"
            return self._emit(self.config.crosswalk_slow_speed_mps, reason)

        # ── Priority 4: FOLLOW_VEHICLE (ACC) ─────────────────────────────────
        if has_lead and lead_speed < ego_speed + 0.5:
            self._fsm_state = self.FSM_FOLLOW_VEHICLE
            target_speed, reason = self._acc_speed(ego_speed, lead_dist, lead_speed)
            return self._emit(target_speed, reason)

        # ── Default: LANE_FOLLOW ──────────────────────────────────────────────
        self._fsm_state = self.FSM_LANE_FOLLOW
        return self._emit(self.config.cruise_speed_mps, self.FSM_LANE_FOLLOW)

    # ------------------------------------------------------------------
    def _emit(self, target_speed: float, reason: str) -> PlannerOutput:
        # Print only when FSM state or reason changes (suppresses per-tick spam)
        state_changed = self._fsm_state != self._prev_fsm_state
        reason_key = reason.split(":")[0].strip()   # compare prefix, not dynamic values
        prev_key   = self._prev_reason.split(":")[0].strip()
        reason_changed = reason_key != prev_key

        if state_changed or reason_changed:
            print(
                f"[behaviour] fsm={self._fsm_state:<18s} "
                f"tgt={target_speed*3.6:5.1f}km/h  {reason}"
            )
            self._prev_fsm_state = self._fsm_state
            self._prev_reason    = reason

        return PlannerOutput(
            fsm_state=self._fsm_state,
            target_speed_mps=target_speed,
            target_lane_offset=0.0,
            reason=reason,
        )

    def _acc_speed(
        self, ego_speed: float, lead_dist: float, lead_speed: float
    ) -> tuple[float, str]:
        """
        Proportional gap controller with hard minimum floor.
        Desired gap = max(follow_min_gap_m, follow_gap_time_s * ego_speed).
        """
        if lead_dist < self.config.follow_min_gap_m:
            reason = f"follow_vehicle GAP_CRITICAL {lead_dist:.1f}m < {self.config.follow_min_gap_m:.0f}m floor"
            return 0.0, reason

        desired_gap = max(
            self.config.follow_min_gap_m,
            self.config.follow_gap_time_s * max(ego_speed, 1.0),
        )
        gap_error = lead_dist - desired_gap
        acc_speed = lead_speed + self.config.acc_proportional_gain * gap_error
        acc_speed = max(self.config.acc_min_speed_mps, acc_speed)
        acc_speed = min(self.config.cruise_speed_mps, acc_speed)

        reason = f"follow_vehicle gap={lead_dist:.1f}m des={desired_gap:.1f}m"
        return acc_speed, reason
