# AutonomousCar — Claude Project Guide

End-to-end autonomous vehicle stack in CARLA, built modularly bottom-up.
Python 3.10 · CARLA 0.9.16 · Town01

---

## Stage Status

| Stage | Status | Key Files |
|---|---|---|
| **1 — Vehicle / Sensor scaffolding** | ✅ Complete | `vehicle/autonomous_vehicle.py`, `main.py`, `control/`, `planning/route_planner.py`, `planning/waypoint_manager.py` |
| **2 — SceneState + GT perception + rule-based planner** | ✅ Complete | `perception/`, `planning/planner.py`, `eval/` |
| **2b — Behaviour planning (FSM)** | ✅ Complete | `planning/behaviour_planner.py`, `planning/planner_output.py`, `planning/lane_aware_waypoint_manager.py`, `scenes/` |
| **3a — Localization** | ⬜ Not started | — |
| **3b — Learned perception (YOLO TL)** | 🟡 Early | `cv_training/` |
| **4 — Full integration** | ⬜ Not started | — |

**Current focus: behaviour planner extensions.** Perception swap (Stage 3b) is deferred until the very end — at that point we may either use GT as annotation for a learned model, or skip modular perception and try end-to-end neural nets.

---

## Architectural Contracts (load-bearing — never violate)

1. **`SceneState` is the ONLY interface between perception and planning.**
   Perception writes it. Planner reads it. No other data crosses this boundary.

2. **`planning/planner.py` never imports `carla`.** It reads only `SceneState`.

3. **`perception/gt_perception.py` never calls the planner.** It only produces `SceneState`.

4. **Stage 1 modules are FROZEN** (no edits without explicit approval):
   - `control/pid_controller.py`
   - `planning/route_planner.py`
   - `planning/waypoint_manager.py`
   - `utils/carla_utils.py`

5. **Unit conventions — enforce at the boundary, never inside:**
   - Speeds: **m/s** internally. Convert km/h → m/s in `gt_perception.py` only.
   - Angles: **radians** internally. Convert CARLA degrees → rad in `gt_perception.py` only.
   - Distances: **metres** everywhere.
   - Conversions in logging/HUD only (SI → human-readable).

6. **Config lives in `config.py`.** Never scatter constants across files.

7. **CARLA coordinate system — left-handed Z-up:**
   - Forward vector: `fwd_x = cos(yaw_rad)`, `fwd_y = sin(yaw_rad)` — **NO Y negation**.
   - "Ahead" check: `dot = dx * fwd_x + dy * fwd_y > 0`.
   - Y negation is **only** for Rerun visualisation output, never for geometry calculations.
   - Recurring bug: applying Y negation in geometry causes vehicles in adjacent lanes to appear as lead vehicles on curves.

9. **Behaviour planner feeds the safety planner — safety always wins:**
   ```python
   planner_out = behaviour_planner.plan(scene_state)        # judgment layer
   planner.config.cruise_speed_mps = planner_out.target_speed_mps  # ACC ceiling
   speed_profile = planner.plan(scene_state)                # safety layer runs on top
   ```
   The behaviour planner sets a desired cruise speed; the safety planner may lower it further (red light, emergency brake). Never the reverse.

10. **CARLA API gotcha — Traffic Manager is on the client:**
    ```python
    tm = client.get_trafficmanager()   # CORRECT for CARLA 0.9.16
    # world.get_trafficmanager()       # AttributeError — does not exist
    ```

11. **Always add a debug/log line when implementing a new feature.** Use `print(f"[tag] ...")` for planner decisions, `RunLogger` for per-frame CSV metrics.

---

## Multi-Agent Workflow

This project uses specialized subagents. Spawn them via the `Agent` tool with `subagent_type`.
Each agent owns its file(s) — do not have agents edit files outside their ownership.

| Agent | `subagent_type` | Owns | Responsibility |
|---|---|---|---|
| SceneState Architect | `scene-state-architect` | `perception/scene_state.py` | Dataclass contract only. No logic, no CARLA imports. |
| GT Perception Engineer | `gt-perception-engineer` | `perception/gt_perception.py` | Reads CARLA ground truth → writes `SceneState`. All unit conversions live here. |
| Planner Engineer | `planner-engineer` | `planning/planner.py` | Rule-based planner: reads `SceneState` → outputs `SpeedProfile`. Never imports `carla`. |
| Behaviour Planner Engineer | `behaviour-planner-engineer` | `planning/behaviour_planner.py`, `planning/planner_output.py`, `planning/lane_aware_waypoint_manager.py` | FSM-based behaviour planner: reads `SceneState` → outputs `PlannerOutput`. Never imports `carla`. |
| Eval Engineer | `eval-engineer` | `eval/run_eval.py`, `eval/routes.py` | Synchronous seeded harness, per-route metrics, CSV + summary output. |
| Explorer | `Explore` | — | Fast read-only codebase search. Use for "where is X defined?" queries. |
| Planner | `Plan` | — | Architecture design before implementation. Use before any non-trivial change. |

### When to spawn agents

- **Spawn in parallel** when tasks are file-disjoint (e.g. planner + eval can build simultaneously).
- **Spawn sequentially** when one agent's output is another's input (e.g. SceneState must be final before GT Perception updates).
- **Don't spawn** for single-file edits or quick bug fixes — handle inline.
- Always give agents the full spec + current file contents — they start cold with no conversation context.

---

## Key Interfaces

### `SceneState` (v2) — `perception/scene_state.py`
```
SceneState
  .timestamp_s         float
  .ego_pose            EgoPose  (x, y, z, yaw_rad, speed_mps)
  .route_waypoints     list[WaypointInfo]   # next ~50 m, parallel to SpeedProfile
  .route_start_idx     int                  # absolute index into full route
  .traffic_lights      list[TrafficLightInfo]  # lane-filtered, relevant only
  .stop_signs          list[StopSignInfo]
  .dynamic_objects     list[DetectedObject]    # within 60 m, ego excluded
  .local_map           OccupancyGrid | None    # Stage 3e
```

### `SpeedProfile` — `planning/planner.py`
```
SpeedProfile
  .target_speeds_mps   list[float]   # parallel to SceneState.route_waypoints
  .reason              str           # primary rule that fired this tick
```

Planner reason values: `cruise` · `speed_limit` · `junction_speed_cap` · `lead_vehicle` · `yellow_stop` · `yellow_continue` · `red_light_stop` · `emergency_brake`

### `GTPerception.update()` signature
```python
def update(self, full_route: list[tuple[carla.Waypoint, RoadOption]], current_idx: int) -> SceneState
```

---

## Planner Rule Priority (highest wins)

| Priority | Rule | Trigger |
|---|---|---|
| 0 | `emergency_brake` | Any object within `emergency_brake_radius_m` (5 m) **AND** within ±20° forward cone |
| 1 | `red_light_stop` | RED light on ego's upcoming lane — physics ramp `v = sqrt(2·a·d)` to stop line |
| 2 | `yellow_stop` / `yellow_continue` | Physics check: `v²/2a < dist` → stop, else continue |
| 3 | Stop sign | Stub — field exists, logic deferred to Stage 3 |
| 4 | `lead_vehicle` | Nearest vehicle within 10 m and ±20° cone |
| 5 | `junction_speed_cap` | Junction waypoint within 15 m → cap at `junction_speed_mps` |
| 6 | `speed_limit` | Per-waypoint: `min(cruise, wp.speed_limit_mps)` |
| 7 | `cruise` | `config.TARGET_SPEED` baseline |

**Known limitation:** lead vehicle filter uses bearing angle only (no lane-aware projection). Vehicles in adjacent lanes can enter the cone on curves. Map-aware fix deferred to Stage 3.

---

## Running the Stack

### Prerequisites
```bash
# CARLA must be running first:
~/Downloads/CARLA_0.9.16/CarlaUE4.sh

# Install CARLA Python bindings (one-time):
python3.10 -m pip install ~/Downloads/CARLA_0.9.16/PythonAPI/carla/dist/carla-0.9.16-cp310-cp310-manylinux_2_31_x86_64.whl

# Always use python3.10 (not python3 / conda base which is 3.13):
python3.10 main.py
```

### Main loop
```bash
python3.10 main.py              # runs spawn[0] → spawn[50] on Town01
```
Terminal prints every 1 s:
```
[ 1200] red_light_stop    spd= 0.0km/h tgt= 0.0km/h wp= 481  TL RED 3.4m  OBJ none
```

### Eval harness
```bash
python3.10 -m eval.run_eval --seed 42 --routes all
python3.10 -m eval.run_eval --seed 42 --routes 1,3,5   # subset
```
Output: `eval/results/baseline_gt_<timestamp>/per_route.csv` + `summary.txt`

### Stage 2 baseline (reference)
Recorded: `eval/results/baseline_gt_20260509_184806/`
All Stage 3 perception swaps are measured against this baseline.

---

## Eval Routes (`eval/routes.py`)

10 routes on Town01. Spawn indices are approximate — verify with `extras/map_overlay.py`.

| # | Label | Start → End | Distance |
|---|---|---|---|
| 1 | long_straight_low_traffic | 0 → 50 | 400 m |
| 2 | long_straight_opposite | 50 → 0 | 400 m |
| 3 | multi_junction | 10 → 80 | 350 m |
| 4 | single_traffic_light | 20 → 30 | 200 m |
| 5 | cross_town_diagonal | 5 → 90 | 600 m |
| 6 | dense_npc_cluster | 15 → 40 | 300 m |
| 7 | bridge_speed_limit | 60 → 70 | 250 m |
| 8 | left_turn_opposing_traffic | 25 → 35 | 200 m |
| 9 | long_endurance | 0 → 100 | 1000 m |
| 10 | short_loop | 45 → 48 | 100 m |

---

## Stage 2 Definition of Done ✅

- [x] `eval/run_eval.py` completes all 10 routes in synchronous mode with seeded NPC traffic
- [x] Baseline CSV + summary saved under `eval/results/baseline_gt_*/`
- [x] Zero red-light violations across all routes
- [x] Zero planner-fault collisions
- [x] Reason histogram non-degenerate (multiple planner branches exercised)

---

## Stage 3b Plan (next)

Goal: replace `TrafficLightInfo.state` (currently from CARLA GT) with YOLO classifier output.

```
perception/
  scene_state.py          # frozen
  gt_perception.py        # frozen (becomes fallback/reference)
  learned_perception.py   # NEW — reads camera frames, writes same SceneState shape
cv_training/              # YOLO training pipeline (early work exists)
```

Interface contract: `LearnedPerception` must implement the same `.update(full_route, current_idx) -> SceneState` signature as `GTPerception`. Planner and eval harness are unchanged.

Measurement: run `eval/run_eval.py` with `LearnedPerception` and diff the CSV against `baseline_gt_*`.

---

## File Layout

```
main.py              Entry point — wires perception → planner → waypoint manager → PID
config.py            All constants live here
vehicle/             Stage 1 — ego spawn, sensor suite (8 cameras + LiDAR)
planning/
  route_planner.py   FROZEN — A* route from start to end
  waypoint_manager.py FROZEN (+set_speed_profile added in Stage 2)
  planner.py         Stage 2 — RuleBasedPlanner, SpeedProfile, PlannerConfig
control/
  pid_controller.py  FROZEN — lateral + longitudinal PID
perception/
  scene_state.py     Stage 2 — dataclass contract (no CARLA imports)
  gt_perception.py   Stage 2 — CARLA GT → SceneState
eval/
  run_eval.py        Stage 2 — synchronous seeded harness
  routes.py          Stage 2 — 10 evaluation routes
  results/           Baseline CSVs (gitignored if large)
utils/
  carla_bootstrap.py Auto-locate CARLA install, patch sys.path
  carla_utils.py     FROZEN — helpers (synchronous_mode, get_ego_state, etc.)
  run_logger.py      Per-frame CSV logger
  logs/              Runtime logs
cv_training/         Stage 3b — YOLO TL classifier training
extras/              Standalone scripts (map_overlay, explore_carla_state)
```
