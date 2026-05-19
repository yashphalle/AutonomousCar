### [In progress]
## 🚗 AutonomousCar - Building an Autonomous Vehicle Block by Block in CARLA

Documenting my project journey here:
[**Building an Autonomous Vehicle Block by Block**](https://medium.com/@yashphalle/building-an-autonomous-vehicle-block-by-block-d7128d564094)


## Progress

| Part | Title | Key Result |
|---|---|---|
| Part 1 | CARLA Setup & Sensor Suite | `AutonomousVehicle` class, 7-camera + LiDAR rig, real-time feed |
| Part 2 | Teaching the Car to See Traffic Lights | YOLOv11 detector, mAP50 = 0.76, precision = 0.89, free auto-labeling pipeline |
| Part 3 | Building the Control Stack | Two-loop PID controller, speed-scaled lookahead, stable route following to 50 km/h |
| Part 4 | Teaching the Car to Take Decisions | Rule-based planner with `SceneState` architecture — 0 collisions, 0 red-light violations |
| Part 5 | Seeing What the Car Sees | Real-time Rerun visualizer — 3D world view, BEV, live speed + planner-rule time-series |

**Up next:** Extending the behaviour planner — Town03 stop-sign validation, full gauntlet runs, and Phase 3 lane change on multi-lane maps. Perception swap is deferred until the very end of the project (we may use the CARLA GT API as an annotation source for a learned model, or skip modular perception entirely in favour of end-to-end neural nets).


## Demos

<table>
  <tr>
    <td align="center" width="50%">
      <a href="https://www.youtube.com/watch?v=Zp2v0woXDkI"><img src="https://img.youtube.com/vi/Zp2v0woXDkI/hqdefault.jpg" width="400" alt="Camera + LiDAR + Controls"></a><br>
      <b>Part 1 - CARLA Setup & Sensor Suite</b><br>
      <sub>Basic vehicle controls + multi-camera and LiDAR feed</sub>
    </td>
    <td align="center" width="50%">
      <a href="https://medium.com/@yashphalle/teaching-the-car-to-see-traffic-lights-building-an-av-block-by-block-part-2-c6166d894c27"><img src="media/yolo_traffic_lights.png" width="400" alt="YOLO Traffic Lights"></a><br>
      <b>Part 2 - Teaching the Car to See Traffic Lights</b><br>
      <sub>YOLOv11 detector + auto-labeling pipeline</sub>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <a href="https://youtu.be/kcWt94IUlJI"><img src="https://img.youtube.com/vi/kcWt94IUlJI/hqdefault.jpg" width="400" alt="PID Waypoint Follower"></a><br>
      <b>Part 3 - PID Waypoint Follower</b><br>
      <sub>Lateral + longitudinal PID, stable to 50 km/h</sub>
    </td>
    <td align="center" width="50%">
      <a href="https://youtu.be/axbufiYNJZU"><img src="https://img.youtube.com/vi/axbufiYNJZU/hqdefault.jpg" width="400" alt="Rule-Based Planner"></a><br>
      <b>Part 4 - Rule-Based Planner</b><br>
      <sub>SceneState + planner running autonomous routes</sub>
    </td>
  </tr>
  <tr>
    <td align="center" colspan="2">
      <a href="https://youtu.be/GMCpoKJ3ty0"><img src="https://img.youtube.com/vi/GMCpoKJ3ty0/hqdefault.jpg" width="400" alt="BEV Visualizer"></a><br>
      <b>Part 5 - BEV Visualizer</b><br>
      <sub>Real-time Rerun visualizer with ego-centric BEV</sub>
    </td>
  </tr>
</table>

## Repository Structure

```
AutonomousCar/
├── main.py                 # Entry point: wires perception → behaviour → planner → control
├── config.py               # Central tunables: town, vehicle, route, speed, PID gains
│
├── vehicle/
│   └── autonomous_vehicle.py   # AutonomousVehicle: connect to CARLA, spawn ego, attach sensors
│
├── perception/
│   ├── scene_state.py          # Dataclass contract between perception and planning (no CARLA)
│   └── gt_perception.py        # Reads CARLA ground truth → writes SceneState
│
├── planning/
│   ├── route_planner.py            # Wraps CARLA's GlobalRoutePlanner (A* route)
│   ├── waypoint_manager.py         # Tracks current target waypoint, slowdown near goal
│   ├── lane_aware_waypoint_manager.py  # Lane-aware variant used by the behaviour planner
│   ├── behaviour_planner.py        # FSM behaviour layer: SceneState → PlannerOutput
│   ├── planner_output.py           # PlannerOutput dataclass (target speed + behaviour state)
│   └── planner.py                  # Rule-based safety planner: SceneState → SpeedProfile
│
├── control/
│   └── pid_controller.py       # PID (used for both steering and longitudinal control)
│
├── eval/
│   ├── run_eval.py             # Synchronous seeded harness — runs all routes, writes CSV
│   ├── routes.py               # 10 Town01 evaluation routes
│   └── results/                # Baseline CSVs + per-route summaries
│
├── utils/
│   ├── carla_bootstrap.py      # Locates CARLA install and adds PythonAPI to sys.path
│   ├── carla_utils.py          # Ego state, angle normalization, throttle/brake mapping
│   └── run_logger.py           # Per-frame CSV logger for offline analysis
│
├── cv_training/
│   ├── train_custom_yolo.py    # YOLOv11 training script for traffic signal detection
│   ├── yolo_dataset/           # YOLO-format dataset (images/, labels/, data.yaml)
│   └── label_studio_autolabel/
│       ├── start_backend.py            # FastAPI ML backend for Label Studio
│       ├── yolo_model.py               # YOLO inference wrapper
│       └── autolabel_traffic_signals.py # Bulk autolabeler → YOLO + Label Studio JSON
│
├── viz.py                  # Rerun BEV live visualizer (lanes, traffic lights, signs)
│
├── scenes/
│   ├── definitions.py          # Named test scenes (stop sign, junctions, gauntlet, etc.)
│   └── spawner.py              # Spawns NPCs / actors for a given scene
│
├── extras/                 # Standalone helper scripts (manual drive, dataset collection, plots)
│
└── logs/                   # Run CSVs written by RunLogger
```

---

## Module Overview

### 1. Simulation & Vehicle ([vehicle/](vehicle/))
[vehicle/autonomous_vehicle.py](vehicle/autonomous_vehicle.py) — `AutonomousVehicle` connects to CARLA, loads a town, spawns the ego vehicle, and attaches the camera + LiDAR suite.

### 2. Planning ([planning/](planning/))
- [planning/route_planner.py](planning/route_planner.py) —  wrapper over CARLA's `GlobalRoutePlanner` to compute a route between two world locations.
- [planning/waypoint_manager.py](planning/waypoint_manager.py) — advances through the route, exposes the current target waypoint, and computes a target speed that tapers down near the goal.

### 3. Control ([control/](control/))
[control/pid_controller.py](control/pid_controller.py) — generic PID with integral clamping. `main.py` instantiates two: one for steering (heading error) and one for longitudinal control (speed error).

### 4. Main loop ([main.py](main.py))
Bootstraps CARLA, spawns the ego via `AutonomousVehicle`, plans a route from `START_SPAWN_IDX` → `END_SPAWN_IDX`, then ticks at `DT` running:
`get_ego_state` → `WaypointManager.update` → heading/speed PID → `output_to_throttle_brake` → apply control → log frame.

### 5. Utilities ([utils/](utils/))
- [utils/carla_bootstrap.py](utils/carla_bootstrap.py) — locates the CARLA install (`CARLA_ROOT` env or common paths) and makes `agents.*` importable.
- [utils/carla_utils.py](utils/carla_utils.py) — `EgoState` dataclass, `normalize_angle`, throttle/brake mapping, waypoint debug-draw helpers.
- [utils/run_logger.py](utils/run_logger.py) — writes per-frame CSV (ego pose, target, errors, control outputs) into [logs/](logs/).

### 6. Computer Vision ([cv_training/](cv_training/))
Currently focused on **traffic signal detection** from the front camera.
- Dataset collected via [extras/collect_traffic_light_data.py](extras/collect_traffic_light_data.py).
- Annotation in Label Studio, with a FastAPI ML backend ([cv_training/label_studio_autolabel/start_backend.py](cv_training/label_studio_autolabel/start_backend.py)) serving YOLO predictions as suggestions for faster labeling.
- Bulk autolabeling via [cv_training/label_studio_autolabel/autolabel_traffic_signals.py](cv_training/label_studio_autolabel/autolabel_traffic_signals.py).
- Training via [cv_training/train_custom_yolo.py](cv_training/train_custom_yolo.py) on the YOLO-format dataset under [cv_training/yolo_dataset/](cv_training/yolo_dataset/).

### 7. Extras ([extras/](extras/))
Standalone exploratory scripts kept out of the main loop: manual driving, multi-camera/LiDAR visualization, and CV data collection.

---

## Running

```bash
# 1. Start the CARLA server (separate terminal)
./CarlaUE4.sh

# 2. Run the autonomous driving loop
python main.py
```


> 🤖⚡**Claude Code - Multi Agent setup** has been greatly accelerating the development of this project!