### [In progress]
## 🚗 AutonomousCar - Building an Autonomous Vehicle Block by Block in CARLA

Documenting my project journey here:
[**Building an Autonomous Vehicle Block by Block**](https://medium.com/@yashphalle/building-an-autonomous-vehicle-block-by-block-d7128d564094)


## Repository Structure

```
AutonomousCar/
├── main.py                 # Entry point: spawns ego, plans route, runs PID control loop
├── config.py               # Central tunables: town, vehicle, route, speed, PID gains
│
├── vehicle/
│   └── autonomous_vehicle.py   # AutonomousVehicle: connect to CARLA, spawn ego, attach sensors
│
├── planning/
│   ├── route_planner.py        # Wraps CARLA's GlobalRoutePlanner
│   └── waypoint_manager.py     # Tracks current target waypoint, handles slow-down near goal
│
├── control/
│   └── pid_controller.py       # PID (used for both steering and longitudinal control)
│
├── utils/
│   ├── carla_bootstrap.py      # Locates CARLA install and adds PythonAPI to sys.path
│   ├── carla_utils.py          # Ego state, angle normalization, throttle/brake mapping, HUD helpers
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
├── extras/                 # Standalone scripts (not part of main loop)
│   ├── camera_feed.py              # Multi-camera + LiDAR top-down visualizer
│   ├── car_control.py              # Minimal manual drive script
│   └── collect_traffic_light_data.py # Dataset collection for CV training
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


> 🤖⚡**Claude Code** has been greatly accelerating the development of this project!