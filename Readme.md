### [In progress]
## AutonomousCar – Building End‑to‑End Autonomous Vehicle in Simulation

This repository contains my work on building an **autonomous vehicle stack in CARLA**, 
Documenting project journey here:  
[**Building an Autonomous Vehicle Block by Block**](https://medium.com/@yashphalle/building-an-autonomous-vehicle-block-by-block-d7128d564094)

---

## Repository Overview


### 1. CARLA Simulation and Vehicle Setup

This part is responsible for **spawning the world, the ego vehicle, and sensors**, and for driving it around in the simulator.

files:

- `vehicle/autonomous_vehicle.py`  
`AutonomousVehicle` class for:
  - Connecting to CARLA
  - Loading a town
  - Spawning the ego vehicle
  - Attaching cameras and LiDAR


- `car_control.py`  
  Small script to control the car (e.g. drive forward a fixed distance, stop).  
  Useful as a building block for more advanced planners.

- `camera_feed.py`  
  Visualizes the **sensor suite**:
  - Multiple camera feeds as thumbnails
  - A top‑down LiDAR projection
  - Overlays simple HUD info such as distance traveled


---

### 2. Computer Vision 

The CV part focuses on **detecting traffic signals** from the front cameras for now.

Key pieces (under `cv_training/` and `cv_training/label_studio_autolabel/`):

- Data collection uses the CARLA scripts above to generate front‑camera images at multiple ranges.
- Labeling & autolabeling:
  - Label Studio is used for annotating traffic signals.
  - A small FastAPI service (`start_backend.py`) lets Label Studio call a VLM/ YOLO model and get predictions as suggestions which we can use for faster annotation.
  - `autolabel_traffic_signals.py` can bulk‑autolabel images stored by Label Studio using a YOLO model, and write YOLO‑format labels plus Label Studio‑compatible JSON.

- Training:
  - `cv_training/yolo_dataset/` contains a YOLO‑ready dataset:
    - `images/train`, `images/val`
    - `labels/train`, `labels/val`
    - `data.yaml` – dataset configuration
  - `cv_training/train_custom_yolo.py` is a **minimal training script** that:
    - Loads a YOLOv11 model (configurable size)
    - Trains it on `yolo_dataset/data.yaml`
    - Writes results into `runs/train/traffic_signal*/`

More CV Detection Parts to be added.
