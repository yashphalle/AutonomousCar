## Label Studio Autolabel Setup

This folder contains the code used to **autolabel traffic signal images in Label Studio using YOLO**.

### Files

- `autolabel_traffic_signals.py`  
  Standalone script that:
  - Scans a Label Studio storage directory (e.g. `LOCALAPPDATA/label-studio/label-studio/media/upload/<project_id>`)
  - Runs a YOLO model (YOLOv11 by default) on each image
  - Writes:
    - YOLO format labels (`annotations/<project_name>/labels/*.txt`)
    - Label Studio JSON annotations (`annotations/<project_name>/annotations/*.json`)
    - Summary files (`summary.json`, `image_paths.txt`, `filename_mapping.json`)

- `start_backend.py`  
  FastAPI service that acts as a **Label Studio ML backend**:
  - Receives tasks from Label Studio on `/predict`
  - Converts Label Studio image URLs (`/data/local-files/?d=...` or `/data/upload/...`) to local file paths
  - Runs YOLO and returns predictions in Label Studio format (`rectanglelabels: ["traffic_signal"]`)

- `yolo_model.py`  
  Alternative ML backend implementation using `LabelStudioMLBase`. Use this if you run the official
  Label Studio ML backend container instead of the simple FastAPI server in `start_backend.py`.

---

### Requirements

Install dependencies (on the machine running this code):

```bash
pip install ultralytics fastapi uvicorn label-studio-ml
```

You also need a YOLO model file available, for example `yolo11x.pt` or `yolo11n.pt`.

---

### 1. Running the standalone autolabel script

1. Make sure Label Studio has already ingested your images and they are stored under
   `LOCALAPPDATA/label-studio/label-studio/media/upload/<project_id>`.

2. From the repo root (or from this folder), run:

```bash
python autolabel_traffic_signals.py \
  --label-studio-path "C:\Users\<USER>\AppData\Local\label-studio\label-studio" \
  --project-name "Carla Traffic Signal Detection" \
  --output-dir "./annotations/Carla Traffic Signal Detection" \
  --model yolo11x.pt
```

Key points:

- `--label-studio-path` (optional): base Label Studio directory; if omitted, it defaults to `LOCALAPPDATA/label-studio/label-studio` on Windows.
- `--project-name`: used only for naming the output folder.
- `--output-dir`: where YOLO labels and Label Studio JSONs will be written.
- `--model`: which YOLO weights to use (any model supported by `ultralytics.YOLO`).

The script will print:

- Where it found images (source directory)
- How many images were processed
- How many images had detections
- Paths to the generated summary and mapping files

You can then import the generated JSON annotations back into Label Studio if needed.

---

### 2. Running the FastAPI ML backend (`start_backend.py`)

This backend lets Label Studio request predictions directly from YOLO.

1. Start the backend server:

```bash
python start_backend.py
```

By default it listens on:

- `http://localhost:9090/`

2. In Label Studio, configure an ML backend pointing to this URL (e.g. via **Settings → Machine Learning**):

- URL: `http://host.docker.internal:9090` (if Label Studio is in Docker)
- or: `http://localhost:9090` (if both are on the host)

3. In your project’s labeling config, ensure you have a single rectangle label named `traffic_signal`, for example:

```xml
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="traffic_signal" background="red"/>
  </RectangleLabels>
</View>
```

4. In the **Labeling Interface**, choose your ML backend and click **Predict** or enable automatic pre-annotation.

The backend will:

- Convert Label Studio URLs to local paths
- Run YOLO
- Return bounding boxes labeled as `traffic_signal`

---

### Note
- `autolabel_traffic_signals.py` is intended to be run offline to bulk‑annotate images; `start_backend.py` is for online, per‑task predictions inside Label Studio.

