import os
import logging
from pathlib import Path
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
PROJECT_ROOT = Path(r'D:\projects\AutonomousCar')
os.environ['LOCAL_FILES_DOCUMENT_ROOT'] = str(PROJECT_ROOT)

# Imports
from ultralytics import YOLO
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn


# Simple YOLO predictor (no Label Studio ML base class)
class YOLOPredictor:
    def __init__(self):
        self.model = YOLO('yolo11x.pt')
        logger.info("✅ YOLO model loaded successfully")
    
    def predict_image(self, image_path: str, from_name: str, to_name: str):
        """Run YOLO prediction and format for Label Studio"""
        
        logger.info(f"🔍 Processing: {image_path}")
        
        try:
            # Run YOLO
            results = self.model.predict(
                source=image_path,
                conf=0.25,
                iou=0.45,
                classes=[9],  # Traffic light in COCO
                verbose=False
            )
            
            result = results[0]
            img_h, img_w = result.orig_shape
            
            # Build prediction results
            pred_result = []
            
            if result.boxes is not None and len(result.boxes) > 0:
                for idx, box in enumerate(result.boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    
                    pred_result.append({
                        "id": f"result_{idx}",
                        "original_width": int(img_w),
                        "original_height": int(img_h),
                        "image_rotation": 0,
                        "value": {
                            "x": float(x1 / img_w * 100),
                            "y": float(y1 / img_h * 100),
                            "width": float((x2 - x1) / img_w * 100),
                            "height": float((y2 - y1) / img_h * 100),
                            "rotation": 0,
                            "rectanglelabels": ["traffic_signal"]
                        },
                        "score": conf,
                        "from_name": from_name,
                        "to_name": to_name,
                        "type": "rectanglelabels"
                    })
            
            logger.info(f"✅ Found {len(pred_result)} traffic signals")
            return pred_result
            
        except Exception as e:
            logger.error(f"❌ Error: {e}", exc_info=True)
            return []


# Create FastAPI app
app = FastAPI(
    title="YOLO ML Backend",
    description="YOLOv11 Traffic Light Detection",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor
predictor = YOLOPredictor()


def parse_label_config(label_config: str):
    """Extract from_name and to_name from label config"""
    import re
    
    # Find RectangleLabels tag
    rect_match = re.search(r'<RectangleLabels\s+name="([^"]+)"\s+toName="([^"]+)"', label_config)
    if rect_match:
        from_name = rect_match.group(1)
        to_name = rect_match.group(2)
    else:
        from_name = "label"
        to_name = "image"
    
    # Find Image tag value
    img_match = re.search(r'<Image\s+name="[^"]+"\s+value="\$([^"]+)"', label_config)
    if img_match:
        value = img_match.group(1)
    else:
        value = "image"
    
    return from_name, to_name, value


@app.get("/")
def root():
    return {
        "status": "running",
        "model": "YOLOv11",
        "version": "1.0.0"
    }


@app.get("/health")
def health():
    return {"status": "UP", "model_version": "yolov11n"}


@app.post("/setup")
async def setup(request: Request):
    """Setup endpoint"""
    return {
        "model_version": "yolov11n",
        "status": "ok"
    }


@app.post("/predict")
async def predict(request: Request):
    """Main prediction endpoint"""
    data = await request.json()
    
    tasks = data.get('tasks', [])
    label_config = data.get('label_config', '')
    
    logger.info(f"📥 Received {len(tasks)} tasks")
    
    # Parse label config
    from_name, to_name, value = parse_label_config(label_config)
    
    predictions = []
    
    for task in tasks:
        # Get image URL
        image_url = task.get('data', {}).get(value, '')
        
        if not image_url:
            logger.warning("No image URL found")
            predictions.append({"result": [], "score": 0})
            continue
        
        # Handle Label Studio local file paths
        if '/data/local-files/?d=' in image_url:
            rel_path = image_url.split('/data/local-files/?d=')[1]
            rel_path = rel_path.replace('/', '\\')  # Windows
            image_path = os.path.join(
                os.environ['LOCAL_FILES_DOCUMENT_ROOT'],
                rel_path
            )
        elif image_url.startswith('/data/upload/'):
            # Convert label-studio URL path to Windows file path
            import os as os_module
            label_studio_base = Path(os_module.environ.get('LOCALAPPDATA')) / 'label-studio' / 'label-studio'
            
            # Remove /data/upload prefix and convert to Windows path
            rel_path = image_url.replace('/data/upload/', '').replace('/', '\\')
            image_path = str(label_studio_base / 'media' / 'upload' / rel_path)
            
            logger.info(f"🔄 Converted URL to path: {image_url} -> {image_path}")
        else:
            image_path = image_url
        
        # Get predictions
        result = predictor.predict_image(image_path, from_name, to_name)
        
        # Calculate average score
        avg_score = sum(r['score'] for r in result) / len(result) if result else 0
        
        predictions.append({
            "result": result,
            "score": avg_score,
            "model_version": "yolov11x"
        })
    
    response = {
        "results": predictions
    }
    
    logger.info(f"📤 Returning {len(predictions)} predictions")
    
    return response


@app.post("/webhook")
async def webhook(request: Request):
    """Webhook endpoint"""
    return {"status": "ok"}


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🚀 YOLO ML Backend Server")
    print("=" * 70)
    print(f"📂 Project: {PROJECT_ROOT}")
    print(f"🖼️  Images: {PROJECT_ROOT / 'images'}")
    print(f"🤖 Model: YOLOv11n (COCO pretrained)")
    print(f"🎯 Target: Traffic lights (class 9)")
    print(f"🌐 Server: http://localhost:9090")
    print(f"🔗 Health: http://localhost:9090/health")
    print("=" * 70)
    print("📝 Endpoints:")
    print("   GET  /         - Server info")
    print("   GET  /health   - Health check")
    print("   POST /setup    - Setup model")
    print("   POST /predict  - Get predictions")
    print("=" * 70)
    print("\n✅ Server starting...\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=9090,
        log_level="info"
    )