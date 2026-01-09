import logging
import os
from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class YOLOv11Model(LabelStudioMLBase):
    """YOLOv11 model for traffic light detection"""
    
    def setup(self):
        """Initialize the model"""
        self.set("model_version", "yolov11n")
        
        # Load YOLO model
        model_path = os.environ.get('MODEL_PATH', 'yolo11n.pt')
        self.model = YOLO(model_path)
        logger.info(f"✅ Model loaded: {model_path}")
        
        # Traffic light class in COCO
        self.traffic_light_class = 9
        
    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """Generate predictions"""
        from_name, to_name, value = self.label_interface.get_first_tag_occurence(
            'RectangleLabels', 'Image'
        )
        
        predictions = []
        
        for task in tasks:
            image_url = task['data'].get(value)
            
            if not image_url:
                predictions.append({"result": []})
                continue
            
            # Handle Label Studio local file paths (Windows)
            if '/data/local-files/?d=' in image_url:
                rel_path = image_url.split('/data/local-files/?d=')[1]
                # Convert URL path separators to Windows backslashes
                rel_path = rel_path.replace('/', '\\')
                
                # Get base path from environment
                base_path = os.environ.get('LOCAL_FILES_DOCUMENT_ROOT', '.')
                image_path = os.path.join(base_path, rel_path)
            else:
                image_path = image_url
            
            logger.info(f"🔍 Processing: {image_path}")
            
            try:
                # Run YOLO prediction
                results = self.model.predict(
                    source=image_path,
                    conf=0.25,
                    iou=0.45,
                    classes=[self.traffic_light_class],
                    verbose=False
                )
                
                result = results[0]
                img_height, img_width = result.orig_shape
                
                prediction_result = []
                
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0])
                        
                        prediction_result.append({
                            "original_width": int(img_width),
                            "original_height": int(img_height),
                            "image_rotation": 0,
                            "value": {
                                "x": float(x1 / img_width * 100),
                                "y": float(y1 / img_height * 100),
                                "width": float((x2 - x1) / img_width * 100),
                                "height": float((y2 - y1) / img_height * 100),
                                "rotation": 0,
                                "rectanglelabels": ["traffic_signal"]
                            },
                            "score": confidence,
                            "from_name": from_name,
                            "to_name": to_name,
                            "type": "rectanglelabels"
                        })
                
                predictions.append({
                    "result": prediction_result,
                    "score": sum(r['score'] for r in prediction_result) / len(prediction_result) if prediction_result else 0,
                    "model_version": self.get("model_version")
                })
                
                logger.info(f"✅ Found {len(prediction_result)} traffic signals")
                
            except Exception as e:
                logger.error(f"❌ Error: {str(e)}", exc_info=True)
                predictions.append({"result": []})
        
        return ModelResponse(predictions=predictions)
    
    def fit(self, event, data, **kwargs):
        """Optional: training implementation"""
        return {"status": "ok"}