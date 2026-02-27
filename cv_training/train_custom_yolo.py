#!/usr/bin/env python3

import argparse
from pathlib import Path
from ultralytics import YOLO
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_model(epochs: int = 100):
    """
    Train YOLO model for traffic signal detection
    
    Hardcoded settings:
    - Model: YOLOv11x (extra large, best accuracy)
    - Dataset: yolo_dataset/data.yaml
    - Batch size: 16
    - Image size: 640
    - Device: auto-detect
    """
    
    # Hardcoded paths and settings
    DATA_YAML = "yolo_dataset/data.yaml"
    MODEL_SIZE = "x"  # Extra large for best accuracy
    BATCH_SIZE = 16
    IMG_SIZE = 640
    PROJECT_DIR = "runs/train"
    EXP_NAME = "traffic_signal"

    logger.info("YOLO TRAINING - Traffic Signal Detection")

    
    # Check dataset
    data_path = Path(DATA_YAML)
    if not data_path.exists():
        logger.error(f"ERROR: data.yaml not found at: {DATA_YAML}")
        logger.error("Please ensure yolo_dataset/data.yaml exists")
        raise FileNotFoundError(f"Dataset not found: {DATA_YAML}")
    
    logger.info(f"Dataset: {DATA_YAML}")
    logger.info(f"Model: YOLOv11{MODEL_SIZE} (extra large - best accuracy)")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info(f"Image size: {IMG_SIZE}")
    logger.info(f"Output: {PROJECT_DIR}/{EXP_NAME}")
    logger.info("="*70)
    
    # Load model
    model_name = f'yolo11{MODEL_SIZE}.pt'
    logger.info(f"\n[1/3] Loading pretrained model: {model_name}")
    try:
        model = YOLO(model_name)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    # Verify dataset structure
    logger.info(f"\n[2/3] Verifying dataset structure...")
    dataset_path = data_path.parent
    train_images = len(list((dataset_path / "images" / "train").glob("*.png")))
    val_images = len(list((dataset_path / "images" / "val").glob("*.png")))
    train_labels = len(list((dataset_path / "labels" / "train").glob("*.txt")))
    val_labels = len(list((dataset_path / "labels" / "val").glob("*.txt")))
    
    logger.info(f"  Train images: {train_images}")
    logger.info(f"  Val images: {val_images}")
    logger.info(f"  Train labels: {train_labels}")
    logger.info(f"  Val labels: {val_labels}")
    
    if train_images == 0 or val_images == 0:
        logger.error("ERROR: No images found in dataset!")
        raise ValueError("Dataset is empty")
    
    if train_images != train_labels or val_images != val_labels:
        logger.warning("WARNING: Image-label count mismatch!")
    
    logger.info("Dataset structure OK")
    
    # Start training
    logger.info(f"\n[3/3] Starting training...")
    logger.info("-"*70)
    
    try:
        results = model.train(
            data=str(data_path),
            epochs=epochs,
            batch=BATCH_SIZE,
            imgsz=IMG_SIZE,
            device=None,  # Auto-detect
            project=PROJECT_DIR,
            name=EXP_NAME,
            verbose=True
        )
        
        logger.info("-"*70)
        logger.info("="*70)
        logger.info("TRAINING COMPLETE!")
        logger.info("="*70)
        logger.info(f"Best model: {results.save_dir}/weights/best.pt")
        logger.info(f"Last model: {results.save_dir}/weights/last.pt")
        logger.info(f"Results directory: {results.save_dir}")
        logger.info("="*70)
        
        return results
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise


def main():
    parser = argparse.ArgumentParser(description='Train YOLO model for traffic signals')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    
    args = parser.parse_args()
    
    train_model(epochs=args.epochs)


if __name__ == '__main__':
    main()
