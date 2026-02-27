#!/usr/bin/env python3

import os
import json
from pathlib import Path
from collections import defaultdict
import cv2
from ultralytics import YOLO
from tqdm import tqdm


class TrafficSignalAutolabeler:
    def __init__(self, 
                 label_studio_base=None, 
                 project_name="Carla Traffic Signal Detection",
                 output_dir=None,
                 model_name='yolo11x.pt'):
        """
        Initialize autolabeler
        
        Args:
            label_studio_base: Base path to label-studio (defaults to LOCALAPPDATA/label-studio/label-studio)
            project_name: Name of the project folder to look for
            output_dir: Output directory (defaults to label-studio/Projects/{project_name})
        """
        import os as os_module
        
        # Default label-studio path
        if label_studio_base is None:
            label_studio_base = Path(os_module.environ.get('LOCALAPPDATA')) / 'label-studio' / 'label-studio'
        
        self.label_studio_base = Path(label_studio_base)
        
        # Use specific upload path: label-studio/media/upload/1 (or project_name if provided)
        # Label-studio stores images with random IDs in upload directories
        upload_base = self.label_studio_base / 'media' / 'upload'
        
        # Try project ID '1' first, then project_name, then just upload directory
        self.project_path = None
        possible_paths = [
            upload_base / '3',  # Specific project ID
            # # upload_base / project_name,
            # # upload_base,  # Fallback to upload directory
            # self.label_studio_base / 'Projects' / project_name,
            # self.label_studio_base / 'Projects',
        ]
        
        print("\nSearching for images in label-studio storage...")
        print(f"Base path: {self.label_studio_base}")
        print("Checking possible paths:")
        for path in possible_paths:
            exists = path.exists()
            print(f"  {path} - {'EXISTS' if exists else 'NOT FOUND'}")
            if exists and self.project_path is None:
                self.project_path = path
            print(f"\nUsing directory: {self.project_path}")
        
        if self.project_path is None:
            raise FileNotFoundError(
                f"Label-studio image directory not found!\n"
                f"Checked paths:\n" + "\n".join(f"  - {p}" for p in possible_paths) + "\n"
                f"Please ensure:\n"
                f"  1. Images have been uploaded to label-studio\n"
                f"  2. Label-studio is running\n"
                f"  3. Specify correct path with --label-studio-path or check the upload directory"
            )
        
        # Set source directory to project path
        self.source_dir = self.project_path
        print(f"Source directory: {self.source_dir}\n")
        
        # Set output directory - default to current repo directory
        if output_dir is None:
            # Save in current repository directory
            repo_dir = Path(__file__).parent
            self.output_dir = repo_dir / 'annotations' / project_name
        else:
            self.output_dir = Path(output_dir)
        
        # Create output directory structure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir = self.output_dir / "labels"
        self.images_dir = self.output_dir / "images"
        self.annotations_dir = self.output_dir / "annotations"
        
        self.labels_dir.mkdir(exist_ok=True)
        self.images_dir.mkdir(exist_ok=True)
        self.annotations_dir.mkdir(exist_ok=True)
        
        # Initialize YOLO model (using latest YOLOv11 - released Sept 2024)
        # Options: yolo11n.pt (nano-fastest), yolo11s.pt (small), yolo11m.pt (medium), 
        #          yolo11l.pt (large), yolo11x.pt (extra large - best accuracy)
        # Also supports: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
        print(f"Loading YOLO model: {model_name}...")
        try:
            self.model = YOLO(model_name)
            print(f"Loaded {model_name} model successfully")
        except Exception as e:
            print(f"Warning: Could not load {model_name}, trying yolo11l.pt: {e}")
            try:
                self.model = YOLO('yolo11l.pt')  # Fallback to YOLOv11 large model
                print("Loaded YOLOv11l model successfully")
            except Exception as e2:
                print(f"Warning: Could not load yolo11l.pt, trying yolov8x.pt: {e2}")
                try:
                    self.model = YOLO('yolov8x.pt')  # Fallback to YOLOv8 extra large
                    print("Loaded YOLOv8x model successfully")
                except Exception as e3:
                    print(f"Warning: Could not load yolov8x.pt, using yolo11m.pt: {e3}")
                    self.model = YOLO('yolo11m.pt')  # Final fallback to YOLOv11 medium
                    print("Loaded YOLOv11m model successfully")
        
        # COCO classes: class 9 is "traffic light"
        self.traffic_light_class_id = 9
        
        # Statistics
        self.stats = {
            'total_images': 0,
            'images_with_detections': 0,
            'total_detections': 0,
            'images_by_folder': defaultdict(int),
            'detections_by_folder': defaultdict(int)
        }
        
        # Store all image paths and their detections
        self.image_data = []
    
    def find_all_images(self):
        """Find all image files in the label-studio directory (with random ID filenames)"""
        image_extensions = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'}
        images = []
        
        print(f"Scanning for images in label-studio storage: {self.source_dir}")
        if not self.source_dir.exists():
            print(f"ERROR: Source directory does not exist: {self.source_dir}")
            print("Please check that label-studio is installed and the path is correct.")
            return []
        
        # Recursively search for images in label-studio storage
        print("Searching recursively...")
        for root, dirs, files in os.walk(self.source_dir):
            for file in files:
                file_path = Path(root) / file
                # Check if it's an image file (label-studio adds random IDs, so we check extension)
                if file_path.suffix.lower() in image_extensions:
                    images.append(file_path)
        
        if len(images) == 0:
            print(f"\nWARNING: No images found in {self.source_dir}")
            print("   Common label-studio image locations:")
            print(f"   - {self.label_studio_base / 'media' / 'upload'}")
            print(f"   - {self.label_studio_base / 'Projects'}")
            print("\n   Make sure:")
            print("   1. Images have been uploaded to label-studio")
            print("   2. Label-studio is running and has created the media directory")
            print("   3. The project name matches the folder in label-studio")
        else:
            print(f"\n✓ Found {len(images)} images in label-studio storage (with random IDs in filenames)")
            # Show first few image paths as examples
            if len(images) > 0:
                print(f"   Example: {images[0]}")
                if len(images) > 1:
                    print(f"   Example: {images[1]}")
        
        return sorted(images)
    
    def detect_traffic_signals(self, image_path):
        """Run YOLO detection on an image and filter for traffic lights"""
        results = self.model(str(image_path), verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Filter for traffic light class (class 9 in COCO)
                if int(box.cls) == self.traffic_light_class_id:
                    # Get bounding box coordinates (x1, y1, x2, y2)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    
                    detections.append({
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': confidence,
                        'class': 'traffic_light'
                    })
        
        return detections
    
    def convert_to_yolo_format(self, bbox, img_width, img_height):
        """Convert absolute bbox to YOLO format (normalized center x, center y, width, height)"""
        x1, y1, x2, y2 = bbox
        
        # Calculate center and dimensions
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        width = x2 - x1
        height = y2 - y1
        
        # Normalize
        center_x /= img_width
        center_y /= img_height
        width /= img_width
        height /= img_height
        
        return [center_x, center_y, width, height]
    
    def save_yolo_annotation(self, image_path, detections, img_width, img_height):
        """Save annotations in YOLO format (.txt file)"""
        # Use the actual filename (with random ID) as found in label-studio
        # Create relative path structure in output, preserving the random ID filename
        try:
            rel_path = image_path.relative_to(self.source_dir)
        except ValueError:
            # If paths are on different drives, use the filename directly
            rel_path = Path(image_path.name)
        
        label_filename = rel_path.with_suffix('.txt')
        label_path = self.labels_dir / label_filename.parent / label_filename.name
        
        # Create subdirectories if needed
        label_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write YOLO format (class_id center_x center_y width height)
        with open(label_path, 'w') as f:
            for det in detections:
                bbox_yolo = self.convert_to_yolo_format(
                    det['bbox'], img_width, img_height
                )
                # Class 0 for traffic light (you can adjust if you have multiple classes)
                f.write(f"0 {bbox_yolo[0]:.6f} {bbox_yolo[1]:.6f} {bbox_yolo[2]:.6f} {bbox_yolo[3]:.6f}\n")
        
        return label_path
    
    def convert_to_label_studio_url(self, image_path):
        """Convert local file path to label-studio URL format"""
        # Get relative path from label-studio base directory
        try:
            # Try relative to label-studio base
            rel_path = image_path.relative_to(self.label_studio_base)
        except ValueError:
            # If that fails, try relative to source_dir
            try:
                rel_path = image_path.relative_to(self.source_dir)
                # Prepend media/upload if needed
                if not str(rel_path).startswith('media'):
                    rel_path = Path('media') / 'upload' / rel_path
            except ValueError:
                # Last resort: just use filename
                rel_path = Path('media') / 'upload' / '1' / image_path.name
        
        # Convert Windows path to URL path format
        # Replace backslashes with forward slashes
        url_path = str(rel_path).replace('\\', '/')
        
        # Label-studio serves from /data/upload/{project_id}/{filename}
        # If path starts with media/upload, convert to /data/upload
        if url_path.startswith('media/upload'):
            url_path = url_path.replace('media/upload', '/data/upload', 1)
        elif not url_path.startswith('/'):
            # If it doesn't start with /, add /data/upload
            url_path = f"/data/upload/{url_path}"
        
        return url_path
    
    def save_label_studio_format(self, image_path, detections, img_width, img_height):
        """Save annotations in Label Studio format (JSON)"""
        # Use the actual filename (with random ID) as found in label-studio
        try:
            rel_path = image_path.relative_to(self.source_dir)
        except ValueError:
            # If paths are on different drives, use the filename directly
            rel_path = Path(image_path.name)
        
        annotation_filename = rel_path.with_suffix('.json')
        annotation_path = self.annotations_dir / annotation_filename.parent / annotation_filename.name
        
        annotation_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert file path to label-studio URL format
        label_studio_url = self.convert_to_label_studio_url(image_path)
        
        # Label Studio format for object detection
        # Store URL path that label-studio can serve (not Windows file path)
        label_studio_data = {
            "data": {
                "image": label_studio_url  # URL path for label-studio web server
            },
            "annotations": [{
                "result": []
            }]
        }
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            # Label Studio expects percentage coordinates
            x_percent = (x1 / img_width) * 100
            y_percent = (y1 / img_height) * 100
            width_percent = ((x2 - x1) / img_width) * 100
            height_percent = ((y2 - y1) / img_height) * 100
            
            label_studio_data["annotations"][0]["result"].append({
                "value": {
                    "x": x_percent,
                    "y": y_percent,
                    "width": width_percent,
                    "height": height_percent,
                    "rectanglelabels": ["traffic_light"]
                },
                "from_name": "label",
                "to_name": "image",
                "type": "rectanglelabels",
                "score": det['confidence']
            })
        
        with open(annotation_path, 'w') as f:
            json.dump(label_studio_data, f, indent=2)
        
        return annotation_path
    
    def process_images(self):
        """Process all images and generate annotations"""
        images = self.find_all_images()
        self.stats['total_images'] = len(images)
        
        print(f"\nProcessing {len(images)} images...")
        
        for image_path in tqdm(images, desc="Processing images"):
            try:
                # Read image to get dimensions
                img = cv2.imread(str(image_path))
                if img is None:
                    print(f"Warning: Could not read {image_path}")
                    continue
                
                img_height, img_width = img.shape[:2]
                
                # Run detection
                detections = self.detect_traffic_signals(image_path)
                
                # Get folder info for statistics
                try:
                    rel_path = image_path.relative_to(self.source_dir)
                    folder_key = str(rel_path.parent)
                except ValueError:
                    # If paths are on different drives, use parent directory name
                    folder_key = str(image_path.parent.name) if image_path.parent.name else "root"
                    rel_path = Path(image_path.name)
                
                self.stats['images_by_folder'][folder_key] += 1
                
                if detections:
                    self.stats['images_with_detections'] += 1
                    self.stats['total_detections'] += len(detections)
                    self.stats['detections_by_folder'][folder_key] += len(detections)
                    
                    # Save YOLO format annotation
                    yolo_path = self.save_yolo_annotation(image_path, detections, img_width, img_height)
                    
                    # Save Label Studio format annotation
                    ls_path = self.save_label_studio_format(image_path, detections, img_width, img_height)
                    
                    # Convert to label-studio URL for reference
                    label_studio_url = self.convert_to_label_studio_url(image_path)
                    
                    # Store metadata with full filename (including random ID)
                    self.image_data.append({
                        'image_path': str(image_path),  # Full Windows path with random ID
                        'image_filename': image_path.name,  # Just filename with random ID
                        'label_studio_url': label_studio_url,  # URL path for label-studio web server
                        'relative_path': str(rel_path),
                        'detections': detections,
                        'yolo_annotation': str(yolo_path),
                        'label_studio_annotation': str(ls_path),
                        'folder': folder_key
                    })
            
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
        
        # Save summary
        self.save_summary()
    
    def save_summary(self):
        """Save processing summary and file paths"""
        summary = {
            'statistics': {
                'total_images': self.stats['total_images'],
                'images_with_detections': self.stats['images_with_detections'],
                'total_detections': self.stats['total_detections'],
                'detection_rate': f"{(self.stats['images_with_detections'] / max(self.stats['total_images'], 1)) * 100:.2f}%"
            },
            'images_by_folder': dict(self.stats['images_by_folder']),
            'detections_by_folder': dict(self.stats['detections_by_folder']),
            'all_images': self.image_data
        }
        
        summary_path = self.output_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Also save a simple text file with all image paths and filenames
        paths_file = self.output_dir / "image_paths.txt"
        with open(paths_file, 'w') as f:
            for item in self.image_data:
                # Write full path and filename separately for clarity
                f.write(f"{item['image_path']}\n")
                f.write(f"  Filename: {item.get('image_filename', Path(item['image_path']).name)}\n")
        
        # Save a mapping file: filename -> full path (useful for label-studio integration)
        filename_mapping = self.output_dir / "filename_mapping.json"
        mapping = {}
        for item in self.image_data:
            filename = item.get('image_filename', Path(item['image_path']).name)
            mapping[filename] = {
                'full_path': item['image_path'],
                'relative_path': item['relative_path'],
                'detections_count': len(item['detections'])
            }
        with open(filename_mapping, 'w') as f:
            json.dump(mapping, f, indent=2)
        
        print(f"\n{'='*60}")
        print("PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"Total images processed: {self.stats['total_images']}")
        print(f"Images with detections: {self.stats['images_with_detections']}")
        print(f"Total detections: {self.stats['total_detections']}")
        print(f"Detection rate: {summary['statistics']['detection_rate']}")
        print(f"\nSource directory (label-studio): {self.source_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Summary saved to: {summary_path}")
        print(f"Image paths saved to: {paths_file}")
        print(f"Filename mapping saved to: {filename_mapping}")
        print(f"\nNote: File paths and filenames (with random IDs) stored as found in label-studio")
        print(f"{'='*60}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Autolabel traffic signals using YOLO on label-studio images')
    parser.add_argument('--label-studio-path', type=str, default=None,
                        help='Path to label-studio base directory (defaults to LOCALAPPDATA/label-studio/label-studio)')
    parser.add_argument('--project-name', type=str, default='Carla Traffic Signal Detection',
                        help='Name of the project folder in label-studio')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (defaults to ./annotations/{project_name} in current repo)')
    parser.add_argument('--model', type=str, default='yolo11x.pt',
                        help='YOLO model to use (yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt for YOLOv11, or yolov8 variants)')
    
    args = parser.parse_args()
    
    autolabeler = TrafficSignalAutolabeler(
        label_studio_base=args.label_studio_path,
        project_name=args.project_name,
        output_dir=args.output_dir,
        model_name=args.model
    )
    autolabeler.process_images()


if __name__ == '__main__':
    main()
