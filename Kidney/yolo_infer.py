# yolo_infer.py
from ultralytics import YOLO
import cv2
from pathlib import Path
import numpy as np

# Load YOLO model once (so it doesn’t reload every request)
MODEL_PATH = "yolov8n.pt"
model = YOLO(MODEL_PATH)

def detect_kidney_stones(image_path: str, output_dir: str) -> str:
    """
    Run YOLO detection on an image and save annotated result.

    Args:
        image_path: path to input image
        output_dir: folder to save annotated image

    Returns:
        Path to annotated image
    """
    results = model(image_path)  # Run detection
    annotated_image = results[0].plot()  # Annotate

    # Save annotated image
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"annotated_{Path(image_path).name}"
    cv2.imwrite(str(out_path), annotated_image)
    return str(out_path)
