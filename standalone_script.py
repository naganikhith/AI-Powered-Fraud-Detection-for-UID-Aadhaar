#!/usr/bin/env python3
"""
Standalone script for testing Aadhaar fraud detection locally.
This runs YOLO object detection + OCR on input Aadhaar images.
"""

import os
import cv2
import easyocr
import argparse
from ultralytics import YOLO
import re

# -------------------------------
# Initialize models and OCR reader
# -------------------------------
DETECTION_MODEL_PATH = "detection_model/yolo11n.pt"  # Your YOLO detection model
CLASSIFICATION_MODEL_PATH = "classification_model/yolo11n-cls.pt"  # Optional classification model

print("[INFO] Loading models...")
detection_model = YOLO(DETECTION_MODEL_PATH)
ocr_reader = easyocr.Reader(['en'])

# -------------------------------
# Helper functions
# -------------------------------
def extract_text(image_path):
    """Extract text from image using EasyOCR."""
    results = ocr_reader.readtext(image_path)
    text = " ".join([res[1] for res in results])
    return text

def detect_entities(image_path):
    """Detect Aadhaar card entities using YOLOv8."""
    results = detection_model.predict(image_path, imgsz=640, conf=0.25)
    boxes = []
    for box in results[0].boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = box
        boxes.append({
            "bbox": [x1, y1, x2, y2],
            "confidence": round(conf, 2),
            "class_id": int(cls),
            "class_name": results[0].names[int(cls)]
        })
    return boxes

def detect_fraud(text):
    """Simple fraud detection check based on text pattern."""
    # Example heuristic: Aadhaar numbers must be 12 digits
    aadhaar_numbers = re.findall(r"\b\d{12}\b", text)
    if not aadhaar_numbers:
        return "⚠️ Possible Fraud: No valid Aadhaar number found."
    else:
        return f"✅ Valid Aadhaar number(s) detected: {aadhaar_numbers}"

# -------------------------------
# Main testing function
# -------------------------------
def main(image_path):
    if not os.path.exists(image_path):
        print(f"[ERROR] File not found: {image_path}")
        return

    print(f"\n[INFO] Processing: {image_path}")
    text = extract_text(image_path)
    detections = detect_entities(image_path)
    fraud_status = detect_fraud(text)

    print("\n=== OCR Extracted Text ===")
    print(text)
    print("\n=== Detected Entities ===")
    for det in detections:
        print(det)
    print("\n=== Fraud Detection Result ===")
    print(fraud_status)

    # Display image
    img = cv2.imread(image_path)
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        label = f"{det['class_name']} ({det['confidence']})"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.imshow("Detected Aadhaar", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# -------------------------------
# CLI Entry Point
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Aadhaar Fraud Detection")
    parser.add_argument("--image", required=True, help="Path to input Aadhaar image")
    args = parser.parse_args()
    main(args.image)
