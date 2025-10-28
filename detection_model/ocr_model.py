#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Aadhaar Card OCR Extraction using YOLO + EasyOCR
With Aadhaar Number Validation (Verhoeff Checksum)
"""

import cv2
import re
import easyocr
from ultralytics import YOLO

# ------------------- CONFIG -------------------
YOLO_WEIGHTS = "detection_model/yolo11n_fast_optimized/weights/best.pt"
reader = easyocr.Reader(['en'], gpu=False)  # Change gpu=True if you have GPU


# ------------------- HELPERS -------------------
def clean_text(text):
    """Remove unwanted characters and normalize whitespace"""
    text = re.sub(r'[^A-Za-z0-9:/\-\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def extract_aadhaar_number(text):
    matches = re.findall(r'\b\d{4}\s\d{4}\s\d{4}\b', text)
    if not matches:
        matches = re.findall(r'\b\d{12}\b', text)
    for m in matches:
        if not (m.startswith('19') or m.startswith('20')):
            return m
    return matches[0] if matches else None


def extract_dob(text):
    match = re.search(r'\b\d{2}[/-]\d{2}[/-]\d{4}\b', text)
    if match:
        dob = match.group()
        dob = re.sub(r'00', '01', dob)
        return dob
    return None


# ------------------- VERHOEFF CHECKSUM -------------------
def verhoeff_checksum(num):
    d = [
        [0,1,2,3,4,5,6,7,8,9],
        [1,2,3,4,0,6,7,8,9,5],
        [2,3,4,0,1,7,8,9,5,6],
        [3,4,0,1,2,8,9,5,6,7],
        [4,0,1,2,3,9,5,6,7,8],
        [5,9,8,7,6,0,4,3,2,1],
        [6,5,9,8,7,1,0,4,3,2],
        [7,6,5,9,8,2,1,0,4,3],
        [8,7,6,5,9,3,2,1,0,4],
        [9,8,7,6,5,4,3,2,1,0]
    ]
    p = [
        [0,1,2,3,4,5,6,7,8,9],
        [1,5,7,6,2,8,3,0,9,4],
        [5,8,0,3,7,9,6,1,4,2],
        [8,9,1,6,0,4,3,5,2,7],
        [9,4,5,3,1,2,6,8,7,0],
        [4,2,8,6,5,7,3,9,0,1],
        [2,7,9,3,8,0,6,4,1,5],
        [7,0,4,6,9,1,3,2,5,8]
    ]
    inv = [0,4,3,2,1,5,6,7,8,9]

    c = 0
    num = num[::-1]
    for i, n in enumerate(num):
        c = d[c][p[(i + 1) % 8][int(n)]]
    return inv[c]


def is_valid_aadhaar(aadhaar):
    aadhaar = aadhaar.replace(" ", "")
    return aadhaar.isdigit() and len(aadhaar) == 12 and verhoeff_checksum(aadhaar) == 0


# ------------------- MAIN EXTRACTION -------------------
def extract_text_fields(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return {"Error": "Invalid image path"}

    model = YOLO(YOLO_WEIGHTS)
    results = model(image_path)
    extracted_data = {}
    full_ocr_text = ""

    for box in results[0].boxes.data.tolist():
        x1, y1, x2, y2, conf, cls_id = box
        field_class = model.names[int(cls_id)]

        # Add padding to ROI
        pad = 5
        x1, y1 = max(0, int(x1) - pad), max(0, int(y1) - pad)
        x2, y2 = min(image.shape[1], int(x2) + pad), min(image.shape[0], int(y2) + pad)
        roi = image[y1:y2, x1:x2]

        # Try thresholded OCR for better clarity
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, 11, 2)

        texts = reader.readtext(thresh, detail=0)
        if not texts:  # fallback to colored ROI
            texts = reader.readtext(roi, detail=0)
        text = clean_text(' '.join(texts))

        extracted_data[field_class] = text
        full_ocr_text += " " + text

    # Regex fallback in case YOLO misses
    aadhaar_number = extract_aadhaar_number(full_ocr_text)
    extracted_data["Aadhaar Number"] = aadhaar_number
    extracted_data["DOB"] = extract_dob(full_ocr_text)
    extracted_data["Name"] = extracted_data.get('3', None)
    extracted_data["Full OCR Text"] = full_ocr_text.strip()

    # Aadhaar validation
    if aadhaar_number:
        extracted_data["Valid Aadhaar"] = is_valid_aadhaar(aadhaar_number)
    else:
        extracted_data["Valid Aadhaar"] = False

    return extracted_data


# ------------------- Example Run -------------------
if __name__ == "__main__":
    path = "sample_aadhaar.jpg"
    data = extract_text_fields(path)
    print(data)
