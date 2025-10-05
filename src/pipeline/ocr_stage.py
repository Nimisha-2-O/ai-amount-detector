"""
OCR stage implementation (MVP)

Functions:
- extract_raw_tokens_from_bytes(image_bytes: bytes) -> dict

Returns a dict:
{
  "raw_tokens": [
    {
      "text": "₹ 1,234.00",
      "conf": 87.0,
      "left": 10,
      "top": 20,
      "width": 120,
      "height": 18
    },
    ...
  ],
  "currency_hint": "INR",        # simple hint if ₹ / Rs / INR seen
  "overall_confidence": 85.2    # average confidence across tokens
}
"""

from typing import Dict, List, Any
import cv2
import numpy as np
import pytesseract
import re

# If Tesseract is installed in default Windows location, set it explicitly.
# Update this path if your installation differs.
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Tesseract config for better numeric recognition (we keep default language)
_TESSERACT_CONFIG = r'--oem 3 --psm 6'

def _preprocess_image_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Convert bytes -> cv2 image and apply preprocessing:
    - convert to grayscale
    - bilateral filter (denoise while keeping edges)
    - adaptive threshold
    - optional resize if very small
    """
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize if image is very small to help OCR
    h, w = gray.shape
    if max(h, w) < 800:
        scale = 800 / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Denoise while preserving edges
    denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # Adaptive threshold (good for uneven lighting)
    th = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=15,
        C=9,
    )

    return th

def _detect_currency_hint(text: str) -> str:
    """
    Very simple currency detection heuristic. Returns 'INR' if Indian rupee symbols
    (₹, Rs, INR) are present, else 'UNKNOWN'.
    """
    if not text:
        return "UNKNOWN"
    if "₹" in text or re.search(r'\bRs\b', text, flags=re.IGNORECASE) or re.search(r'\bINR\b', text, flags=re.IGNORECASE):
        return "INR"
    return "UNKNOWN"

def extract_raw_tokens_from_bytes(image_bytes: bytes) -> Dict[str, Any]:
    """
    Main OCR function for Step 2.
    """
    # Preprocess
    preproc = _preprocess_image_bytes(image_bytes)

    # Use pytesseract to get detailed data
    # We use image_to_data which returns word-level info including confidences and bounding boxes.
    data = pytesseract.image_to_data(preproc, config=_TESSERACT_CONFIG, output_type=pytesseract.Output.DICT)

    raw_tokens: List[Dict[str, Any]] = []
    n_boxes = len(data.get("level", []))
    confidences = []
    full_text_accum = []

    for i in range(n_boxes):
        text = data.get("text", [])[i].strip() if data.get("text") else ""
        conf_raw = data.get("conf", [])[i] if data.get("conf") else "-1"
        # pytesseract sometimes returns '-1' for non-text boxes; handle that.
        try:
            conf = float(conf_raw)
        except Exception:
            conf = -1.0

        if text == "" or conf < 0:
            continue

        left = int(data.get("left", [])[i])
        top = int(data.get("top", [])[i])
        width = int(data.get("width", [])[i])
        height = int(data.get("height", [])[i])

        raw_tokens.append({
            "text": text,
            "conf": conf,
            "left": left,
            "top": top,
            "width": width,
            "height": height
        })
        confidences.append(conf)
        full_text_accum.append(text)

    overall_confidence = float(sum(confidences) / len(confidences)) if confidences else 0.0
    currency_hint = _detect_currency_hint(" ".join(full_text_accum))

    return {
        "raw_tokens": raw_tokens,
        "currency_hint": currency_hint,
        "overall_confidence": round(overall_confidence, 2)
    }
