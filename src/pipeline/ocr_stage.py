"""
OCR stage: responsible for extracting raw tokens (strings) from text or image input
using pytesseract + OpenCV preprocessing.
This file WILL contain:
- functions to accept an image path or bytes
- preprocessing helpers (deskew, denoise, threshold)
- function: extract_raw_tokens(image_path: str) -> dict
"""

from typing import Dict, List

def extract_raw_tokens_from_image(image_path: str) -> Dict:
    """
    Placeholder implementation for Step 1 skeleton.
    Return structure:
    {
       "raw_tokens": [],
       "currency_hint": "INR",
       "confidence": 0.0
    }
    """
    # TODO: implement using cv2 + pytesseract
    return {"raw_tokens": [], "currency_hint": "INR", "confidence": 0.0}
