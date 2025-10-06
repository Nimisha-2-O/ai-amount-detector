"""
Step 1 — AI-driven OCR/Text Extraction using Google Gemini

- Accepts raw text or image input.
- Uses Tesseract OCR for visual text extraction (AI model).
- Uses Gemini 2.5 Pro for intelligent extraction.
- Returns clean structured JSON.

Dependencies:
    pip install google-generativeai pytesseract opencv-python pillow numpy
"""

import os
import json
import re
from typing import Union, Dict, Any
import google.generativeai as genai
from PIL import Image
import pytesseract
import numpy as np
import cv2
import logging
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# -------------------------------
# Logging setup
# -------------------------------
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("ocr_stage_gemini")

# -------------------------------
# Gemini Configuration
# -------------------------------
def configure_gemini(api_key: str, model: str = "gemini-2.5-pro"):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model)


# -------------------------------
# OCR Utility
# -------------------------------
def preprocess_for_ocr(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    gray = cv2.filter2D(gray, -1, kernel)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel_m = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel_m)
    return th


def extract_text_from_image(image_input: Union[str, bytes, np.ndarray, Image.Image]) -> str:
    """Run Tesseract OCR on an image input (path, bytes, array, or PIL)."""
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image not found: {image_input}")
        img = cv2.imread(image_input)
    elif isinstance(image_input, (bytes, bytearray)):
        img_array = np.frombuffer(image_input, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    elif isinstance(image_input, np.ndarray):
        img = image_input
    elif isinstance(image_input, Image.Image):
        img = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
    else:
        raise TypeError("Unsupported image input type.")

    processed = preprocess_for_ocr(img)
    text = pytesseract.image_to_string(processed)
    return text.strip()


# -------------------------------
# Gemini Reasoning Extraction
# -------------------------------
def gemini_extract_amounts(model, text: str) -> Dict[str, Any]:
    """Send OCR text to Gemini and parse structured output."""
    prompt = fprompt = f"""
You are an expert financial document parser specialized in reading text extracted from scanned
bills, receipts, and invoices (including hospital bills, restaurant receipts, and shopping invoices).

The text below may contain OCR errors such as:
- 'O' or 'o' instead of '0'
- 'I', 'l', or '|' instead of '1'
- 'S' instead of '5'
- Missing spaces, or broken numeric tokens (e.g., "3 70.40" instead of "370.40")

Your goal:
1. Identify all numeric or percentage values that represent financial amounts, totals, discounts,
   taxes, balances, or payments.
2. Correct likely OCR errors in numeric tokens.
3. Merge broken tokens into proper numeric values.
4. Detect the most likely currency hint (e.g., "INR", "USD", "EUR", etc.) from symbols or words in text.
5. Return only clean structured JSON, with no extra text or explanation.

Output must be a **strict JSON object** with the following schema:

{{
  "raw_tokens": [list of cleaned numeric or percentage strings],
  "currency_hint": "INR" | "USD" | "EUR" | null,
  "confidence": float (0.0 - 1.0)
}}

Guidelines:
- Keep the tokens in the same order they appear in the text.
- If you find duplicate numbers (like subtotal and total both same), still include both.
- Confidence should reflect how certain you are about the correctness of extraction.
- If you find no valid amounts, return exactly:
  {{"status": "no_amounts_found", "reason": "document too noisy"}}

Example 1:
Input: Total: INR 1200 | Paid: 1000 | Due: 200 | Discount: 10%
Output: {{"raw_tokens": ["1200","1000","200","10%"], "currency_hint": "INR", "confidence": 0.95}}

Example 2:
Input: T0tal : Rs l200 | Pald : 1000 | DUE : 200
Output: {{"raw_tokens": ["1200","1000","200"], "currency_hint": "INR", "confidence": 0.85}}

Example 3:
Input: Subtotal $49.99 | Tax 5.00 | Total 54.99
Output: {{"raw_tokens": ["49.99","5.00","54.99"], "currency_hint": "USD", "confidence": 0.94}}

Now analyze and extract from this text:
{text}
"""


    try:
        response = model.generate_content(prompt)
        raw_output = response.text.strip()

        # Remove code fences or markdown
        raw_output = re.sub(r"^```(json)?|```$", "", raw_output, flags=re.MULTILINE).strip()

        # Try JSON parse
        parsed = json.loads(raw_output)

        # Validate structure
        if isinstance(parsed, dict) and "raw_tokens" in parsed:
            return parsed
        else:
            return {"status": "no_amounts_found", "reason": "invalid_json_structure"}
    except Exception as e:
        logger.warning(f"Gemini extraction failed: {e}")
        return {"status": "no_amounts_found", "reason": "gemini_api_error"}


# -------------------------------
# Main Class
# -------------------------------
class OCRStage:
    def __init__(self, gemini_api_key: str, model_name: str = "gemini-2.5-pro"):
        self.model = configure_gemini(gemini_api_key, model_name)

    def run(self, input_data: Union[str, bytes, np.ndarray, Image.Image], input_mode: str = "auto") -> Dict[str, Any]:
        """
        input_mode: "text", "image", or "auto" (detect automatically)
        """
        # Detect mode automatically
        if input_mode == "auto":
            if isinstance(input_data, str) and os.path.exists(input_data):
                input_mode = "image"
            elif isinstance(input_data, str):
                input_mode = "text"
            elif isinstance(input_data, (bytes, bytearray, np.ndarray, Image.Image)):
                input_mode = "image"
            else:
                raise ValueError("Cannot auto-detect input mode. Use 'text' or 'image' explicitly.")

        # Extract text if image
        if input_mode == "image":
            try:
                ocr_text = extract_text_from_image(input_data)
            except Exception as e:
                logger.error(f"OCR extraction failed: {e}")
                return {"status": "no_amounts_found", "reason": "ocr_failed"}

            logger.info(f"OCR extracted text: {ocr_text[:120]}...")
            return gemini_extract_amounts(self.model, ocr_text)

        # Direct text input
        elif input_mode == "text":
            text = str(input_data).strip()
            if not text:
                return {"status": "no_amounts_found", "reason": "empty_text"}
            return gemini_extract_amounts(self.model, text)

        else:
            raise ValueError("Invalid input_mode. Choose from 'text', 'image', or 'auto'.")



