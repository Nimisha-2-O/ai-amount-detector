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
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "ocr_pipeline.log")

# Configure logging to file + console
logger = logging.getLogger("ocr_stage_gemini")
logger.setLevel(logging.INFO)

# File handler
file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
file_handler.setLevel(logging.INFO)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Log format
formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

logger.info(" Logging initialized. All OCR and Gemini outputs will be saved in logs/ocr_pipeline.log")


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
    """
    Balanced preprocessing that enhances text clarity
    without destroying fine details. Works well for bills,
    receipts, and forms with faint or colored text.
    """

    # 1️⃣ Convert to grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 2️⃣ Remove mild noise while preserving edges
    gray = cv2.fastNlMeansDenoising(gray, h=15, templateWindowSize=7, searchWindowSize=21)

    # 3️⃣ Gentle contrast stretching (normalize intensity range)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    # 4️⃣ Apply CLAHE for local contrast enhancement (limited clip)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # 5️⃣ Light gamma correction to brighten dark text
    gamma = 1.2
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)]).astype("uint8")
    gray = cv2.LUT(gray, table)

    # 6️⃣ (Optional) adaptive sharpening to highlight edges
    blurred = cv2.GaussianBlur(gray, (0, 0), 3)
    sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)

    # 7️⃣ Slight morphological opening to remove background dots/noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    clean = cv2.morphologyEx(sharpened, cv2.MORPH_OPEN, kernel)

    return clean


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
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(processed, config=custom_config)

    return text.strip()


# -------------------------------
# Gemini Reasoning Extraction
# -------------------------------
def gemini_extract_amounts(model, text: str) -> Dict[str, Any]:
    """Send OCR text to Gemini and parse structured output."""
    prompt = f"""
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

Now analyze and extract from this text:
{text}

VERY IMPORTANT:
Respond with ONLY a valid JSON object — no markdown, no code fences, no explanations.
The output must start with '{{' and end with '}}' and be fully parseable by json.loads().
"""

    try:
        response = model.generate_content(prompt)
        raw_output = response.text.strip()
        logger.info("🤖 Full Gemini raw output:\n" + raw_output)

        # 🔧 Fix: Remove any Markdown code fences (```json, ``` etc.)
        raw_output = re.sub(r"```[\s\S]*?```", lambda m: m.group(0).strip('`').strip(), raw_output)
        raw_output = re.sub(r"^```(json)?|```$", "", raw_output, flags=re.MULTILINE).strip()

        # 🔧 Extract only JSON portion (handles markdown wrappers and extra text)
        json_match = re.search(r"\{[\s\S]*\}", raw_output)
        if json_match:
            json_text = json_match.group(0).strip()
        else:
            json_text = raw_output.strip()

        # 🔧 Try to parse JSON safely
        try:
            parsed = json.loads(json_text)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}\nRaw output:\n{raw_output}")
            return {"status": "no_amounts_found", "reason": "invalid_json_format"}

        # ✅ Validate structure
        if isinstance(parsed, dict) and "raw_tokens" in parsed:
            return parsed
        elif isinstance(parsed, dict) and "status" in parsed:
            return parsed
        else:
            logger.warning(f"Unexpected JSON keys: {list(parsed.keys())}")
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

            logger.info("Full OCR extracted text:\n" + ocr_text)
            return gemini_extract_amounts(self.model, ocr_text)

        # Direct text input
        elif input_mode == "text":
            text = str(input_data).strip()
            if not text:
                return {"status": "no_amounts_found", "reason": "empty_text"}
            return gemini_extract_amounts(self.model, text)

        else:
            raise ValueError("Invalid input_mode. Choose from 'text', 'image', or 'auto'.")



