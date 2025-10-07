import logging
import io
import os
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse

# Initialize logger
logger = logging.getLogger(__name__)
from src.pipeline.ocr_stage import OCRStage
from src.pipeline.normalization import NormalizationStage
from src.pipeline.classification import ClassificationStage
from src.pipeline.final_output import FinalOutputStage
from typing import Optional
from PIL import Image
from dotenv import load_dotenv
load_dotenv()
# -----------------------------
# Initialize FastAPI app
# -----------------------------
app = FastAPI(
    title="AI Amount Detector (Gemini-powered OCR Stage)",
    description="Extracts numeric tokens and currency hints from text or image using Gemini 2.5 pro",
    version="1.0.0"
)

# -----------------------------
# Health / Readiness Endpoint
# -----------------------------
@app.get("/health")
async def health_check():
    return JSONResponse(content={
        "status": "ok",
        "service": "ai-amount-detector",
        "version": "1.0.0"
    }, status_code=200)

# -----------------------------
# Environment / Configuration
# -----------------------------
# Option 1: Set GEMINI_API_KEY as environment variable
# Option 2: Pass manually below

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(" Missing Gemini API key. Set GEMINI_API_KEY environment variable.")

# Initialize OCRStage (Gemini model)
ocr_stage = OCRStage(gemini_api_key= GEMINI_API_KEY, model_name="gemini-2.5-pro")

from src.pipeline.normalization import NormalizationStage
USE_GEMINI_FOR_NORMALIZATION = os.getenv("NORMALIZATION_USE_GEMINI", "false").lower() == "true"
normalization_stage = NormalizationStage(
    use_gemini=USE_GEMINI_FOR_NORMALIZATION,
    gemini_model=ocr_stage.model if USE_GEMINI_FOR_NORMALIZATION else None
)

# -----------------------------
# API Endpoint: /extract
# -----------------------------
@app.post("/extract")
async def extract_amounts(
    text: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None)
):
    """
    POST /extract
    Accepts either:
        - text: plain text (Form field)
        - image: uploaded image file
    Returns: JSON output from Gemini extraction
    """
    try:
        if text:
            # Mode: Text
            result = ocr_stage.run(text, input_mode="text")
            return JSONResponse(content=result)

        elif image:
            # Mode: Image
            content = await image.read()
            img = Image.open(io.BytesIO(content))
            result = ocr_stage.run(img, input_mode="image")
            return JSONResponse(content=result)

        else:
            raise HTTPException(status_code=400, detail="Provide either 'text' or 'image' input.")

    except Exception as e:
        return JSONResponse(
            content={"status": "error", "message": str(e)},
            status_code=500
        )

USE_GEMINI_FOR_NORMALIZATION = os.getenv("NORMALIZATION_USE_GEMINI", "false").lower() == "true"
normalization_stage = NormalizationStage(
    use_gemini=USE_GEMINI_FOR_NORMALIZATION,
    gemini_model=ocr_stage.model if USE_GEMINI_FOR_NORMALIZATION else None
)

@app.post("/extract/normalize")
async def extract_and_normalize(
    text: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    verbose: bool = False,   # set ?verbose=true to get full details (ocr + normalization.details)
):
    """
    Run OCR (step1) -> Normalization (step2).
    By default returns only:
      {
        "normalized_amounts": [...],
        "normalization_confidence": 0.95
      }
    Set verbose=true to return the full structure (for debugging/demo).
    """
    try:
        # 1) Run OCR stage (reuse existing logic)
        if text:
            ocr_res = ocr_stage.run(text, input_mode="text")
        elif image:
            content = await image.read()
            img = Image.open(io.BytesIO(content))
            ocr_res = ocr_stage.run(img, input_mode="image")
        else:
            raise HTTPException(status_code=400, detail="Provide either 'text' or 'image' input.")

        # 2) If OCR indicates no amounts, pass that back
        status = ocr_res.get("status") if isinstance(ocr_res, dict) else None
        if status and str(status).startswith("no_amounts_found"):
            # Keep existing guardrail output so callers can handle it
            return JSONResponse(content=ocr_res, status_code=200)

        # 3) Run normalization
        norm_res = normalization_stage.run(ocr_res)

        # 4) If verbose requested, return full object (ocr + normalization)
        if verbose:
            return JSONResponse(content={"ocr": ocr_res, "normalization": norm_res}, status_code=200)

        # 5) Default: compact / minimal response required by your endpoint
        compact = {
            "normalized_amounts": norm_res.get("normalized_amounts", []),
            "normalization_confidence": norm_res.get("normalization_confidence", 0.0)
        }
        return JSONResponse(content=compact, status_code=200)

    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)



classification_stage = ClassificationStage(gemini_model=ocr_stage.model)

@app.post("/extract/classify")
async def extract_and_classify(
    text: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None)
):
    try:
        # 1️ OCR stage
        if text:
            ocr_res = ocr_stage.run(text, input_mode="text")
            text_content = text
        elif image:
            content = await image.read()
            img = Image.open(io.BytesIO(content))
            ocr_res = ocr_stage.run(img, input_mode="image")
            text_content = ocr_res.get("raw_text", "")
        else:
            raise HTTPException(status_code=400, detail="Provide either text or image input.")

        print(f" OCR Result: {ocr_res}")

        # 2️ Normalization
        norm_res = normalization_stage.run(ocr_res)
        print(f" Normalization Result: {norm_res}")

        # 3️ Classification
        class_res = classification_stage.run(text_content, norm_res)
        print(f" Classification Result: {class_res}")

        return JSONResponse(content=class_res, status_code=200)

    except Exception as e:
        print(f" Error in extract_and_classify: {e}")
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)


# Configure Gemini usage for final output snippet extraction
USE_GEMINI_FOR_FINAL_SNIPPET = os.getenv("FINAL_SNIPPET_USE_GEMINI", "false").lower() == "true"
final_output_stage = FinalOutputStage(
    use_gemini=USE_GEMINI_FOR_FINAL_SNIPPET,
    gemini_model=ocr_stage.model if USE_GEMINI_FOR_FINAL_SNIPPET else None
)

@app.post("/extract/final")
async def extract_full_pipeline(
    text: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None)
):
    try:
        # 1️ OCR
        if text:
            ocr_res = ocr_stage.run(text, input_mode="text")
            text_content = text
        elif image:
            content = await image.read()
            img = Image.open(io.BytesIO(content))
            ocr_res = ocr_stage.run(img, input_mode="image")
            text_content = ocr_res.get("raw_text", "")
        else:
            raise HTTPException(status_code=400, detail="Provide either 'text' or 'image' input.")

        # 2️ Normalization
        norm_res = normalization_stage.run(ocr_res)

        # 3️ Classification
        class_res = classification_stage.run(text_content, norm_res)

        # 4️ Final output assembly
        final_json = final_output_stage.run(ocr_res, norm_res, class_res, text_content)
        return JSONResponse(content=final_json, status_code=200)

    except Exception as e:
        logger.error(f"Final pipeline error: {e}")
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)

