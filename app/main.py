from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Any, Dict
from src.pipeline.ocr_stage import extract_raw_tokens_from_bytes

app = FastAPI(title="AI Amount Detector - MVP")

class HealthResponse(BaseModel):
    status: str

@app.get("/health", response_model=HealthResponse)
async def health():
    return {"status": "ok"}

class OCRResponse(BaseModel):
    status: str
    data: Dict[str, Any]

@app.post("/extract/ocr", response_model=OCRResponse)
async def extract_ocr(file: UploadFile = File(...)):
    """
    Accept an image upload (jpg/png/pdf pages as images) and return raw OCR tokens.
    """
    # Basic validation: content-type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    contents = await file.read()
    try:
        result = extract_raw_tokens_from_bytes(contents)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR failure: {str(e)}")

    return {"status": "ok", "data": result}
