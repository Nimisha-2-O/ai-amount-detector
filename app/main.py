from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Any, Dict
from src.pipeline.ocr_stage import extract_raw_tokens_from_bytes

from src.pipeline.normalization import normalize_tokens
from src.pipeline.classification import classify_amounts
from src.pipeline.final_output import assemble_final_output

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
        raise HTTPException(
            status_code=400, detail="Invalid file type. Please upload an image."
        )

    contents = await file.read()
    try:
        result = extract_raw_tokens_from_bytes(contents)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR failure: {str(e)}")

    return {"status": "ok", "data": result}


@app.post("/extract/normalize", response_model=Dict)
async def extract_normalize(file: UploadFile = File(...)):
    """
    Run OCR + normalization (Stages 1 & 2 together)
    """
    contents = await file.read()
    try:
        ocr_result = extract_raw_tokens_from_bytes(contents)
        norm_result = normalize_tokens(ocr_result["raw_tokens"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Normalization failure: {str(e)}")

    return {
        "status": "ok",
        "data": {
            "ocr_confidence": ocr_result["overall_confidence"],
            "currency_hint": ocr_result["currency_hint"],
            "normalized": norm_result,
        },
    }


# --- New endpoint: Full pipeline ---
@app.post("/extract/final", response_model=Dict)
async def extract_final(file: UploadFile = File(...)):
    """
    Run full pipeline: OCR + normalization + classification + final output.
    """
    contents = await file.read()
    try:
        ocr_result = extract_raw_tokens_from_bytes(contents)
        norm_result = normalize_tokens(ocr_result["raw_tokens"])
        # Extract normalized amounts and their contexts
        normalized_amounts = [a["cleaned"] for a in norm_result["normalized_amounts"]]
        contexts = [a["raw"] for a in norm_result["normalized_amounts"]]
        classified = classify_amounts(normalized_amounts, contexts)
        final_output = assemble_final_output(
            classified, currency=ocr_result.get("currency_hint", "INR")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failure: {str(e)}")

    return {"status": "ok", "data": final_output}
