import io
import os
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from src.pipeline.ocr_stage import OCRStage
from typing import Optional
from PIL import Image
from dotenv import load_dotenv
load_dotenv()
# -----------------------------
# Initialize FastAPI app
# -----------------------------
app = FastAPI(
    title="AI Amount Detector (Gemini-powered OCR Stage)",
    description="Extracts numeric tokens and currency hints from text or image using Gemini 1.5",
    version="1.0.0"
)

# -----------------------------
# Environment / Configuration
# -----------------------------
# Option 1: Set GEMINI_API_KEY as environment variable
# Option 2: Pass manually below

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("❌ Missing Gemini API key. Set GEMINI_API_KEY environment variable.")

# Initialize OCRStage (Gemini model)
ocr_stage = OCRStage(gemini_api_key= GEMINI_API_KEY, model_name="gemini-2.0-flash")


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
# @app.post("/extract/ocr", response_model=OCRStage.OCRResponse)
# async def extract_ocr(file: UploadFile = File(...)):
#     """
#     Accept an image upload (jpg/png/pdf pages as images) and return raw OCR tokens.
#     """
#     # Basic validation: content-type
#     if not file.content_type.startswith("image/"):
#         raise HTTPException(
#             status_code=400, detail="Invalid file type. Please upload an image."
#         )

#     contents = await file.read()
#     try:
#         result = extract_raw_tokens_from_bytes(contents)
#     except ValueError as ve:
#         raise HTTPException(status_code=400, detail=str(ve))
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"OCR failure: {str(e)}")

#     return {"status": "ok", "data": result}


# @app.post("/extract/normalize", response_model=Dict)
# async def extract_normalize(file: UploadFile = File(...)):
#     """
#     Run OCR + normalization (Stages 1 & 2 together)
#     """
#     contents = await file.read()
#     try:
#         ocr_result = extract_raw_tokens_from_bytes(contents)
#         norm_result = normalize_tokens(ocr_result["raw_tokens"])
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Normalization failure: {str(e)}")

#     return {
#         "status": "ok",
#         "data": {
#             "ocr_confidence": ocr_result["overall_confidence"],
#             "currency_hint": ocr_result["currency_hint"],
#             "normalized": norm_result,
#         },
#     }


# # --- New endpoint: Full pipeline ---
# @app.post("/extract/final", response_model=Dict)
# async def extract_final(file: UploadFile = File(...)):
#     """
#     Run full pipeline: OCR + normalization + classification + final output.
#     """
#     contents = await file.read()
#     try:
#         ocr_result = extract_raw_tokens_from_bytes(contents)
#         norm_result = normalize_tokens(ocr_result["raw_tokens"])
#         # Extract normalized amounts and their contexts
#         normalized_amounts = [a["cleaned"] for a in norm_result["normalized_amounts"]]
#         contexts = [a["raw"] for a in norm_result["normalized_amounts"]]
#         classified = classify_amounts(normalized_amounts, contexts)
#         final_output = assemble_final_output(
#             classified, currency=ocr_result.get("currency_hint", "INR")
#         )
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Pipeline failure: {str(e)}")

#     return {"status": "ok", "data": final_output}
