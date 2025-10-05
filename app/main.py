from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="AI Amount Detector - MVP")

class HealthResponse(BaseModel):
    status: str

@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Simple health check endpoint.
    """
    return {"status": "ok"}

class ExtractResponse(BaseModel):
    status: str
    message: str

@app.post("/extract", response_model=ExtractResponse)
async def extract():
    """
    Placeholder endpoint for the full pipeline (OCR -> Normalization -> Classification -> Final JSON).
    We'll implement the pipeline in src/pipeline/* modules in later steps.
    """
    return {"status": "not_implemented", "message": "Pipeline not yet implemented. Proceed to Step 2 to add it."}
