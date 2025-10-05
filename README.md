# AI Amount Detector (MVP) - ai-amount-detector

## Project
Problem Statement 8 — AI-Powered Amount Detection in Medical Documents

## Stack (strict)
- Backend Framework: FastAPI
- OCR Engine: pytesseract (requires Tesseract install)
- Image Preprocessing: OpenCV
- Validation: Pydantic
- Optional: rapidfuzz
- Tunnel (demo): ngrok / pyngrok

## How to run (Step 1 - skeleton)
1. Clone the repo
2. Create and activate virtualenv:
   - `python -m venv .venv`
   - `.\.venv\Scripts\Activate.ps1`
3. Install requirements:
   - `pip install -r requirements.txt`
4. Run:
   - `uvicorn app.main:app --reload --host 127.0.0.1 --port 8000`
5. Health check:
   - `curl http://127.0.0.1:8000/health`

## Repo structure (initial)
- app/main.py
- src/pipeline/ocr_stage.py
- src/pipeline/normalization.py
- src/pipeline/classification.py
- src/pipeline/final_output.py
- Nimisha_Shrivastava_8/  (submission folder)
