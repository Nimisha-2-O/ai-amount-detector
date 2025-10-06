# src/pipeline/classification.py
import json
import re
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("classification_stage")
logger.setLevel(logging.INFO)

try:
    import google.generativeai as genai
except Exception:
    genai = None


def extract_json_from_gemini(raw_text: str) -> str:
    """
    Cleans Gemini output to extract valid JSON.
    Handles cases with ```json fences, spaces, or newlines.
    """
    if not raw_text:
        return ""

    # 1️⃣ Remove any markdown code fences, anywhere in the text
    raw_text = re.sub(r"```(?:json)?", "", raw_text, flags=re.IGNORECASE)
    raw_text = raw_text.replace("```", "")

    # 2️⃣ Strip leading/trailing whitespace/newlines
    raw_text = raw_text.strip()

    # 3️⃣ Find the first '{' and last '}' to isolate JSON block
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return raw_text[start:end + 1].strip()

    # 4️⃣ Fallback — return cleaned text
    return raw_text

    

class ClassificationStage:
    """
    Step 3 — Classification by Context
    Use Gemini to assign meaning (total_bill, paid, due, etc.)
    to each numeric value, based on nearby words in text.
    """
    def __init__(self, gemini_model: Optional[Any] = None):
        if not gemini_model:
            raise ValueError("Gemini model instance required.")
        self.model = gemini_model

    def _build_prompt(self, text: str, normalization_output: Dict[str, Any]) -> str:
        normalized_amounts = normalization_output.get("normalized_amounts", [])
        confidence = normalization_output.get("normalization_confidence", 0.75)

        prompt = f"""
You are an expert in understanding billing and receipt documents.

Given the following text extracted from a bill or receipt:

{text}

And the following numeric amounts detected and normalized:
{normalized_amounts}

Your task:
Classify each numeric amount by its context into one of these categories:
- total_bill (grand total, total amount)
- paid (payment made)
- due (amount pending)
- discount (discount, rebate)
- tax (GST, CGST, SGST, VAT)
- subtotal (pre-tax total)
- refund (refund given)
- other (unclear context)

Respond only in this **strict JSON format**:
{{
  "amounts": [
    {{"type": "<category>", "value": <number>}},
    ...
  ],
  "confidence": <float between 0 and 1>
}}

Use context words like "Total", "Paid", "Due" to decide.
The confidence should represent your overall certainty.
Do not include explanations or markdown.
"""
        return prompt.strip()

    def run(self, text: str, normalization_output: Dict[str, Any]) -> Dict[str, Any]:
        """Classify normalized amounts using Gemini AI and text context."""
        if not normalization_output or "normalized_amounts" not in normalization_output:
            return {"status": "error", "reason": "invalid_input"}

        prompt = self._build_prompt(text, normalization_output)
        try:
            response = self.model.generate_content(prompt)
            raw_text = response.text.strip()

            # Clean Gemini output
            raw_text = re.sub(r"```[\s\S]*?```", "", raw_text)
            json_text = extract_json_from_gemini(raw_text)
            parsed = json.loads(json_text)

            if isinstance(parsed, dict) and "amounts" in parsed:
                return parsed
            else:
                logger.warning(f"Unexpected Gemini output: {raw_text}")
                return {"status": "error", "reason": "invalid_json_structure"}

        except Exception as e:
            logger.error(f"Gemini classification failed: {e}")
            return {"status": "error", "reason": "gemini_api_error"}
