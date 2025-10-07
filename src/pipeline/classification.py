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

    # 1️ Remove any markdown code fences, anywhere in the text
    raw_text = re.sub(r"```(?:json)?", "", raw_text, flags=re.IGNORECASE)
    raw_text = raw_text.replace("```", "")

    # 2️ Strip leading/trailing whitespace/newlines
    raw_text = raw_text.strip()

    # 3️ Find the first '{' and last '}' to isolate JSON block
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        json_content = raw_text[start:end + 1].strip()
        # Validate that it looks like JSON
        if json_content.startswith("{") and json_content.endswith("}"):
            return json_content

    # 4️ Try to find JSON in code blocks
    json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
    match = re.search(json_pattern, raw_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # 5️ Try to find JSON anywhere in the text (more flexible)
    json_pattern_flexible = r'\{[^{}]*"amounts"[^{}]*\}'
    match = re.search(json_pattern_flexible, raw_text, re.DOTALL)
    if match:
        return match.group(0).strip()

    # 6️ Fallback — return cleaned text if it looks like JSON
    if raw_text.startswith("{") and raw_text.endswith("}"):
        return raw_text

    # 7️ Last resort — return empty string
    return ""

    

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
        logger.info(f"Classification input - text: {text[:100]}...")
        logger.info(f"Classification input - normalization_output: {normalization_output}")
        
        # Check if normalization_output has error status
        if isinstance(normalization_output, dict) and normalization_output.get("status") == "error":
            logger.error(f"Normalization stage failed: {normalization_output}")
            return {"status": "error", "reason": "normalization_failed"}
        
        if not normalization_output or "normalized_amounts" not in normalization_output:
            logger.error(f"Invalid normalization output: {normalization_output}")
            return {"status": "error", "reason": "invalid_input"}

        # Check if we have any amounts to classify
        normalized_amounts = normalization_output.get("normalized_amounts", [])
        if not normalized_amounts:
            logger.warning("No normalized amounts to classify")
            return {"status": "error", "reason": "no_amounts_to_classify"}

        prompt = self._build_prompt(text, normalization_output)
        logger.info(f"Classification prompt: {prompt[:200]}...")
        
        try:
            response = self.model.generate_content(prompt)
            
            # Check if response is valid
            if not response or not hasattr(response, 'text'):
                logger.error("Invalid response from Gemini API")
                return {"status": "error", "reason": "invalid_gemini_response"}
            
            raw_text = response.text.strip()
            
            # Check if response is empty
            if not raw_text:
                logger.error("Empty response from Gemini API")
                return {"status": "error", "reason": "empty_gemini_response"}
            
            # Log the raw response for debugging
            logger.info(f"Raw Gemini response: {raw_text}")

            # Extract JSON from Gemini output
            json_text = extract_json_from_gemini(raw_text)
            
            # Log the extracted JSON for debugging
            logger.info(f"Extracted JSON: {json_text}")
            
            # Check if we have valid JSON text
            if not json_text or json_text.strip() == "":
                logger.error(f"No JSON content found in Gemini response. Raw response: '{raw_text}'")
                return {"status": "error", "reason": "empty_response"}

            # Additional validation - check if it looks like JSON
            json_text = json_text.strip()
            if not (json_text.startswith('{') and json_text.endswith('}')):
                logger.error(f"Extracted text doesn't look like JSON: '{json_text}'")
                return {"status": "error", "reason": "invalid_json_format"}

            parsed = json.loads(json_text)

            if isinstance(parsed, dict) and "amounts" in parsed:
                logger.info(f"Classification successful: {parsed}")
                return parsed
            else:
                logger.warning(f"Unexpected Gemini output structure: {parsed}")
                return {"status": "error", "reason": "invalid_json_structure"}

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}, Raw text: {raw_text}")
            return {"status": "error", "reason": "json_parse_error"}
        except Exception as e:
            logger.error(f"Gemini classification failed: {e}")
            return {"status": "error", "reason": "gemini_api_error"}
