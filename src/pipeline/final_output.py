# src/pipeline/final_output.py

import re
import logging
from typing import Dict, Any, List

logger = logging.getLogger("final_output_stage")
logger.setLevel(logging.INFO)


class FinalOutputStage:
    """
    Step 4 — Final Output Assembly
    Combines OCR + Normalization + Classification results into final structured JSON.
    """

    def __init__(self):
        pass

    def _find_source_snippet(self, text: str, value: float) -> str:
        """
        Find the nearest text snippet that contains the numeric value.
        Used for provenance tracing like: "text: 'Total: INR 1200'"
        """
        try:
            # Handle both integer and float values properly
            if isinstance(value, float) and value.is_integer():
                # For values like 1200.0, search for both 1200 and 1200.0
                int_value = int(value)
                pattern = re.compile(rf"(.{{0,25}}\b{int_value}(?:\.0+)?\b.{{0,25}})", re.IGNORECASE)
            else:
                # For decimal values, search for the exact value
                pattern = re.compile(rf"(.{{0,25}}\b{re.escape(str(value))}\b.{{0,25}})", re.IGNORECASE)
            
            match = pattern.search(text)
            if match:
                snippet = match.group(1).strip()
                return f"text: '{snippet}'"
        except Exception as e:
            logger.warning(f"Failed to find source snippet for {value}: {e}")
        return f"text: contains '{value}'"

    def run(self, ocr_output: Dict[str, Any], normalization_output: Dict[str, Any],
            classification_output: Dict[str, Any], text_content: str) -> Dict[str, Any]:
        """
        Combine all stages into one final output.
        Expected inputs:
          - OCR: {raw_tokens, currency_hint, ...}
          - Normalization: {normalized_amounts, details, ...}
          - Classification: {amounts, confidence, ...}
          - text_content: original text (for provenance)
        """
        try:
            # --- 1️⃣ Basic validations
            if not ocr_output or not ocr_output.get("raw_tokens"):
                return {"status": "no_amounts_found", "reason": "invalid_ocr_output"}
            
            # Check if OCR has error status
            if "status" in ocr_output and ocr_output["status"] != "ok":
                return {"status": "no_amounts_found", "reason": "ocr_stage_failed"}

            # Check normalization output
            if not normalization_output:
                return {"status": "no_amounts_found", "reason": "invalid_normalization_output"}
            
            # Check if normalization stage failed
            if "status" in normalization_output and normalization_output["status"] != "ok":
                return {"status": "no_amounts_found", "reason": "normalization_stage_failed"}
            
            if "normalized_amounts" not in normalization_output:
                return {"status": "no_amounts_found", "reason": "invalid_normalization_output"}

            # Check classification output
            if not classification_output:
                return {"status": "no_amounts_found", "reason": "invalid_classification_output"}
            
            # Check if classification stage failed
            if "status" in classification_output and classification_output["status"] != "ok":
                return {"status": "no_amounts_found", "reason": "classification_stage_failed"}
            
            if "amounts" not in classification_output:
                return {"status": "no_amounts_found", "reason": "invalid_classification_output"}

            currency = ocr_output.get("currency_hint", "INR")
            classified_amounts = classification_output.get("amounts", [])

            # --- 2️⃣ Build final labeled list with provenance
            final_amounts: List[Dict[str, Any]] = []
            for amt in classified_amounts:
                value = amt.get("value")
                amt_type = amt.get("type")
                if value is None or amt_type is None:
                    continue
                source = self._find_source_snippet(text_content, value)
                final_amounts.append({
                    "type": amt_type,
                    "value": value,
                    "source": source
                })

            if not final_amounts:
                return {"status": "no_amounts_found", "reason": "no_labeled_values"}

            # --- 3️⃣ Return final combined structure
            final_json = {
                "currency": currency,
                "amounts": final_amounts,
                "status": "ok"
            }

            logger.info(f"✅ Final Output: {final_json}")
            return final_json

        except Exception as e:
            logger.error(f"FinalOutputStage failed: {e}")
            return {"status": "error", "message": str(e)}
