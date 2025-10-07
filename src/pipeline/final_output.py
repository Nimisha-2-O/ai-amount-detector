# src/pipeline/final_output.py

import re
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger("final_output_stage")
logger.setLevel(logging.INFO)


class FinalOutputStage:
    """
    Step 4 — Final Output Assembly
    Combines OCR + Normalization + Classification results into final structured JSON.
    """

    def __init__(self, use_gemini: bool = False, gemini_model: Optional[Any] = None):
        # use_gemini is enabled only if a model is provided
        self.use_gemini = use_gemini and (gemini_model is not None)
        self.model = gemini_model if self.use_gemini else None

    def _find_source_snippet_llm(self, text: str, value: float) -> Optional[str]:
        """
        Use Gemini to extract a concise clause that best evidences the given numeric value.
        Returns a string like "Paid: 1000" or None on failure.
        """
        if not self.use_gemini or not self.model:
            return None
        try:
            prompt = f"""
You are helping assemble provenance snippets for amounts found in bills/receipts.
Given the full text and a target numeric value, return the shortest phrase/segment
from the text that directly supports the value (e.g., "Total: INR 1200", "Paid: 1000").

Rules:
- Prefer the smallest self-contained clause around the number.
- Do not include unrelated neighboring fields split by separators like '|', ';', ',', or newlines.
- Avoid capturing percentages for plain numbers (do not match 10 if the text has 10%).
- Return ONLY the snippet text, no quotes, no markdown, no extra commentary.

Full text:
{text}

Target number: {value}
"""
            resp = self.model.generate_content(prompt)
            candidate = getattr(resp, "text", "").strip()
            if not candidate:
                return None
            # Basic sanity check: ensure the numeric value appears in the candidate
            str_val = str(int(value)) if (isinstance(value, float) and value.is_integer()) else str(value)
            if str_val not in candidate:
                return None
            # Collapse whitespace
            candidate = re.sub(r"\s+", " ", candidate).strip()
            return candidate
        except Exception:
            return None

    def _find_source_snippet(self, text: str, value: float) -> str:
        """
        Prefer Gemini-based snippet extraction when enabled; otherwise,
        return a concise clause around the numeric value using deterministic logic.
        """
        # 1) Try LLM route if available
        llm_snippet = self._find_source_snippet_llm(text, value)
        if llm_snippet:
            return f"text: '{llm_snippet}'"

        # 2) Deterministic fallback
        try:
            if isinstance(value, float) and value.is_integer():
                int_value = int(value)
                number_pattern = rf"\b{int_value}(?:\.0+)?\b(?!\s*%)"
            else:
                escaped = re.escape(str(value))
                number_pattern = rf"\b{escaped}\b(?!\s*%)"

            match = re.search(number_pattern, text, flags=re.IGNORECASE)
            if not match and isinstance(value, float) and value.is_integer():
                int_value = int(value)
                match = re.search(rf"\b{int_value}\b(?!\s*%)", text, flags=re.IGNORECASE)

            if match:
                start_idx, end_idx = match.span()
                separators = ['|', ';', ',', '\n']
                left_boundary = 0
                right_boundary = len(text)
                for sep in separators:
                    lb = text.rfind(sep, 0, start_idx)
                    if lb != -1:
                        left_boundary = max(left_boundary, lb + 1)
                for sep in separators:
                    rb = text.find(sep, end_idx)
                    if rb != -1:
                        right_boundary = min(right_boundary, rb)
                snippet = text[left_boundary:right_boundary].strip()
                snippet = re.sub(r"\s+", " ", snippet)
                if snippet:
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
            # --- 1️ Basic validations
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

            # --- 2️ Build final labeled list with provenance
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

            # --- 3️ Return final combined structure
            final_json = {
                "currency": currency,
                "amounts": final_amounts,
                "status": "ok"
            }

            logger.info(f" Final Output: {final_json}")
            return final_json

        except Exception as e:
            logger.error(f"FinalOutputStage failed: {e}")
            return {"status": "error", "message": str(e)}
