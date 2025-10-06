# src/pipeline/normalization.py
import re
import json
import logging
from typing import Dict, Any, List, Optional
from itertools import zip_longest

logger = logging.getLogger("normalization_stage")
logger.setLevel(logging.INFO)

# Try to import genai only when LLM fallback is enabled (avoid hard dependency if not used)
try:
    import google.generativeai as genai  # optional; only used if a model is provided
except Exception:
    genai = None

class NormalizationStage:
    DIGIT_FIX_MAP = {
        'O': '0', 'o': '0',
        'I': '1', 'l': '1', '|': '1', 'L': '1',
        'S': '5', 's': '5',
        'B': '8',
        'Z': '2',
        ' ': '',  # remove stray spaces between digits
    }

    def __init__(self, use_gemini: bool = False, gemini_model: Optional[Any] = None):
        """
        use_gemini: whether to call LLM for ambiguous tokens
        gemini_model: pass the same model instance from OCRStage (ocr_stage.model)
        """
        self.use_gemini = use_gemini and (gemini_model is not None)
        self.gemini_model = gemini_model if self.use_gemini else None

    def _apply_char_map(self, s: str) -> str:
        return ''.join(self.DIGIT_FIX_MAP.get(ch, ch) for ch in s)

    def _strip_currency_and_words(self, s: str) -> str:
        # remove common words like 'total', 'paid', 'due', 'rs', 'inr', 'rupees'
        s = re.sub(r'(?i)\b(total|paid|due|discount|subtotal|balance|rs|inr|rupees|amount)\b', '', s)
        # remove currency symbols
        s = re.sub(r'[₹\$€£,]', lambda m: ',' if m.group(0)==',' else '', s)
        # collapse spaces
        s = s.strip()
        return s

    def _keep_allowed_chars(self, s: str) -> str:
        # Keep digits, dot, comma, percent, minus
        return re.sub(r'[^0-9\.\,\%\-\+]', '', s)

    def _heuristic_resolve_separators(self, s: str) -> str:
        # If both '.' and ',' present -> interpret ',' as thousands sep (remove commas)
        if '.' in s and ',' in s:
            s = s.replace(',', '')
        else:
            # if only commas present and no dots => treat commas as thousands separators
            if ',' in s and '.' not in s:
                s = s.replace(',', '')
        return s

    def _clean_token(self, raw: str) -> str:
        if raw is None:
            return ""
        orig = str(raw)
        mapped = self._apply_char_map(orig)
        stripped = self._strip_currency_and_words(mapped)
        kept = self._keep_allowed_chars(stripped)
        resolved = self._heuristic_resolve_separators(kept)
        # remove leading/trailing dots/commas
        resolved = resolved.strip('.,')
        return resolved

    def _parse_number(self, token: str) -> Optional[float]:
        if not token:
            return None
        # skip percentages
        if '%' in token:
            return None
        # normalize leading pluses
        token = token.lstrip('+')
        try:
            if '.' in token:
                return float(token)
            return int(token)
        except Exception:
            # fallback: extract first numeric substring
            m = re.search(r'-?\d+(?:\.\d+)?', token)
            if m:
                num = m.group(0)
                if '.' in num:
                    return float(num)
                return int(num)
        return None

    def _replacements_count(self, raw: str, cleaned: str) -> int:
        # crude replacement count using zip_longest
        return sum(1 for a, b in zip_longest(str(raw), cleaned, fillvalue='') if a != b)

    def _call_gemini_batch(self, ambiguous_tokens: List[str]) -> Dict[str, Any]:
        """
        Ask Gemini to correct ambiguous tokens in a single batch call.
        Returns mapping raw->corrected_string
        """
        if not self.gemini_model:
            return {}
        # few-shot prompt
        prompt = f"""
You are an expert text parser for OCR'd numeric tokens from medical bills/receipts.
Given a list of ambiguous OCR tokens, return a strict JSON object mapping each raw token
to the most-likely corrected numeric string (or null if you cannot correct).
Only respond with a JSON object. No extra text.

Input tokens:
{json.dumps(ambiguous_tokens)}

Output schema:
{{
  "corrections": [
    {{"raw":"<original>","corrected":"<cleaned_numeric_or_null>","explanation":"<why>"}},
    ...
  ]
}}
"""
        try:
            resp = self.gemini_model.generate_content(prompt)
            raw_text = resp.text.strip()
            # extract JSON inside response
            m = re.search(r"\{[\s\S]*\}", raw_text)
            json_text = m.group(0) if m else raw_text
            parsed = json.loads(json_text)
            result = {}
            for item in parsed.get("corrections", []):
                raw = item.get("raw")
                corrected = item.get("corrected")
                result[raw] = {"corrected": corrected, "explanation": item.get("explanation")}
            return result
        except Exception as e:
            logger.warning(f"Gemini normalization call failed: {e}")
            return {}

    def run(self, ocr_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run normalization stage.
        Input: OCR stage JSON with "raw_tokens" and optional "confidence"
        Output:
        {
          "normalized_amounts": [1200, ...],
          "normalization_confidence": 0.82,
          "details": [
            {"raw": "...", "cleaned": "...", "value": 1200, "method": "rule"|"llm", "confidence": 0.9}
          ]
        }
        """
        if not isinstance(ocr_output, dict) or "raw_tokens" not in ocr_output:
            return {"status": "error", "reason": "invalid_input"}

        raw_tokens = ocr_output.get("raw_tokens", [])
        ocr_conf = float(ocr_output.get("confidence", 0.75) or 0.75)

        details = []
        normalized_values = []
        ambiguous_raws = []

        # 1) deterministic pass
        for raw in raw_tokens:
            cleaned = self._clean_token(raw)
            value = self._parse_number(cleaned)
            replacements = self._replacements_count(raw, cleaned)
            rule_conf = max(0.5, 1.0 - 0.15 * replacements)  # heuristic
            token_conf = round(ocr_conf * rule_conf, 4)

            if value is None or replacements >= 2:
                # mark ambiguous for LLM fallback
                ambiguous_raws.append(raw)
                details.append({
                    "raw": raw,
                    "cleaned": cleaned,
                    "value": None,
                    "method": "ambiguous",
                    "confidence": token_conf
                })
            else:
                details.append({
                    "raw": raw,
                    "cleaned": cleaned,
                    "value": value,
                    "method": "rule",
                    "confidence": token_conf
                })
                normalized_values.append(value)

        # 2) LLM fallback (batch) for ambiguous tokens
        if ambiguous_raws and self.use_gemini:
            corrections = self._call_gemini_batch(ambiguous_raws)
            for i, d in enumerate(details):
                if d["method"] == "ambiguous":
                    raw = d["raw"]
                    corr_entry = corrections.get(raw)
                    if corr_entry and corr_entry.get("corrected"):
                        cleaned2 = corr_entry["corrected"]
                        parsed = self._parse_number(cleaned2)
                        if parsed is not None:
                            # give slightly lower confidence than clean rules but higher than ambiguous
                            llm_conf = round(min(0.95, ocr_conf * 0.92), 4)
                            d.update({
                                "cleaned": cleaned2,
                                "value": parsed,
                                "method": "llm",
                                "confidence": llm_conf,
                                "explanation": corr_entry.get("explanation")
                            })
                            normalized_values.append(parsed)
                        else:
                            # no correction found
                            d.update({"explanation": corr_entry.get("explanation") if corr_entry else None})
                    else:
                        # no correction returned for that raw token
                        pass

        if not normalized_values:
            return {"status": "no_amounts_found", "reason": "no_numeric_tokens"}

        # 3) aggregate normalization confidence
        confidences = [d["confidence"] for d in details if d.get("value") is not None]
        normalization_conf = float(sum(confidences) / len(confidences)) if confidences else ocr_conf

        result = {
            "normalized_amounts": normalized_values,
            "normalization_confidence": round(normalization_conf, 2),
            "details": details
        }
        logger.info(f"Normalization result: {result}")
        return result
