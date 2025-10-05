"""
Improved Normalization stage
-----------------------------
Cleans OCR-extracted tokens and extracts valid numeric values only.
Rejects non-numeric words (like 'Docpulse').
"""

import re
from typing import Dict, List, Any
from rapidfuzz import fuzz

_OCR_CORRECTIONS = {
    "O": "0",
    "o": "0",
    "l": "1",
    "I": "1",
    "Z": "2",
    "S": "5",
    "s": "5",
    "B": "8"
}


def _clean_token(text: str) -> str:
    """Apply character corrections and remove non-digit symbols."""
    cleaned = text
    for wrong, right in _OCR_CORRECTIONS.items():
        cleaned = cleaned.replace(wrong, right)
    cleaned = re.sub(r"[^0-9.,₹RsINR]", "", cleaned)
    return cleaned.strip(",.")


def _looks_like_amount(text: str) -> bool:
    """Stricter numeric pattern check."""
    if not text:
        return False
    # If at least 70% of chars are digits
    digits = sum(c.isdigit() for c in text)
    ratio = digits / len(text)
    if ratio < 0.7:
        return False
    # Numeric pattern (200, 420.00, 1,234 etc.)
    return bool(re.match(r"^\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?$|^\d+(\.\d+)?$", text))


def normalize_tokens(raw_tokens: List[Dict[str, Any]]) -> Dict[str, Any]:
    results = []
    confidences = []

    for tok in raw_tokens:
        text = tok.get("text", "").strip()
        conf = float(tok.get("conf", 0.0))

        # Skip if confidence too low or empty text
        if not text or conf < 40:
            continue

        cleaned = _clean_token(text)

        # Skip if cleaned token has too few digits
        digits = sum(c.isdigit() for c in cleaned)
        if digits == 0:
            continue

        if not _looks_like_amount(cleaned):
            continue

        # Fuzzy similarity confidence
        ratio = fuzz.ratio(text, cleaned)
        combined_conf = (conf / 100.0) * (ratio / 100.0)

        try:
            value = float(cleaned.replace(",", ""))
            results.append({
                "raw": text,
                "cleaned": value,
                "confidence": round(combined_conf, 2)
            })
            confidences.append(combined_conf)
        except Exception:
            continue

    avg_conf = round(sum(confidences) / len(confidences), 2) if confidences else 0.0
    return {"normalized_amounts": results, "normalization_confidence": avg_conf}
