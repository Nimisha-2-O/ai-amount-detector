"""
Normalization stage: convert OCR tokens into numeric values, fix OCR digit errors.
Functions:
- normalize_tokens(raw_tokens: List[str]) -> Dict
"""

from typing import Dict, List

def normalize_tokens(raw_tokens: List[str]) -> Dict:
    """
    Placeholder normalization that returns:
    {
      "normalized_amounts": [],
      "normalization_confidence": 0.0
    }
    """
    # TODO: implement rapidfuzz correction heuristics + numeric parsing
    return {"normalized_amounts": [], "normalization_confidence": 0.0}
