"""
Classification stage: use surrounding text to map numbers to labels like
total_bill, paid, due, discount, etc.
Functions:
- classify_amounts(tokens_with_context: List[Dict]) -> Dict
"""

from typing import Dict, List

def classify_amounts(normalized_amounts: List[int], contexts: List[str]) -> Dict:
    """
    Placeholder classification:
    {
      "amounts": [],
      "confidence": 0.0
    }
    """
    # TODO: implement rule-based + context matching to assign labels
    return {"amounts": [], "confidence": 0.0}
