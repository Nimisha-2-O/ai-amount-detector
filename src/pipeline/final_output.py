"""
Final output assembly: produce the final JSON with currency, provenance, and amounts.
Functions:
- assemble_final_output(classified_amounts: List[Dict], currency: str) -> Dict
"""

from typing import Dict, List

def assemble_final_output(classified_amounts: List[Dict], currency: str = "INR") -> Dict:
    """
    Placeholder final output:
    {
      "currency": "INR",
      "amounts": [],
      "status": "ok"
    }
    """
    # TODO: assemble and compute final confidences and provenance
    return {"currency": currency, "amounts": [], "status": "ok"}
