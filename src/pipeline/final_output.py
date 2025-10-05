"""
Final output assembly: produce the final JSON with currency, provenance, and amounts.
Functions:
- assemble_final_output(classified_amounts: List[Dict], currency: str) -> Dict
"""

from typing import Dict, List


def assemble_final_output(
    classified_amounts: List[Dict], currency: str = "INR"
) -> Dict:
    """
    Assembles the final output JSON with currency, labeled amounts, and status.
    Each amount dict should have: label, amount, confidence, context.
    """
    output = {"currency": currency, "amounts": [], "status": "ok"}
    if not classified_amounts:
        output["status"] = "no_amounts_found"
        return output
    # Flatten if input is wrapped in {"amounts": ...}
    if isinstance(classified_amounts, dict) and "amounts" in classified_amounts:
        amounts = classified_amounts["amounts"]
    else:
        amounts = classified_amounts
    output["amounts"] = [
        {
            "label": a.get("label", "other"),
            "amount": a.get("amount"),
            "confidence": a.get("confidence", 0.0),
            "context": a.get("context", ""),
        }
        for a in amounts
    ]
    return output
