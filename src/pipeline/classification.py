"""
Classification stage: use surrounding text to map numbers to labels like
total_bill, paid, due, discount, etc.
Functions:
- classify_amounts(tokens_with_context: List[Dict]) -> Dict
"""

from typing import Dict, List


def classify_amounts(normalized_amounts: List[int], contexts: List[str]) -> Dict:
    """
    Classifies each amount using context keywords.
    Returns dict with list of dicts: [{label, amount, confidence, context}]
    """
    LABEL_KEYWORDS = {
        "total_bill": ["total", "bill", "amount payable", "invoice total"],
        "paid": ["paid", "received", "payment"],
        "due": ["due", "balance", "outstanding"],
        "discount": ["discount", "rebate", "offer"],
        "tax": ["tax", "gst", "vat"],
    }
    results = []
    for amt, ctx in zip(normalized_amounts, contexts):
        label = "other"
        confidence = 0.5
        ctx_lower = ctx.lower()
        for lbl, keywords in LABEL_KEYWORDS.items():
            for kw in keywords:
                if kw in ctx_lower:
                    label = lbl
                    confidence = 0.95 if kw == keywords[0] else 0.8
                    break
            if label != "other":
                break
        results.append(
            {"label": label, "amount": amt, "confidence": confidence, "context": ctx}
        )
    return {
        "amounts": results,
        "confidence": (
            float(sum(r["confidence"] for r in results)) / len(results)
            if results
            else 0.0
        ),
    }
