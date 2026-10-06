"""
evaluate.py
-----------
Task 3 – Evaluation of the contradiction-detection system.

Contains:
  - A manually labelled evaluation dataset (25 review pairs)
  - Evaluation runner: Accuracy, Precision, Recall, F1-score
  - Example-prediction display (5 selected cases)

Labels
------
  1 = contradiction   (Positive ↔ Negative, semantically similar)
  0 = non-contradiction (same sentiment, low similarity, or neutral involved)

IMPORTANT: The evaluation uses the same lightweight rule-based system
(semantic similarity + sentiment polarity) as the main prototype.
It does NOT verify factual claims.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure project root is on path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.explainer import compute_confidence, generate_explanation, validate_pair
from src.preprocess import rating_to_sentiment
from src.similarity import encode_texts, cosine_similarity_pair

# ---------------------------------------------------------------------------
# Manually Labelled Dataset (25 pairs)
# ---------------------------------------------------------------------------
# Each record: review_1, rating_1, review_2, rating_2, expected_label
#   1 = contradiction  (Pos<->Neg, high similarity)
#   0 = non-contradiction

EVAL_DATASET: List[Dict[str, Any]] = [
    # ── True Contradictions (label = 1) ─────────────────────────────────────
    {
        "review_1": "The battery easily lasts two full days even with heavy use. Outstanding performance.",
        "rating_1": 5,
        "review_2": "The battery barely lasts five hours. Absolutely terrible battery life.",
        "rating_2": 1,
        "expected_label": 1,
        "note": "Classic contradiction – battery life",
    },
    {
        "review_1": "Sound quality is exceptional. Rich bass and crystal clear highs. Best headphones I have ever owned.",
        "rating_1": 5,
        "review_2": "Horrible sound quality. Tinny and distorted at any volume. Complete waste of money.",
        "rating_2": 1,
        "expected_label": 1,
        "note": "Audio quality contradiction",
    },
    {
        "review_1": "Build quality is superb. Very sturdy and feels premium.",
        "rating_1": 4,
        "review_2": "Cheap plastic build. Fell apart within a week of light use.",
        "rating_2": 2,
        "expected_label": 1,
        "note": "Build quality contradiction",
    },
    {
        "review_1": "Connects to Bluetooth instantly every time. Rock solid connection.",
        "rating_1": 5,
        "review_2": "Bluetooth drops every few minutes. Constant disconnections ruin the experience.",
        "rating_2": 1,
        "expected_label": 1,
        "note": "Bluetooth connectivity contradiction",
    },
    {
        "review_1": "Very comfortable to wear for long periods. Lightweight and soft ear cushions.",
        "rating_1": 5,
        "review_2": "Extremely uncomfortable after just 20 minutes. The ear cups hurt and the headband is too tight.",
        "rating_2": 2,
        "expected_label": 1,
        "note": "Comfort contradiction",
    },
    {
        "review_1": "Mouse works perfectly on any surface. No tracking issues whatsoever.",
        "rating_1": 5,
        "review_2": "Mouse loses tracking constantly, especially on dark surfaces. Unusable.",
        "rating_2": 1,
        "expected_label": 1,
        "note": "Mouse tracking contradiction",
    },
    {
        "review_1": "Setup was a breeze. Plug and play, worked immediately out of the box.",
        "rating_1": 5,
        "review_2": "Nightmare to set up. Spent hours with drivers and it still does not work properly.",
        "rating_2": 1,
        "expected_label": 1,
        "note": "Setup experience contradiction",
    },
    {
        "review_1": "Excellent value for the price. Performs like a product twice the cost.",
        "rating_1": 5,
        "review_2": "Terrible value. Overpriced junk that breaks after a month.",
        "rating_2": 1,
        "expected_label": 1,
        "note": "Value-for-money contradiction",
    },
    {
        "review_1": "Charges quickly. From zero to full in under an hour.",
        "rating_1": 4,
        "review_2": "Takes over three hours to charge. Charging is painfully slow.",
        "rating_2": 2,
        "expected_label": 1,
        "note": "Charging speed contradiction",
    },
    {
        "review_1": "Very quiet fan. Can barely hear it even under heavy load.",
        "rating_1": 5,
        "review_2": "Extremely loud fan noise. Sounds like a jet engine at medium load.",
        "rating_2": 2,
        "expected_label": 1,
        "note": "Fan noise contradiction",
    },
    {
        "review_1": "Works great and I haven't had to change batteries since I bought it three months ago.",
        "rating_1": 5,
        "review_2": "Bought this mouse and it just died after two months. Replaced batteries and it still doesn't work.",
        "rating_2": 1,
        "expected_label": 1,
        "note": "Real dataset pair – Logitech mouse",
    },
    {
        "review_1": "Easy to use, charges quickly and lasts a long time.",
        "rating_1": 5,
        "review_2": "Slower charger than I would like. Takes a whole day to charge and batteries don't last long.",
        "rating_2": 2,
        "expected_label": 1,
        "note": "Real dataset pair – EBL charger",
    },
    # ── True Non-Contradictions: Same Sentiment (label = 0) ─────────────────
    {
        "review_1": "Excellent product. Great battery life and comfortable design.",
        "rating_1": 5,
        "review_2": "Really happy with this purchase. Battery is fantastic and it feels great to wear.",
        "rating_2": 5,
        "expected_label": 0,
        "note": "Both positive – same sentiment",
    },
    {
        "review_1": "Terrible product. Battery died in two days and sound is awful.",
        "rating_1": 1,
        "review_2": "Very disappointing. Battery fails quickly and the audio quality is poor.",
        "rating_2": 2,
        "expected_label": 0,
        "note": "Both negative – same sentiment",
    },
    {
        "review_1": "I love the wireless range. Works across the entire house without dropouts.",
        "rating_1": 5,
        "review_2": "Wireless range is impressive. No connectivity problems from 30 feet away.",
        "rating_2": 5,
        "expected_label": 0,
        "note": "Both positive – wireless range",
    },
    {
        "review_1": "Screen quality is awful. Very washed out and dull colours.",
        "rating_1": 1,
        "review_2": "Display looks terrible. Colours are dull and brightness is not enough.",
        "rating_2": 2,
        "expected_label": 0,
        "note": "Both negative – display",
    },
    # ── True Non-Contradictions: Low Similarity (label = 0) ─────────────────
    {
        "review_1": "The carry case is very nice and compact.",
        "rating_1": 5,
        "review_2": "Disappointing sound. Muddy bass and no highs at all.",
        "rating_2": 1,
        "expected_label": 0,
        "note": "Low similarity – different topics (case vs. sound)",
    },
    {
        "review_1": "Quick delivery and good packaging.",
        "rating_1": 5,
        "review_2": "The product broke after a week of use.",
        "rating_2": 1,
        "expected_label": 0,
        "note": "Low similarity – shipping vs. durability",
    },
    {
        "review_1": "The colour options are great. I like the red variant.",
        "rating_1": 4,
        "review_2": "Battery dies so fast. Needs charging every few hours.",
        "rating_2": 2,
        "expected_label": 0,
        "note": "Low similarity – colour vs. battery",
    },
    # ── True Non-Contradictions: Neutral Sentiment Involved (label = 0) ──────
    {
        "review_1": "It is okay. Nothing special but gets the job done.",
        "rating_1": 3,
        "review_2": "Battery life is terrible. Barely lasts an hour.",
        "rating_2": 1,
        "expected_label": 0,
        "note": "Neutral review 1 – insufficient evidence",
    },
    {
        "review_1": "Amazing product. Best I have ever used.",
        "rating_1": 5,
        "review_2": "Average product. Does what it says but nothing impressive.",
        "rating_2": 3,
        "expected_label": 0,
        "note": "Neutral review 2 – insufficient evidence",
    },
    {
        "review_1": "It is fine. Neither great nor bad. Does the basic job.",
        "rating_1": 3,
        "review_2": "Pretty decent for the price. Works as expected.",
        "rating_2": 3,
        "expected_label": 0,
        "note": "Both neutral – no contradiction",
    },
    # ── Edge Cases (label = 0) ───────────────────────────────────────────────
    {
        "review_1": "The battery easily lasts two full days even with heavy use. Outstanding performance.",
        "rating_1": 5,
        "review_2": "The battery easily lasts two full days even with heavy use. Outstanding performance.",
        "rating_2": 5,
        "expected_label": 0,
        "note": "Duplicate reviews – no contradiction",
    },
    {
        "review_1": "Good.",
        "rating_1": 5,
        "review_2": "Bad.",
        "rating_2": 1,
        "expected_label": 0,
        "note": "Too short for meaningful similarity – edge case",
    },
    {
        "review_1": "Absolutely fantastic in every possible way. Surpassed all my expectations completely.",
        "rating_1": 5,
        "review_2": "Works fine for basic tasks. Not for power users though.",
        "rating_2": 3,
        "expected_label": 0,
        "note": "Positive vs Neutral – insufficient evidence",
    },
]


# ---------------------------------------------------------------------------
# Selected Example Cases for Display
# ---------------------------------------------------------------------------
# Indices into EVAL_DATASET that represent the 5 required example categories
EXAMPLE_INDICES = {
    "True Contradiction": 0,             # Battery life – clear contradiction
    "True Non-Contradiction (Same Sentiment)": 12,  # Both positive
    "Same Sentiment (Negative)": 13,     # Both negative
    "Low Similarity": 16,                # Different topics
    "Neutral / Insufficient Evidence": 19,  # Neutral involved
}


# ---------------------------------------------------------------------------
# Evaluation Runner
# ---------------------------------------------------------------------------

def _predict_label(
    review_1: str,
    rating_1: float,
    review_2: str,
    rating_2: float,
    threshold: float = 0.60,
) -> Tuple[int, float, str, str]:
    """
    Predict whether a pair is a contradiction (1) or not (0).

    Returns
    -------
    (predicted_label, confidence, verdict, explanation)
    """
    # --- Validate inputs first -------------------------------------------
    ok, msg = validate_pair(review_1, rating_1, review_2, rating_2)
    if not ok:
        explanation = f"Validation failed: {msg}"
        return 0, 0.0, "invalid_input", explanation

    # --- Convert ratings to sentiments ------------------------------------
    sentiment_1 = rating_to_sentiment(float(rating_1))
    sentiment_2 = rating_to_sentiment(float(rating_2))

    # --- Encode and compute similarity ------------------------------------
    try:
        embeddings = encode_texts([review_1, review_2], show_progress=False)
    except Exception as exc:
        explanation = f"Model loading or encoding failed: {exc}"
        return 0, 0.0, "model_error", explanation

    similarity = cosine_similarity_pair(embeddings[0], embeddings[1])

    # --- Compute confidence and verdict ----------------------------------
    confidence, verdict = compute_confidence(
        sentiment_1, sentiment_2, similarity, threshold=threshold
    )
    explanation = generate_explanation(
        verdict, sentiment_1, sentiment_2, similarity, confidence
    )

    predicted_label = 1 if verdict == "contradiction" else 0
    return predicted_label, confidence, verdict, explanation


def _compute_metrics(
    y_true: List[int],
    y_pred: List[int],
) -> Dict[str, float]:
    """Compute accuracy, precision, recall, F1 from binary lists."""
    assert len(y_true) == len(y_pred), "Length mismatch between true and predicted labels."

    tp = sum(t == 1 and p == 1 for t, p in zip(y_true, y_pred))
    tn = sum(t == 0 and p == 0 for t, p in zip(y_true, y_pred))
    fp = sum(t == 0 and p == 1 for t, p in zip(y_true, y_pred))
    fn = sum(t == 1 and p == 0 for t, p in zip(y_true, y_pred))

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "total": total,
    }


def run_evaluation(threshold: float = 0.60) -> None:
    """
    Run the full evaluation pipeline and print results.

    Parameters
    ----------
    threshold : float
        Cosine similarity threshold passed to compute_confidence().
    """
    print("\n" + "=" * 70)
    print("  TASK 3 – EVALUATION: CONTRADICTION DETECTION SYSTEM")
    print(f"  Dataset size: {len(EVAL_DATASET)} labelled pairs")
    print(f"  Similarity threshold: {threshold}")
    print("=" * 70)

    print("\n[evaluate] Encoding all evaluation review pairs ...")

    # Collect all texts for a single batch encode
    texts: List[str] = []
    for item in EVAL_DATASET:
        texts.append(item["review_1"])
        texts.append(item["review_2"])

    try:
        all_embeddings = encode_texts(texts, show_progress=False)
    except Exception as exc:
        print(f"[evaluate] FATAL: Could not load model: {exc}")
        return

    y_true: List[int] = []
    y_pred: List[int] = []
    predictions: List[Dict[str, Any]] = []

    for idx, item in enumerate(EVAL_DATASET):
        r1 = item["review_1"]
        r1_rating = item["rating_1"]
        r2 = item["review_2"]
        r2_rating = item["rating_2"]
        expected = item["expected_label"]

        # --- Validate ---
        ok, msg = validate_pair(r1, r1_rating, r2, r2_rating)
        if not ok:
            verdict = "invalid_input"
            confidence = 0.0
            explanation = f"Validation failed: {msg}"
            predicted = 0
        else:
            s1 = rating_to_sentiment(float(r1_rating))
            s2 = rating_to_sentiment(float(r2_rating))
            emb1 = all_embeddings[idx * 2]
            emb2 = all_embeddings[idx * 2 + 1]
            similarity = cosine_similarity_pair(emb1, emb2)
            confidence, verdict = compute_confidence(s1, s2, similarity, threshold)
            explanation = generate_explanation(verdict, s1, s2, similarity, confidence)
            predicted = 1 if verdict == "contradiction" else 0

        y_true.append(expected)
        y_pred.append(predicted)

        s1_label = rating_to_sentiment(float(r1_rating)) if ok else "N/A"
        s2_label = rating_to_sentiment(float(r2_rating)) if ok else "N/A"

        predictions.append({
            "idx": idx + 1,
            "note": item.get("note", ""),
            "expected": expected,
            "predicted": predicted,
            "correct": predicted == expected,
            "verdict": verdict,
            "confidence": confidence,
            "sentiment_1": s1_label,
            "sentiment_2": s2_label,
            "explanation": explanation,
            "review_1_snippet": r1[:80],
            "review_2_snippet": r2[:80],
        })

    # --- Compute metrics ---
    metrics = _compute_metrics(y_true, y_pred)

    # --- Print all predictions table ---
    print("\n" + "-" * 70)
    print("  ALL PREDICTIONS")
    print("-" * 70)
    header = f"{'#':>3}  {'Expected':>8}  {'Predicted':>9}  {'OK':>3}  {'Conf':>6}  Note"
    print(header)
    print("-" * 70)
    for p in predictions:
        tick = "OK " if p["correct"] else "MISS"
        conf_str = f"{p['confidence']:.2f}" if p["confidence"] > 0 else "  --"
        line = (
            f"{p['idx']:>3}  {p['expected']:>8}  {p['predicted']:>9}  "
            f"{tick:>4}  {conf_str:>6}  {p['note']}"
        )
        print(line)

    # --- Print metrics ---
    m = metrics
    print("\n" + "=" * 70)
    print("  EVALUATION METRICS")
    print("=" * 70)
    print(f"  Total pairs evaluated : {m['total']}")
    print(f"  True Positives (TP)   : {m['tp']}")
    print(f"  True Negatives (TN)   : {m['tn']}")
    print(f"  False Positives (FP)  : {m['fp']}")
    print(f"  False Negatives (FN)  : {m['fn']}")
    print(f"  Accuracy              : {m['accuracy']:.4f}  ({m['accuracy']*100:.1f}%)")
    print(f"  Precision             : {m['precision']:.4f}  ({m['precision']*100:.1f}%)")
    print(f"  Recall                : {m['recall']:.4f}  ({m['recall']*100:.1f}%)")
    print(f"  F1-Score              : {m['f1']:.4f}  ({m['f1']*100:.1f}%)")
    print("=" * 70)

    # --- Print 5 selected example predictions ---
    print("\n" + "=" * 70)
    print("  SELECTED EXAMPLE PREDICTIONS (5 CASES)")
    print("=" * 70)

    for category, dataset_idx in EXAMPLE_INDICES.items():
        p = predictions[dataset_idx]
        print(f"\n  Category : {category}")
        print(f"  Note     : {p['note']}")
        print(f"  Review 1 : {p['review_1_snippet']}...")
        print(f"  Review 2 : {p['review_2_snippet']}...")
        print(f"  Sentiment: {p['sentiment_1']} vs {p['sentiment_2']}")
        print(f"  Expected : {'Contradiction' if p['expected'] else 'Non-Contradiction'}")
        print(f"  Predicted: {'Contradiction' if p['predicted'] else 'Non-Contradiction'}")
        conf_display = f"{p['confidence']*100:.0f}%" if p['confidence'] > 0 else "N/A"
        print(f"  Confidence    : {conf_display}")
        print(f"  Explanation   : {p['explanation']}")
        result = "CORRECT" if p["correct"] else "INCORRECT"
        print(f"  Result   : {result}")
        print("-" * 70)


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_evaluation(threshold=0.60)
