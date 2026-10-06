"""
explainer.py
------------
Task 3 – Intelligent Contradiction Explanation & Ranking.

Provides:
  - Contradiction confidence scoring (rule-based, NOT a calibrated probability)
  - Explanation generation based on similarity + sentiment polarity
  - Input validation and graceful error handling
  - Ranking of detected contradiction pairs

Confidence Score Methodology
-----------------------------
The confidence score is a prototype heuristic defined as:

    confidence = cosine_similarity  (when Positive ↔ Negative and similarity ≥ threshold)

This is NOT a statistically calibrated probability.  It is an indicator of
how strongly two reviews appear to discuss the same product aspect while
expressing opposite sentiments.  Higher cosine similarity → higher confidence
that the reviews are discussing the same topic.

Score interpretation:
  0.90 – 1.00  Very High    – Almost certainly discussing the same aspect
  0.75 – 0.89  High         – Strongly related content, opposing opinions
  0.60 – 0.74  Moderate     – Topically related but less specific overlap
  < 0.60       Low / None   – Insufficient semantic similarity

IMPORTANT: This module does NOT verify factual claims.  It detects potential
contradictions based on semantic similarity and sentiment polarity only.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_SENTIMENTS = {"Positive", "Negative", "Neutral"}
MIN_TEXT_LEN = 5       # characters – anything shorter is considered empty
MAX_RATING = 5.0
MIN_RATING = 1.0


# ---------------------------------------------------------------------------
# Input Validation
# ---------------------------------------------------------------------------

def validate_review(review_text: Optional[str], rating: Optional[float]) -> Tuple[bool, str]:
    """
    Validate a single review's text and rating.

    Parameters
    ----------
    review_text : str or None
        Raw review text.
    rating : float or None
        Star rating value.

    Returns
    -------
    (is_valid, reason)
        is_valid – True if the review can be used for comparison.
        reason   – Human-readable explanation when invalid.
    """
    if review_text is None:
        return False, "Review text is missing (None)."
    if not isinstance(review_text, str):
        return False, f"Review text is not a string (got {type(review_text).__name__})."
    cleaned = re.sub(r"\s+", " ", review_text).strip()
    if len(cleaned) < MIN_TEXT_LEN:
        return False, "Review text is empty or too short to analyse."
    if rating is None:
        return False, "Rating is missing (None)."
    try:
        rating = float(rating)
    except (TypeError, ValueError):
        return False, f"Invalid rating value: {rating!r}."
    if not (MIN_RATING <= rating <= MAX_RATING):
        return False, f"Rating {rating} is outside the valid range {MIN_RATING}–{MAX_RATING}."
    return True, "OK"


def validate_pair(
    review_1: Optional[str],
    rating_1: Optional[float],
    review_2: Optional[str],
    rating_2: Optional[float],
    asin_1: Optional[str] = None,
    asin_2: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Validate a pair of reviews before attempting contradiction detection.

    Parameters
    ----------
    review_1, review_2 : str or None
        Review text strings.
    rating_1, rating_2 : float or None
        Star ratings for each review.
    asin_1, asin_2 : str or None, optional
        Product ASINs.  If both supplied, checks they match.

    Returns
    -------
    (is_valid, reason)
    """
    ok1, msg1 = validate_review(review_1, rating_1)
    if not ok1:
        return False, f"Review 1 is invalid: {msg1}"

    ok2, msg2 = validate_review(review_2, rating_2)
    if not ok2:
        return False, f"Review 2 is invalid: {msg2}"

    # Duplicate review check (exact text match)
    if isinstance(review_1, str) and isinstance(review_2, str):
        if review_1.strip() == review_2.strip():
            return False, "The two reviews are identical – no contradiction possible."

    # Product ASIN consistency check (optional)
    if asin_1 is not None and asin_2 is not None:
        if str(asin_1).strip() != str(asin_2).strip():
            return False, (
                f"Reviews belong to different products "
                f"('{asin_1}' vs '{asin_2}')."
            )

    return True, "OK"


# ---------------------------------------------------------------------------
# Confidence Scoring
# ---------------------------------------------------------------------------

def compute_confidence(
    sentiment_1: str,
    sentiment_2: str,
    similarity_score: float,
    threshold: float = 0.60,
) -> Tuple[float, str]:
    """
    Compute a prototype contradiction confidence score and verdict.

    The score is the cosine similarity clamped to [0, 1] and only returned
    when the pair meets the full contradiction criteria.  All other cases
    return 0.0 with an appropriate verdict.

    Parameters
    ----------
    sentiment_1, sentiment_2 : str
        'Positive', 'Negative', or 'Neutral'.
    similarity_score : float
        Cosine similarity in [-1, 1].
    threshold : float
        Minimum similarity to consider reviews topically related.

    Returns
    -------
    (confidence, verdict)
        confidence – float in [0, 1].  0.0 means no contradiction.
        verdict    – One of:
                     'contradiction'       Meets all criteria.
                     'same_sentiment'      Sentiment not opposing.
                     'neutral_involved'    One or both reviews are Neutral.
                     'low_similarity'      Similarity below threshold.
                     'unknown_sentiment'   Unrecognised sentiment label.
    """
    # Guard against unknown sentiment labels
    for s in (sentiment_1, sentiment_2):
        if s not in VALID_SENTIMENTS:
            return 0.0, "unknown_sentiment"

    # Neutral sentiment → insufficient evidence for contradiction
    if "Neutral" in (sentiment_1, sentiment_2):
        return 0.0, "neutral_involved"

    # Same-sentiment pairs → no contradiction
    if sentiment_1 == sentiment_2:
        return 0.0, "same_sentiment"

    # Opposite sentiment confirmed – check similarity gate
    sim = max(0.0, min(1.0, float(similarity_score)))  # clamp to [0, 1]
    if sim < threshold:
        return 0.0, "low_similarity"

    # Full contradiction: confidence = the cosine similarity itself
    return round(sim, 4), "contradiction"


def confidence_label(confidence: float) -> str:
    """
    Map a confidence score to a human-readable tier label.

    Parameters
    ----------
    confidence : float
        Value in [0, 1] from compute_confidence().

    Returns
    -------
    str
        One of 'Very High', 'High', 'Moderate', 'Low'.
    """
    if confidence >= 0.90:
        return "Very High"
    elif confidence >= 0.75:
        return "High"
    elif confidence >= 0.60:
        return "Moderate"
    else:
        return "Low"


# ---------------------------------------------------------------------------
# Explanation Generation
# ---------------------------------------------------------------------------

def generate_explanation(
    verdict: str,
    sentiment_1: str,
    sentiment_2: str,
    similarity_score: float,
    confidence: float,
) -> str:
    """
    Generate a simple rule-based natural-language explanation for a review pair.

    Parameters
    ----------
    verdict : str
        The verdict string returned by compute_confidence().
    sentiment_1, sentiment_2 : str
        Sentiment labels for each review.
    similarity_score : float
        Cosine similarity between the two review embeddings.
    confidence : float
        Confidence score from compute_confidence().

    Returns
    -------
    str
        A one-to-two sentence human-readable explanation.
    """
    sim_pct = f"{similarity_score * 100:.0f}%"

    if verdict == "contradiction":
        tier = confidence_label(confidence)
        if confidence >= 0.90:
            return (
                f"Both reviews have very high semantic similarity ({sim_pct}) and express "
                f"directly opposite sentiments ({sentiment_1} vs {sentiment_2}). "
                f"This is a {tier.lower()}-confidence potential contradiction."
            )
        elif confidence >= 0.75:
            return (
                f"Both reviews appear to discuss the same product aspect "
                f"(similarity {sim_pct}) but reach opposing conclusions "
                f"({sentiment_1} vs {sentiment_2}). "
                f"This is a {tier.lower()}-confidence potential contradiction."
            )
        else:
            return (
                f"The reviews share moderate semantic overlap ({sim_pct}) and "
                f"express opposite sentiments ({sentiment_1} vs {sentiment_2}). "
                f"This is a {tier.lower()}-confidence potential contradiction."
            )

    elif verdict == "same_sentiment":
        return (
            f"Both reviews express the same sentiment ({sentiment_1}), "
            f"so no contradiction was detected despite their similarity ({sim_pct})."
        )

    elif verdict == "neutral_involved":
        involved = "Review 1" if sentiment_1 == "Neutral" else "Review 2"
        if sentiment_1 == "Neutral" and sentiment_2 == "Neutral":
            involved = "Both reviews"
        return (
            f"{involved} has a Neutral sentiment (3-star rating), which provides "
            f"insufficient evidence to declare a contradiction."
        )

    elif verdict == "low_similarity":
        return (
            f"The reviews are not semantically similar enough (similarity {sim_pct}) "
            f"to indicate they are discussing the same product aspect, "
            f"so no contradiction was detected."
        )

    elif verdict == "unknown_sentiment":
        return (
            "One or both reviews have an unrecognised sentiment label. "
            "Cannot determine whether a contradiction exists."
        )

    # Fallback (should not normally be reached)
    return "Unable to generate an explanation for this review pair."


# ---------------------------------------------------------------------------
# Enrichment helper (used by contradiction.py)
# ---------------------------------------------------------------------------

def enrich_pair(pair: Dict[str, Any], threshold: float = 0.60) -> Dict[str, Any]:
    """
    Add confidence score, tier label, and explanation to an existing pair dict.

    This is called by contradiction.detect_contradictions() for every candidate
    pair that passes the initial filters, but it can also be applied to
    non-contradiction pairs for full diagnostic output.

    Parameters
    ----------
    pair : dict
        A record with keys: sentiment_1, sentiment_2, similarity_score,
        and optionally contradiction_detected.
    threshold : float
        The similarity threshold used when detecting the pair.

    Returns
    -------
    dict
        The same dict with three extra keys added:
          confidence        float in [0, 1]
          confidence_label  str   ('Very High', 'High', 'Moderate', 'Low')
          explanation       str
    """
    confidence, verdict = compute_confidence(
        pair["sentiment_1"],
        pair["sentiment_2"],
        pair["similarity_score"],
        threshold=threshold,
    )
    pair["confidence"] = confidence
    pair["confidence_label"] = confidence_label(confidence)
    pair["explanation"] = generate_explanation(
        verdict,
        pair["sentiment_1"],
        pair["sentiment_2"],
        pair["similarity_score"],
        confidence,
    )
    return pair


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def rank_pairs(pairs: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    """
    Sort contradiction pairs by confidence (descending), then similarity (descending).

    Parameters
    ----------
    pairs : list[dict]
        Pairs enriched by enrich_pair().

    Returns
    -------
    list[dict]
        Ranked list.
    """
    return sorted(
        pairs,
        key=lambda p: (p.get("confidence", 0.0), p.get("similarity_score", 0.0)),
        reverse=True,
    )
