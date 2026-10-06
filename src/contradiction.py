"""
contradiction.py
----------------
Core contradiction-detection logic.

A review pair is considered a **potential contradiction** when ALL of the
following conditions are met:

  1. Both reviews belong to the same product (same parent_asin).
  2. Their semantic similarity score >= similarity_threshold
     (they discuss the same topic / aspect).
  3. Their sentiment labels are OPPOSITE:
       Positive <-> Negative   (Neutral is excluded from direct contradictions)

Task 3 Enhancement
------------------
Each detected pair is now enriched with:
  - confidence        : float in [0, 1] – how strongly the pair meets criteria
  - confidence_label  : str   – 'Very High' / 'High' / 'Moderate' / 'Low'
  - explanation       : str   – rule-based natural-language explanation

Pairs are ranked by confidence (descending), then similarity (descending).

This is a prototype based on semantic similarity + sentiment polarity.
It does NOT perform fact-verification.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from itertools import combinations
from typing import List, Dict, Any

from src.similarity import encode_texts, cosine_similarity_pair
from src.explainer import enrich_pair, rank_pairs, validate_pair


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_contradictions(
    df: pd.DataFrame,
    similarity_threshold: float = 0.60,
    max_pairs_per_product: int = 5,
    top_n: int = 20,
) -> List[Dict[str, Any]]:
    """
    Detect contradictory review pairs across all products in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame from preprocess.preprocess().
        Must contain: parent_asin, product_title, review_text, sentiment, rating.
    similarity_threshold : float
        Minimum cosine similarity to consider two reviews topically related.
        Range [0, 1]. Default 0.60.
    max_pairs_per_product : int
        Maximum number of contradictory pairs to keep per product.
    top_n : int
        Total maximum number of pairs to return, ranked by confidence (desc).

    Returns
    -------
    list[dict]
        List of contradiction records, each with keys:
          product_title, parent_asin, review_1, review_2,
          sentiment_1, sentiment_2, similarity_score, contradiction_detected,
          confidence, confidence_label, explanation.
    """
    print(f"[contradiction] Encoding {len(df):,} reviews ...")
    embeddings = encode_texts(df["review_text"].tolist(), show_progress=True)

    results: List[Dict[str, Any]] = []
    groups = df.groupby("parent_asin")

    print(f"[contradiction] Scanning {len(groups):,} products for contradictions ...")

    for asin, group in groups:
        # Skip products with only one review (nothing to compare)
        if len(group) < 2:
            continue

        # Retrieve pre-computed embeddings for this group's rows
        orig_indices = group.index.tolist()
        group_embs = embeddings[orig_indices]
        group = group.reset_index(drop=True)

        product_title = group["product_title"].iloc[0]
        pair_count = 0

        for i, j in combinations(range(len(group)), 2):
            if pair_count >= max_pairs_per_product:
                break

            sent_i = group["sentiment"].iloc[i]
            sent_j = group["sentiment"].iloc[j]

            # Only Positive <-> Negative pairs are direct contradictions
            if not _is_opposite_sentiment(sent_i, sent_j):
                continue

            # Validate both reviews before computing similarity
            r1 = group["review_text"].iloc[i]
            r2 = group["review_text"].iloc[j]
            rating_i = group["rating"].iloc[i] if "rating" in group.columns else None
            rating_j = group["rating"].iloc[j] if "rating" in group.columns else None
            ok, _ = validate_pair(r1, rating_i, r2, rating_j, asin_1=asin, asin_2=asin)
            if not ok:
                continue

            sim = cosine_similarity_pair(group_embs[i], group_embs[j])

            if sim < similarity_threshold:
                continue  # Not topically similar enough

            pair = {
                "product_title": product_title,
                "parent_asin": asin,
                "review_1": r1,
                "review_2": r2,
                "sentiment_1": sent_i,
                "sentiment_2": sent_j,
                "similarity_score": round(float(sim), 4),
                "contradiction_detected": True,
            }
            # Task 3: enrich with confidence, label, and explanation
            pair = enrich_pair(pair, threshold=similarity_threshold)
            results.append(pair)
            pair_count += 1

    # Task 3: rank by confidence descending, then similarity descending
    results = rank_pairs(results)

    print(f"[contradiction] Found {len(results)} contradictory pair(s) total.")
    return results[:top_n]


def format_results(pairs: List[Dict[str, Any]]) -> str:
    """
    Pretty-print contradiction results as a readable text report.

    Parameters
    ----------
    pairs : list[dict]
        Output of detect_contradictions() (Task 3 enriched format).

    Returns
    -------
    str
        Formatted multi-line string report.
    """
    if not pairs:
        return "No contradictions detected with the current threshold."

    lines = [
        "=" * 70,
        "  AI-BASED PRODUCT REVIEW CONTRADICTION DETECTION – RESULTS",
        f"  Total contradictory pairs found: {len(pairs)}",
        "=" * 70,
    ]

    for idx, pair in enumerate(pairs, start=1):
        status = "[DETECTED]" if pair.get("contradiction_detected") else "[NONE]"
        conf = pair.get("confidence", 0.0)
        conf_display = f"{conf * 100:.0f}%" if conf > 0 else "N/A"
        tier = pair.get("confidence_label", "")
        explanation = pair.get("explanation", "")

        lines += [
            f"\n[Pair {idx}]",
            f"  Product           : {pair['product_title'][:80]}",
            f"  ASIN              : {pair['parent_asin']}",
            f"  Similarity Score  : {pair['similarity_score']:.4f}",
            f"  Contradiction     : {status}",
            f"  Confidence        : {conf_display}  ({tier})",
            f"  Explanation       : {explanation}",
            f"\n  Review 1 [{pair['sentiment_1']}]:",
            f"    {pair['review_1'][:200]}",
            f"\n  Review 2 [{pair['sentiment_2']}]:",
            f"    {pair['review_2'][:200]}",
            "-" * 70,
        ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_opposite_sentiment(s1: str, s2: str) -> bool:
    """Return True only for strict Positive <-> Negative pairs."""
    opposites = {("Positive", "Negative"), ("Negative", "Positive")}
    return (s1, s2) in opposites
