
import math


def precision_at_k(recommended, relevant, k):
    """Fraction of the top-k recommendation slots that are relevant."""
    if k < 1:
        raise ValueError("k must be at least 1.")

    top_k = recommended[:k]
    relevant_set = set(relevant)
    hits = sum(1 for item in top_k if item in relevant_set)

    # Standard Precision@K uses k as the denominator.
    return hits / k


def recall_at_k(recommended, relevant, k):
    """Fraction of all known relevant items found in the top-k."""
    if k < 1:
        raise ValueError("k must be at least 1.")

    relevant_set = set(relevant)

    if not relevant_set:
        return 0.0

    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant_set)

    return hits / len(relevant_set)


def ndcg_at_k(recommended, relevant, k):
    """Compute NDCG@K using binary relevance labels."""
    if k < 1:
        raise ValueError("k must be at least 1.")

    relevant_set = set(relevant)
    top_k = recommended[:k]

    dcg = sum(
        1 / math.log2(rank + 2)
        for rank, item in enumerate(top_k)
        if item in relevant_set
    )

    ideal_hits = min(len(relevant_set), k)

    idcg = sum(
        1 / math.log2(rank + 2)
        for rank in range(ideal_hits)
    )

    if idcg == 0:
        return 0.0

    return dcg / idcg
