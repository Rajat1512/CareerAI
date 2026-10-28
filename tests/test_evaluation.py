
import pytest

from src.evaluation import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
)


def test_precision_at_k():
    recommended = ["A", "B", "C", "D"]
    relevant = {"A", "C"}

    assert precision_at_k(recommended, relevant, 2) == 0.5
    assert precision_at_k(recommended, relevant, 4) == 0.5


def test_recall_at_k():
    recommended = ["A", "B", "C", "D"]
    relevant = {"A", "C"}

    assert recall_at_k(recommended, relevant, 2) == 0.5
    assert recall_at_k(recommended, relevant, 4) == 1.0


def test_ndcg_rewards_better_ranking():
    relevant = {"A", "C"}

    good_ranking = ["A", "C", "B"]
    poor_ranking = ["B", "A", "C"]

    assert ndcg_at_k(good_ranking, relevant, 3) > ndcg_at_k(
        poor_ranking, relevant, 3
    )


def test_empty_recommendations():
    assert precision_at_k([], {"A"}, 5) == 0.0
    assert recall_at_k([], {"A"}, 5) == 0.0
    assert ndcg_at_k([], {"A"}, 5) == 0.0


def test_invalid_k_raises_error():
    with pytest.raises(ValueError):
        precision_at_k(["A"], {"A"}, 0)

    with pytest.raises(ValueError):
        recall_at_k(["A"], {"A"}, 0)

    with pytest.raises(ValueError):
        ndcg_at_k(["A"], {"A"}, 0)


def test_empty_relevant_set():
    assert recall_at_k(["A"], set(), 3) == 0.0
    assert ndcg_at_k(["A"], set(), 3) == 0.0


def test_precision_uses_k_as_denominator_when_fewer_items_returned():
    from src.evaluation import precision_at_k

    assert precision_at_k(["Photographer"], ["Photographer"], 5) == 0.2


def test_precision_is_zero_when_no_recommendations_returned():
    from src.evaluation import precision_at_k

    assert precision_at_k([], ["Photographer"], 5) == 0.0


def test_ndcg_is_one_when_relevant_items_are_ranked_ideally():
    from src.evaluation import ndcg_at_k

    assert ndcg_at_k(
        ["A", "B", "C"],
        ["A", "B"],
        3,
    ) == 1.0


def test_ndcg_penalizes_relevant_items_ranked_lower():
    from src.evaluation import ndcg_at_k

    ideal = ndcg_at_k(["A", "B", "C"], ["A", "B"], 3)
    lower = ndcg_at_k(["C", "A", "B"], ["A", "B"], 3)

    assert lower < ideal


def test_ndcg_is_zero_when_no_relevant_items_exist():
    from src.evaluation import ndcg_at_k

    assert ndcg_at_k(["A", "B"], [], 2) == 0.0
