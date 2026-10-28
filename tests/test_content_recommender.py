
from src.content_recommender import ContentRecommender


def test_photography_recommends_photographer():
    recommender = ContentRecommender()
    results = recommender.recommend(["photography"], top_k=5)

    assert results
    assert results[0]["title"] == "Photographer"


def test_empty_profile_returns_no_recommendations():
    recommender = ContentRecommender()

    assert recommender.recommend([]) == []
    assert recommender.recommend([], []) == []


def test_sanskrit_returns_relevant_careers():
    recommender = ContentRecommender()
    results = recommender.recommend(["sanskrit"], top_k=5)

    titles = {result["title"] for result in results}

    assert titles
    assert titles.intersection({
        "Sanskrit Teacher",
        "Translator",
        "Linguistics Researcher",
        "Cultural Heritage Specialist",
    })


def test_top_k_limits_result_count():
    recommender = ContentRecommender()
    results = recommender.recommend(
        ["python", "programming"],
        top_k=3,
    )

    assert len(results) <= 3


def test_scores_are_sorted_descending():
    recommender = ContentRecommender()
    results = recommender.recommend(
        ["python", "programming"],
        top_k=5,
    )

    scores = [result["score"] for result in results]

    assert scores == sorted(scores, reverse=True)


def test_results_include_separate_scores():
    recommender = ContentRecommender()
    results = recommender.recommend(["sanskrit"], top_k=5)

    assert results
    for result in results:
        assert "cosine_similarity" in result
        assert "ranking_score" in result
        assert result["score"] == result["ranking_score"]
        assert 0.0 <= result["cosine_similarity"] <= 1.0


def test_sanskrit_teacher_ranks_first():
    recommender = ContentRecommender()
    results = recommender.recommend(["sanskrit"], top_k=5)

    assert results[0]["title"] == "Sanskrit Teacher"
    assert "sanskrit" in results[0]["matched_skills"]


def test_results_are_sorted_by_ranking_score():
    recommender = ContentRecommender()
    results = recommender.recommend(["python"], top_k=5)

    scores = [result["ranking_score"] for result in results]
    assert scores == sorted(scores, reverse=True)


from src.content_recommender import normalize_term


def test_normalize_known_synonyms():
    assert normalize_term("AI") == "artificial intelligence"
    assert normalize_term("ML") == "machine learning"
    assert normalize_term("coding") == "programming"
    assert normalize_term("photographer") == "photography"


def test_normalize_preserves_unknown_terms():
    assert normalize_term("gardening") == "gardening"
    assert normalize_term("  pottery  ") == "pottery"


def test_normalization_is_case_insensitive():
    assert normalize_term("CoDiNg") == "programming"
    assert normalize_term("PHOTOGRAPHER") == "photography"


def test_recommendations_include_explanations_and_missing_skills():
    recommender = ContentRecommender()
    results = recommender.recommend(["photography"], top_k=1)

    assert results
    result = results[0]

    assert result["title"] == "Photographer"
    assert "explanation" in result
    assert "photography" in result["explanation"]
    assert "missing_skills" in result
    assert "composition" in result["missing_skills"]


def test_explanation_handles_no_exact_matches():
    from src.content_recommender import build_explanation

    explanation = build_explanation([], [], [])

    assert explanation == (
        "This career has textual similarity to your profile."
    )


