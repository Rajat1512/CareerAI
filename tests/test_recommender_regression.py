"""Regression tests for ranking quality and output contract on the full catalogue."""

import math

import pytest

from src.catalogue import SOURCE_CURATED, SOURCE_MERGED
from src.content_recommender import (
    ContentRecommender,
    skill_matches_term,
    stem,
)


REQUIRED_FIELDS = {
    "id", "title", "description", "score", "ranking_score",
    "cosine_similarity", "matched_skills", "matched_interests",
    "matched_terms", "missing_skills", "explanation", "source", "skills",
}


@pytest.fixture(scope="module")
def recommender():
    return ContentRecommender()


def titles(results):
    return [result["title"] for result in results]


def test_default_catalogue_is_full_esco_plus_curated(recommender):
    assert len(recommender.careers) > 3039
    assert "ESCO" in recommender.dataset_source


def test_photographer_is_merged_record(recommender):
    result = recommender.recommend(["photography"], top_k=1)[0]

    assert result["title"] == "Photographer"
    assert result["source"] == SOURCE_MERGED
    assert result["esco_uri"]


def test_sanskrit_teacher_is_curated_record(recommender):
    result = recommender.recommend(["sanskrit"], top_k=1)[0]

    assert result["source"] == SOURCE_CURATED


@pytest.mark.parametrize(
    "skills, interests, expected_any, top_n",
    [
        (["python", "programming"], ["technology"], {"Software Developer"}, 3),
        (["machine learning", "python"], ["artificial intelligence"],
         {"AI/ML Engineer", "Machine Learning Engineer",
          "Artificial intelligence engineer", "Machine Learning Research Scientist"}, 3),
        (["python", "statistics", "predictive modelling"], ["data science"],
         {"Data Scientist", "Statistician", "Predictive Analytics Scientist"}, 3),
        (["sql", "data analysis"], ["business"],
         {"Data Analyst", "Business Analyst", "Business Intelligence Analyst",
          "Research Data Analyst"}, 3),
        (["translation"], ["languages"], {"Translator"}, 3),
        (["cooking"], [], {"Chef", "Cook"}, 2),
        (["nursing", "patient care"], [], {"Nurse responsible for general care",
                                           "Specialist nurse", "Nurse assistant"}, 5),
        (["photojournalism"], [], {"Photojournalist"}, 3),
    ],
)
def test_expected_careers_rank_near_top(recommender, skills, interests, expected_any, top_n):
    results = recommender.recommend(skills, interests, top_k=top_n)

    assert expected_any.intersection(titles(results)), titles(results)


def test_every_result_has_consistent_fields(recommender):
    results = recommender.recommend(["python"], ["research"], top_k=10)

    assert results
    for result in results:
        assert REQUIRED_FIELDS.issubset(result)
        assert math.isfinite(result["score"])
        assert 0.0 <= result["score"] <= 1.0
        assert isinstance(result["missing_skills"], list)
        assert result["explanation"].endswith(".")


def test_matched_and_missing_skills_belong_to_the_career(recommender):
    for result in recommender.recommend(["photography", "lighting"], top_k=5):
        skills = set(result["skills"])
        assert set(result["matched_skills"]) <= skills
        assert set(result["missing_skills"]) <= skills
        assert not set(result["matched_skills"]) & set(result["missing_skills"])


def test_programming_does_not_claim_other_languages(recommender):
    for result in recommender.recommend(["programming"], top_k=10):
        assert not any(
            skill.startswith(("Java (", "Prolog (", "Ruby ("))
            for skill in result["matched_skills"]
        ), result["matched_skills"]


def test_rankings_are_deterministic(recommender):
    first = recommender.recommend(["writing", "editing"], ["literature"], top_k=10)
    second = recommender.recommend(["writing", "editing"], ["literature"], top_k=10)

    assert first == second


def test_duplicate_and_differently_formatted_inputs_are_equivalent(recommender):
    plain = recommender.recommend(["python"], top_k=5)
    noisy = recommender.recommend(["Python", " python ", "PYTHON"], top_k=5)

    assert titles(plain) == titles(noisy)


def test_comma_separated_string_is_split(recommender):
    assert titles(recommender.recommend("python, programming", top_k=5)) == titles(
        recommender.recommend(["python", "programming"], top_k=5)
    )


def test_unknown_terms_return_no_results(recommender):
    assert recommender.recommend(["qwxzvbnm"], top_k=5) == []


def test_unknown_terms_are_reported(recommender):
    _, _, unmatched = recommender.analyse_profile(["qwxzvbnm", "python"])

    assert unmatched == ["qwxzvbnm"]


def test_whitespace_only_profile_returns_nothing(recommender):
    assert recommender.recommend(["   ", ""], ["  "]) == []
    assert recommender.recommend(None) == []


@pytest.mark.parametrize("bad_top_k", [0, -3, "5", 2.5, True, None])
def test_invalid_top_k_raises_value_error(recommender, bad_top_k):
    with pytest.raises(ValueError):
        recommender.recommend(["python"], top_k=bad_top_k)


def test_large_top_k_is_safe(recommender):
    results = recommender.recommend(["sanskrit"], top_k=100_000)

    assert 0 < len(results) <= len(recommender.careers)


def test_whole_word_matching():
    assert skill_matches_term("Python (computer programming)", "python")
    assert skill_matches_term("edit photographs", "photograph")
    assert not skill_matches_term("design database scheme", "data")
    assert not skill_matches_term("art collections", "cart")


@pytest.mark.parametrize(
    "a, b",
    [
        ("photography", "photographer"),
        ("programming", "programmer"),
        ("teaching", "teacher"),
        ("nursing", "nurse"),
        ("linguistics", "linguist"),
        ("analysis", "analyse"),
        ("statistics", "statistical"),
        ("journalism", "journalist"),
        ("tourism", "tourist"),
    ],
)
def test_stemmer_unifies_word_variants(a, b):
    assert stem(a) == stem(b)


@pytest.mark.parametrize(
    "a, b",
    [("engineering", "engine"), ("communication", "community"), ("data", "database")],
)
def test_stemmer_keeps_unrelated_words_apart(a, b):
    assert stem(a) != stem(b)


def test_small_provided_catalogue_still_works():
    careers = [
        {"title": "Baker", "description": "Bakes bread.",
         "skills": ["baking", "bread"], "interests": ["food"]},
        {"title": "Coder", "description": "Writes software.",
         "skills": ["python"], "interests": ["technology"]},
    ]
    recommender = ContentRecommender(careers=careers)

    assert recommender.dataset_source == "Provided catalogue"
    assert titles(recommender.recommend(["baking"], top_k=1)) == ["Baker"]


def test_empty_provided_catalogue_is_rejected():
    with pytest.raises(ValueError):
        ContentRecommender(careers=[])


def test_skill_gap_and_title_search(recommender):
    career = recommender.search_titles("photographer", limit=1)[0]
    have, missing = recommender.skill_gap(career, "photography, lighting")

    assert career["title"] == "Photographer"
    assert "photography" in have and "lighting" in have
    assert "composition" in missing
