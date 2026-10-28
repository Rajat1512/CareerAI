"""Evaluate CareerAI rankings on the manually labelled profiles.

Run from the project root:

    python -m src.run_evaluation              # default model
    python -m src.run_evaluation --ablation   # also compare scoring variants

Outputs (overwritten on each run):
    reports/evaluation_report.csv     per-profile results of the default model
    reports/evaluation_summary.json   macro averages, ablation, metadata
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from src.content_recommender import DEFAULT_WEIGHTS, ContentRecommender
from src.evaluation import ndcg_at_k, precision_at_k, recall_at_k


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROFILES_PATH = PROJECT_ROOT / "data" / "evaluation_profiles.json"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORT_CSV = REPORTS_DIR / "evaluation_report.csv"
SUMMARY_JSON = REPORTS_DIR / "evaluation_summary.json"

ABLATION_VARIANTS = {
    "tfidf_cosine_only": {"explicit": 0.0, "cosine": 1.0, "related": 0.0},
    "explicit_match_only": {"explicit": 1.0, "cosine": 0.0, "related": 0.0},
    "hybrid_without_related_terms": {"related": 0.0},
    "hybrid_default": {},
}


def load_profiles(path: Path = PROFILES_PATH):
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)
    if isinstance(data, list):  # older file format
        return data, 5, ""
    return data["profiles"], int(data.get("k", 5)), data.get("description", "")


def validate_profiles(profiles, recommender):
    titles = {career["title"] for career in recommender.careers}
    problems = []
    for profile in profiles:
        unknown = sorted(set(profile["relevant_careers"]) - titles)
        if unknown:
            problems.append(f"{profile['profile']}: {unknown}")
        if len(set(profile["relevant_careers"])) != len(profile["relevant_careers"]):
            problems.append(f"{profile['profile']}: duplicate labels")
    if problems:
        raise ValueError(
            "Evaluation labels do not match catalogue titles:\n" + "\n".join(problems)
        )


def evaluate(recommender, profiles, k):
    rows = []
    for profile in profiles:
        results = recommender.recommend(
            skills=profile["skills"],
            interests=profile.get("interests", []),
            top_k=k,
        )
        recommended = [result["title"] for result in results]
        relevant = profile["relevant_careers"]
        rows.append({
            "profile": profile["profile"],
            "skills": ", ".join(profile["skills"]),
            "interests": ", ".join(profile.get("interests", [])),
            "recommended": " | ".join(recommended),
            "relevant_in_top_k": " | ".join(t for t in recommended if t in relevant),
            "n_labelled_relevant": len(relevant),
            f"precision_at_{k}": round(precision_at_k(recommended, relevant, k), 4),
            f"recall_at_{k}": round(recall_at_k(recommended, relevant, k), 4),
            f"ndcg_at_{k}": round(ndcg_at_k(recommended, relevant, k), 4),
        })
    return rows


def macro_average(rows, k):
    n = len(rows) or 1
    return {
        metric: round(sum(row[f"{metric}_at_{k}"] for row in rows) / n, 4)
        for metric in ("precision", "recall", "ndcg")
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ablation", action="store_true",
                        help="Also evaluate alternative scoring weights.")
    args = parser.parse_args()

    profiles, k, description = load_profiles()
    recommender = ContentRecommender()
    validate_profiles(profiles, recommender)

    print(f"Catalogue: {recommender.dataset_source}")
    print(f"Validated labels for {len(profiles)} profiles against "
          f"{len(recommender.careers)} catalogue titles.")
    print(f"\nCareerAI Recommendation Evaluation (K={k})")
    print("=" * 60)

    rows = evaluate(recommender, profiles, k)
    for row in rows:
        print(f"\nProfile: {row['profile']}")
        print(f"  Recommended: {row['recommended']}")
        print(f"  Relevant in top {k}: {row['relevant_in_top_k'] or '-'} "
              f"(of {row['n_labelled_relevant']} labelled)")
        print(f"  P@{k} {row[f'precision_at_{k}']:.3f} | "
              f"R@{k} {row[f'recall_at_{k}']:.3f} | "
              f"NDCG@{k} {row[f'ndcg_at_{k}']:.3f}")

    summary = macro_average(rows, k)
    print("\n" + "=" * 60)
    print(f"Macro-average across {len(rows)} profiles")
    print(f"Precision@{k}: {summary['precision']:.3f}")
    print(f"Recall@{k}:    {summary['recall']:.3f}")
    print(f"NDCG@{k}:      {summary['ndcg']:.3f}")

    ablation = {}
    if args.ablation:
        print("\nAblation (macro-averages)")
        print(f"{'variant':32} {'P@k':>6} {'R@k':>6} {'NDCG':>6}")
        for name, overrides in ABLATION_VARIANTS.items():
            weights = {**DEFAULT_WEIGHTS, **overrides}
            variant = ContentRecommender(weights=weights)
            scores = macro_average(evaluate(variant, profiles, k), k)
            ablation[name] = {"weights": weights, **scores}
            print(f"{name:32} {scores['precision']:6.3f} "
                  f"{scores['recall']:6.3f} {scores['ndcg']:6.3f}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with REPORT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with SUMMARY_JSON.open("w", encoding="utf-8") as file:
        json.dump({
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "catalogue": recommender.dataset_source,
            "catalogue_records": len(recommender.careers),
            "k": k,
            "profiles": len(rows),
            "weights": recommender.weights,
            "macro_average": summary,
            "ablation": ablation,
            "label_note": description,
        }, file, indent=2)

    print(f"\nPer-profile report: {REPORT_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Summary:            {SUMMARY_JSON.relative_to(PROJECT_ROOT)}")
    print(
        "\nNote: a small, manually labelled set (one labeller). Use it to "
        "catch regressions, not as a measure of real-world accuracy. "
        f"Recall@{k} is capped below 1.0 when a profile has more than {k} "
        "labelled careers."
    )


if __name__ == "__main__":
    main()
