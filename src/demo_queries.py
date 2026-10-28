"""Print recommendations for a few example profiles.

Run from the project root:  python -m src.demo_queries
"""

from src.content_recommender import ContentRecommender


def main():
    recommender = ContentRecommender()

    print("Dataset:", recommender.dataset_source)
    print("Occupation records:", len(recommender.careers))

    queries = [
        {
            "skills": ["Python", "statistics", "predictive modelling"],
            "interests": ["machine learning", "data science"],
        },
        {
            "skills": ["PhD student in prediction"],
            "interests": ["research"],
        },
        {
            "skills": ["Java", "programming"],
            "interests": ["software development"],
        },
    ]

    for query in queries:
        print("\n" + "=" * 60)
        print("Profile:", query)

        results = recommender.recommend(
            skills=query["skills"],
            interests=query["interests"],
            top_k=5,
        )

        if not results:
            print("No matching careers found.")
            continue

        for rank, result in enumerate(results, start=1):
            print(f"\n{rank}. {result['title']}")
            print(
                f"   Cosine similarity: "
                f"{result['cosine_similarity']:.4f}"
            )
            print(
                f"   Ranking score: "
                f"{result['ranking_score']:.4f}"
            )
            print(
                "   Matched skills:",
                result["matched_skills"][:8],
            )
            print("   Explanation:", result["explanation"])


if __name__ == "__main__":
    main()
