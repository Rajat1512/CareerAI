
import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CAREERS_FILE = PROJECT_ROOT / "data" / "careers.json"


def normalize_text(value):
    """Lowercase text and normalize whitespace."""
    if value is None:
        return ""

    text = str(value).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_list(values):
    """Normalize list entries and remove empty or duplicate values."""
    if not isinstance(values, list):
        return []

    normalized = []
    seen = set()

    for value in values:
        item = normalize_text(value)

        if item and item not in seen:
            normalized.append(item)
            seen.add(item)

    return normalized


def load_and_prepare_careers(file_path=CAREERS_FILE):
    """Load career records and prepare consistent text for recommendation."""
    with open(file_path, "r", encoding="utf-8-sig") as file:
        careers = json.load(file)

    if not isinstance(careers, list):
        raise ValueError("Career data must be a JSON list.")

    prepared_careers = []
    seen_titles = set()

    for career in careers:
        if not isinstance(career, dict):
            raise ValueError("Each career must be a JSON object.")

        title = str(career.get("title", "")).strip()

        if not title:
            raise ValueError("Every career must have a title.")

        normalized_title = normalize_text(title)

        if normalized_title in seen_titles:
            raise ValueError(f"Duplicate career title: {title}")

        seen_titles.add(normalized_title)

        skills = normalize_list(career.get("skills", []))
        interests = normalize_list(career.get("interests", []))
        learning = normalize_list(career.get("learning", []))
        description = str(career.get("description", "")).strip()

        # Combine meaningful career information into searchable text.
        document = " ".join([
            normalize_text(title),
            normalize_text(description),
            " ".join(skills),
            " ".join(interests),
            " ".join(learning),
        ])

        prepared_careers.append({
            **career,
            "title": title,
            "description": description,
            "skills": skills,
            "interests": interests,
            "learning": learning,
            "search_text": document,
        })

    return prepared_careers
