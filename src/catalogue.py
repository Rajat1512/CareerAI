"""Load, validate and merge the CareerAI career catalogues.

The application catalogue is built from two sources:

* ESCO (data/careers_esco.json) - the official European occupation taxonomy,
  produced by ``python -m src.import_esco``. Required.
* CareerAI curated careers (data/careers.json) - 49 hand-written records from
  the original project. Optional, but loaded when present.

A curated career whose title exactly matches an ESCO preferred label (for
example "Photographer") is merged into that ESCO record instead of being
added twice. The ESCO identifier, description and skills are kept, and the
curated skills/interests are stored in separate fields so the source of every
skill stays visible. Curated careers with no exact ESCO equivalent are added
as separate records marked as curated (not official ESCO records).
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from src.preprocessing import CAREERS_FILE, load_and_prepare_careers


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ESCO_FILE = PROJECT_ROOT / "data" / "careers_esco.json"
ESCO_REPORT_FILE = PROJECT_ROOT / "data" / "esco_import_report.json"

SOURCE_ESCO = "ESCO"
SOURCE_CURATED = "CareerAI curated"
SOURCE_MERGED = "ESCO + CareerAI curated"

ESCO_RESTORE_HELP = (
    "The ESCO occupation catalogue (data/careers_esco.json) is missing or "
    "invalid. Restore it from the project repository, or rebuild it:\n"
    "  1. Put occupations_en.csv, skills_en.csv and "
    "occupationSkillRelations_en.csv (ESCO v1.2.1, English, CSV) in "
    "data/esco_raw/\n"
    "  2. Run: python -m src.import_esco"
)


class DatasetError(RuntimeError):
    """Raised when a required dataset is missing or malformed."""


def _clean_text(value) -> str:
    return " ".join(str(value or "").split())


def _clean_list(values) -> list[str]:
    if not isinstance(values, list):
        return []
    cleaned = []
    seen = set()
    for value in values:
        text = _clean_text(value)
        if text and text.casefold() not in seen:
            cleaned.append(text)
            seen.add(text.casefold())
    return cleaned


def title_key(title: str) -> str:
    """Comparison key used to detect duplicate occupation titles."""
    return re.sub(r"\s+", " ", str(title).casefold()).strip()


def display_title(title: str) -> str:
    """ESCO labels are lower case; capitalise only the first letter."""
    title = _clean_text(title)
    return title[:1].upper() + title[1:] if title else title


def _read_json(path: Path):
    try:
        with path.open("r", encoding="utf-8-sig") as file:
            return json.load(file)
    except FileNotFoundError as error:
        raise DatasetError(ESCO_RESTORE_HELP) from error
    except (OSError, json.JSONDecodeError) as error:
        raise DatasetError(
            f"Could not read {path.name}: {error}\n\n{ESCO_RESTORE_HELP}"
        ) from error


def load_esco_records(path: Path = ESCO_FILE) -> list[dict]:
    """Load and validate ESCO occupations in the normalized schema."""
    raw = _read_json(Path(path))

    if not isinstance(raw, list) or not raw:
        raise DatasetError(
            f"{Path(path).name} must contain a non-empty JSON list.\n\n"
            + ESCO_RESTORE_HELP
        )

    records = []
    seen_ids = set()
    seen_titles = set()
    skipped = 0

    for item in raw:
        if not isinstance(item, dict):
            skipped += 1
            continue

        occupation_id = _clean_text(item.get("id"))
        title = _clean_text(item.get("title"))
        if not occupation_id or not title:
            skipped += 1
            continue
        if occupation_id in seen_ids:
            raise DatasetError(f"Duplicate ESCO occupation id: {occupation_id}")
        if title_key(title) in seen_titles:
            raise DatasetError(f"Duplicate ESCO occupation title: {title}")
        seen_ids.add(occupation_id)
        seen_titles.add(title_key(title))

        essential = _clean_list(item.get("essential_skills"))
        optional = [
            skill
            for skill in _clean_list(item.get("optional_skills"))
            if skill.casefold() not in {s.casefold() for s in essential}
        ]

        # Backwards compatibility with the earlier importer, which stored
        # one undifferentiated "skills" list.
        if not essential and not optional:
            essential = _clean_list(item.get("skills"))

        records.append({
            "id": occupation_id,
            "title": display_title(title),
            "esco_title": title,
            "alt_titles": _clean_list(item.get("alt_titles")),
            "description": _clean_text(item.get("description")),
            "isco_group": _clean_text(item.get("isco_group")),
            "essential_skills": essential,
            "optional_skills": optional,
            "curated_skills": [],
            "interests": [],
            "learning": [],
            "source": SOURCE_ESCO,
            "source_detail": _clean_text(item.get("source")) or "ESCO",
            "esco_uri": occupation_id,
        })

    if skipped and skipped > len(raw) * 0.01:
        raise DatasetError(
            f"{skipped} of {len(raw)} ESCO records are malformed "
            "(missing id/title).\n\n" + ESCO_RESTORE_HELP
        )

    return records


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.casefold()).strip("-")


def merge_catalogues(esco_records: list[dict], curated: list[dict]):
    """Merge curated careers into ESCO records without duplicating titles.

    Returns (records, merge_report).
    """
    by_title = {title_key(record["esco_title"]): record for record in esco_records}
    merged_titles = []
    added_titles = []
    records = [dict(record) for record in esco_records]
    index_by_id = {record["id"]: i for i, record in enumerate(records)}

    for career in curated:
        title = _clean_text(career.get("title"))
        if not title:
            continue

        curated_skills = _clean_list(career.get("skills"))
        interests = _clean_list(career.get("interests"))
        learning = _clean_list(career.get("learning"))
        description = _clean_text(career.get("description"))

        match = by_title.get(title_key(title))
        if match is not None:
            record = records[index_by_id[match["id"]]]
            esco_skill_keys = {
                s.casefold()
                for s in record["essential_skills"] + record["optional_skills"]
            }
            record.update({
                "title": title,
                "curated_skills": [
                    s for s in curated_skills if s.casefold() not in esco_skill_keys
                ],
                "interests": interests,
                "learning": learning,
                "curated_summary": description,
                "description": record["description"] or description,
                "source": SOURCE_MERGED,
            })
            merged_titles.append(title)
            continue

        records.append({
            "id": f"careerai:{_slug(title)}",
            "title": title,
            "esco_title": "",
            "alt_titles": [],
            "description": description,
            "isco_group": "",
            "essential_skills": [],
            "optional_skills": [],
            "curated_skills": curated_skills,
            "interests": interests,
            "learning": learning,
            "source": SOURCE_CURATED,
            "source_detail": "CareerAI curated catalogue (not an ESCO record)",
            "esco_uri": "",
        })
        added_titles.append(title)

    validate_catalogue(records)

    report = {
        "esco_records": len(esco_records),
        "curated_records": len(curated),
        "curated_merged_into_esco": sorted(merged_titles),
        "curated_added_separately": sorted(added_titles),
        "total_records": len(records),
    }
    return records, report


def validate_catalogue(records: list[dict]) -> None:
    """Fail loudly on duplicate ids/titles or records without content."""
    ids = set()
    titles = set()
    for record in records:
        if record["id"] in ids:
            raise DatasetError(f"Duplicate career id: {record['id']}")
        ids.add(record["id"])
        key = title_key(record["title"])
        if key in titles:
            raise DatasetError(f"Duplicate career title: {record['title']}")
        titles.add(key)
        if not (
            record["essential_skills"]
            or record["optional_skills"]
            or record["curated_skills"]
        ):
            raise DatasetError(f"Career has no skills: {record['title']}")


def all_skills(record: dict) -> list[str]:
    """All skills of a career, most important first."""
    return (
        list(record.get("curated_skills", []))
        + list(record.get("essential_skills", []))
        + list(record.get("optional_skills", []))
    )


def load_catalogue(
    esco_path: Path = ESCO_FILE,
    curated_path: Path | None = CAREERS_FILE,
):
    """Load the full application catalogue. Never falls back silently."""
    esco_records = load_esco_records(esco_path)

    curated = []
    if curated_path is not None and Path(curated_path).exists():
        try:
            curated = load_and_prepare_careers(curated_path)
        except (OSError, ValueError) as error:
            raise DatasetError(
                f"Curated catalogue {Path(curated_path).name} is invalid: {error}"
            ) from error

    return merge_catalogues(esco_records, curated)


@lru_cache(maxsize=1)
def _cached_default(esco_mtime: float, curated_mtime: float):
    return load_catalogue()


def load_default_catalogue():
    """Cached loader for the default data files (reloads if they change)."""
    try:
        esco_mtime = ESCO_FILE.stat().st_mtime
    except OSError as error:
        raise DatasetError(ESCO_RESTORE_HELP) from error
    curated_mtime = CAREERS_FILE.stat().st_mtime if CAREERS_FILE.exists() else 0.0
    return _cached_default(esco_mtime, curated_mtime)


def load_import_report(path: Path = ESCO_REPORT_FILE) -> dict:
    try:
        with Path(path).open("r", encoding="utf-8") as file:
            report = json.load(file)
        return report if isinstance(report, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}
