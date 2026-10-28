import json

import pandas as pd
import pytest

from src.catalogue import (
    SOURCE_CURATED,
    SOURCE_ESCO,
    SOURCE_MERGED,
    DatasetError,
    load_catalogue,
    load_default_catalogue,
    load_esco_records,
    load_import_report,
)
from src.import_esco import build_catalogue


def test_full_esco_catalogue_loads():
    records = load_esco_records()
    report = load_import_report()

    assert len(records) == 3039
    assert len(records) == report["final_counts"]["unique_occupations"]
    assert len({record["id"] for record in records}) == len(records)
    assert all(record["id"].startswith("http://data.europa.eu/esco/occupation/")
               for record in records)


def test_esco_relationship_count_matches_import_report():
    records = load_esco_records()
    report = load_import_report()
    relations = sum(
        len(r["essential_skills"]) + len(r["optional_skills"]) for r in records
    )

    assert relations == report["final_counts"]["valid_relationship_rows"]
    assert relations == report["raw_counts"]["relationship_rows"]


def test_curated_careers_are_merged_without_duplicates():
    careers, report = load_default_catalogue()
    titles = [career["title"].casefold() for career in careers]

    assert len(titles) == len(set(titles))
    assert report["esco_records"] == 3039
    assert report["curated_records"] == 49
    assert len(report["curated_merged_into_esco"]) + len(
        report["curated_added_separately"]
    ) == 49
    assert report["total_records"] == 3039 + len(report["curated_added_separately"])


def test_merged_and_curated_records_keep_their_source():
    careers, _ = load_default_catalogue()
    by_title = {career["title"]: career for career in careers}

    photographer = by_title["Photographer"]
    assert photographer["source"] == SOURCE_MERGED
    assert photographer["esco_uri"].startswith("http://data.europa.eu/esco/")
    assert "photography" in photographer["curated_skills"]
    assert photographer["essential_skills"]

    sanskrit = by_title["Sanskrit Teacher"]
    assert sanskrit["source"] == SOURCE_CURATED
    assert sanskrit["esco_uri"] == ""
    assert sanskrit["id"].startswith("careerai:")

    assert by_title["Photojournalist"]["source"] == SOURCE_ESCO


def test_missing_esco_file_raises_clear_error(tmp_path):
    with pytest.raises(DatasetError) as error:
        load_catalogue(esco_path=tmp_path / "missing.json")

    assert "python -m src.import_esco" in str(error.value)


def test_malformed_esco_file_raises_clear_error(tmp_path):
    path = tmp_path / "careers_esco.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(DatasetError):
        load_esco_records(path)


def test_empty_esco_file_is_rejected(tmp_path):
    path = tmp_path / "careers_esco.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(DatasetError):
        load_esco_records(path)


def test_duplicate_esco_ids_are_rejected(tmp_path):
    record = {"id": "x", "title": "a", "essential_skills": ["s"]}
    path = tmp_path / "careers_esco.json"
    path.write_text(json.dumps([record, {**record, "title": "b"}]), encoding="utf-8")

    with pytest.raises(DatasetError):
        load_esco_records(path)


def test_importer_removes_duplicate_uris_before_joining():
    occupations = pd.DataFrame({
        "conceptUri": ["o1", "o1", "o2"],
        "preferredLabel": ["baker", "baker", "cook"],
        "altLabels": ["pastry baker", "pastry baker", ""],
        "description": ["Bakes bread.", "Bakes bread.", "Cooks food."],
        "modifiedDate": ["2025-01-01", "2025-02-01", "2025-01-01"],
    })
    skills = pd.DataFrame({
        "conceptUri": ["s1", "s1", "s2"],
        "preferredLabel": ["bake bread", "bake bread", "cooking"],
        "modifiedDate": ["2025-01-01", "2025-02-01", "2025-01-01"],
    })
    relations = pd.DataFrame({
        "occupationUri": ["o1", "o1", "o2"],
        "skillUri": ["s1", "s2", "s2"],
        "relationType": ["essential", "optional", "essential"],
    })

    catalogue, report = build_catalogue(occupations, skills, relations)

    assert report["final_counts"]["valid_relationship_rows"] == 3
    assert report["final_counts"]["unique_occupations"] == 2
    assert report["cleaning"]["duplicate_occupation_uri_rows_removed"] == 1
    assert report["cleaning"]["duplicate_skill_uri_rows_removed"] == 1
    baker = next(item for item in catalogue if item["title"] == "baker")
    assert baker["essential_skills"] == ["bake bread"]
    assert baker["optional_skills"] == ["cooking"]
    assert baker["alt_titles"] == ["pastry baker"]
