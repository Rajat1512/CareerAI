"""Import the ESCO occupation/skill CSV files into data/careers_esco.json.

Run from the project root:

    python -m src.import_esco

The raw ESCO v1.2.1 English CSV files must be in data/esco_raw/ (or pass
--input-dir). The output catalogue is written once and then loaded by the app;
the app never downloads or rebuilds the dataset at start-up.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "esco_raw"
OUTPUT_CATALOGUE = PROJECT_ROOT / "data" / "careers_esco.json"
OUTPUT_REPORT = PROJECT_ROOT / "data" / "esco_import_report.json"
DEFAULT_VERSION = "v1.2.1"


def read_csv_safely(path: Path) -> pd.DataFrame:
    """Read a CSV, trying the standard ESCO format first, then sniffing."""
    errors = []

    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        for options in ({"sep": ","}, {"sep": None, "engine": "python"}):
            try:
                df = pd.read_csv(
                    path,
                    encoding=encoding,
                    dtype=str,
                    keep_default_na=False,
                    **options,
                )
                if len(df.columns) > 1:
                    return df
            except Exception as exc:
                errors.append(f"{encoding}/{options}: {exc}")

    raise ValueError(
        f"Could not read {path.name} as a delimited file. "
        + " | ".join(errors)
    )


def find_csv(input_dir: Path, keyword: str) -> Path:
    """Find one CSV whose filename contains the requested keyword."""
    matches = [
        path
        for path in input_dir.rglob("*")
        if path.is_file()
        and keyword.lower() in path.stem.lower()
        and path.suffix.lower() in {".csv", ".txt"}
    ]

    if not matches:
        raise FileNotFoundError(
            f"No .csv or .txt file containing '{keyword}' was found "
            f"inside: {input_dir}\n"
            "Check that the ESCO files have been extracted and have "
            "their real file extensions."
        )

    # Prefer the exact English ESCO filename when there are several matches.
    matches.sort(
        key=lambda path: (
            0 if path.stem.lower().endswith("_en") else 1,
            len(path.name),
        )
    )
    return matches[0]


def _normalize_column(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def optional_column(
    df: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    normalized = {_normalize_column(str(col)): str(col) for col in df.columns}
    for candidate in candidates:
        if _normalize_column(candidate) in normalized:
            return normalized[_normalize_column(candidate)]
    return None


def find_column(
    df: pd.DataFrame,
    candidates: list[str],
    description: str,
) -> str:
    """Match a column by normalized name, including common URI variants."""
    column = optional_column(df, candidates)
    if column is None:
        raise ValueError(
            f"Could not identify the {description} column.\n"
            f"Available columns: {list(df.columns)}"
        )
    return column


def load_source_files(input_dir: Path):
    occupation_path = find_csv(input_dir, "occupations_en")
    skill_path = find_csv(input_dir, "skills_en")
    relation_path = find_csv(input_dir, "occupationSkillRelations_en")

    return (
        occupation_path,
        skill_path,
        relation_path,
        read_csv_safely(occupation_path),
        read_csv_safely(skill_path),
        read_csv_safely(relation_path),
    )


def deduplicate_concepts(df: pd.DataFrame, uri_column: str):
    """Keep one row per concept URI (the most recently modified one).

    The ESCO v1.2.1 CSV export repeats a few concepts with identical content
    and different modifiedDate values. Without this step the joins below fan
    out and inflate the relationship counts.
    """
    df = df.copy()
    df[uri_column] = df[uri_column].str.strip()
    before = len(df)
    date_column = optional_column(df, ["modifiedDate"])

    if date_column:
        df = df.sort_values(date_column, kind="stable")

    df = df.drop_duplicates(uri_column, keep="last")
    return df, before - len(df)


def split_labels(value) -> list[str]:
    """Split an ESCO multi-line label field into unique, non-empty labels."""
    labels = []
    seen = set()
    for label in str(value or "").splitlines():
        label = " ".join(label.split())
        if label and label.casefold() not in seen:
            labels.append(label)
            seen.add(label.casefold())
    return labels


def build_catalogue(
    occupations: pd.DataFrame,
    skills: pd.DataFrame,
    relations: pd.DataFrame,
    version: str = DEFAULT_VERSION,
):
    occ_id = find_column(
        occupations,
        ["conceptUri", "occupationUri", "uri", "id"],
        "occupation identifier",
    )
    occ_label = find_column(
        occupations,
        ["preferredLabel", "occupationLabel", "label", "title"],
        "occupation preferred label",
    )
    occ_description = optional_column(
        occupations,
        ["description", "definition", "scopeNote"],
    )
    occ_alt = optional_column(occupations, ["altLabels"])
    occ_isco = optional_column(occupations, ["iscoGroup"])
    occ_code = optional_column(occupations, ["code"])

    skill_id = find_column(
        skills,
        ["conceptUri", "skillUri", "uri", "id"],
        "skill identifier",
    )
    skill_label = find_column(
        skills,
        ["preferredLabel", "skillLabel", "label", "title"],
        "skill preferred label",
    )

    relation_occ_id = find_column(
        relations,
        ["occupationUri", "occupationURI", "occupation", "occupationId"],
        "occupation identifier in relationship file",
    )
    relation_skill_id = find_column(
        relations,
        ["skillUri", "skillURI", "skill", "skillId"],
        "skill identifier in relationship file",
    )
    relation_type = optional_column(relations, ["relationType"])

    raw_occupation_rows = len(occupations)
    raw_skill_rows = len(skills)
    relation_input_count = len(relations)

    occupations, duplicate_occupation_rows = deduplicate_concepts(
        occupations, occ_id
    )
    skills, duplicate_skill_rows = deduplicate_concepts(skills, skill_id)

    occ_table = pd.DataFrame({
        "occupation_uri": occupations[occ_id],
        "occupation_label": occupations[occ_label].str.strip(),
        "occupation_description": (
            occupations[occ_description] if occ_description else ""
        ),
        "alt_labels": occupations[occ_alt] if occ_alt else "",
        "isco_group": occupations[occ_isco] if occ_isco else "",
        "esco_code": occupations[occ_code] if occ_code else "",
    })
    skill_table = pd.DataFrame({
        "skill_uri": skills[skill_id],
        "skill_label": skills[skill_label].str.strip(),
    })

    rel = pd.DataFrame({
        "occupation_uri": relations[relation_occ_id].str.strip(),
        "skill_uri": relations[relation_skill_id].str.strip(),
        "relation_type": (
            relations[relation_type].str.strip().str.lower()
            if relation_type
            else "essential"
        ),
    })

    empty_id_rows = int(
        ((rel["occupation_uri"] == "") | (rel["skill_uri"] == "")).sum()
    )
    rel = rel[(rel["occupation_uri"] != "") & (rel["skill_uri"] != "")]

    # If the same pair is listed twice, keep the stronger (essential) link.
    rel = rel.assign(
        _rank=(rel["relation_type"] != "essential").astype(int)
    ).sort_values("_rank", kind="stable")
    duplicate_rows = int(rel.duplicated(["occupation_uri", "skill_uri"]).sum())
    rel = rel.drop_duplicates(["occupation_uri", "skill_uri"]).drop(
        columns="_rank"
    )

    # Join using canonical identifiers, not labels.
    joined = rel.merge(
        occ_table, on="occupation_uri", how="left", indicator="_occ_join"
    )
    joined = joined.merge(
        skill_table, on="skill_uri", how="left", indicator="_skill_join"
    )

    unresolved_occupation_rows = int((joined["_occ_join"] != "both").sum())
    unresolved_skill_rows = int((joined["_skill_join"] != "both").sum())

    valid = joined[
        (joined["_occ_join"] == "both")
        & (joined["_skill_join"] == "both")
        & (joined["skill_label"].str.strip() != "")
    ].copy()

    if len(valid) != len(valid.drop_duplicates(["occupation_uri", "skill_uri"])):
        raise ValueError("Join produced duplicate occupation-skill rows.")

    catalogue = []
    occupations_without_skills = 0
    grouped = {uri: g for uri, g in valid.groupby("occupation_uri", sort=False)}

    for occupation in occ_table.sort_values(
        "occupation_label", key=lambda s: s.str.casefold(), kind="stable"
    ).itertuples(index=False):
        group = grouped.get(occupation.occupation_uri)
        if group is None:
            occupations_without_skills += 1
            continue

        def labels_for(kind):
            return sorted(
                set(group.loc[group["relation_type"] == kind, "skill_label"]),
                key=str.casefold,
            )

        essential = labels_for("essential")
        optional = [
            label
            for label in sorted(
                set(group.loc[group["relation_type"] != "essential", "skill_label"]),
                key=str.casefold,
            )
            if label not in set(essential)
        ]

        title = occupation.occupation_label
        alt_titles = [
            label
            for label in split_labels(occupation.alt_labels)
            if label.casefold() != title.casefold()
        ]

        catalogue.append({
            "id": occupation.occupation_uri,
            "title": title,
            "alt_titles": alt_titles,
            "description": " ".join(
                str(occupation.occupation_description).split()
            ),
            "isco_group": str(occupation.isco_group).strip(),
            "esco_code": str(occupation.esco_code).strip(),
            "essential_skills": essential,
            "optional_skills": optional,
            "source": f"ESCO {version}",
        })

    report = {
        "source": "ESCO",
        "version": version,
        "language": "en",
        "imported_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_files": {
            "occupation_columns": list(occupations.columns),
            "skill_columns": list(skills.columns),
            "relationship_columns": list(relations.columns),
        },
        "raw_counts": {
            "occupation_rows": int(raw_occupation_rows),
            "skill_rows": int(raw_skill_rows),
            "relationship_rows": int(relation_input_count),
        },
        "cleaning": {
            "duplicate_occupation_uri_rows_removed": int(duplicate_occupation_rows),
            "duplicate_skill_uri_rows_removed": int(duplicate_skill_rows),
            "relationship_rows_with_empty_ids": empty_id_rows,
            "duplicate_relationship_rows_removed": duplicate_rows,
            "unresolved_occupation_relationship_rows": unresolved_occupation_rows,
            "unresolved_skill_relationship_rows": unresolved_skill_rows,
            "occupations_without_linked_skills": occupations_without_skills,
        },
        "final_counts": {
            "valid_relationship_rows": int(len(valid)),
            "essential_relationship_rows": int(
                (valid["relation_type"] == "essential").sum()
            ),
            "optional_relationship_rows": int(
                (valid["relation_type"] != "essential").sum()
            ),
            "unique_occupations": int(len(catalogue)),
            "unique_skills_linked": int(valid["skill_uri"].nunique()),
            "occupation_catalogue_records": int(len(catalogue)),
        },
    }

    return catalogue, report


def main():
    parser = argparse.ArgumentParser(
        description="Import ESCO occupation and skill data for CareerAI."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Folder containing the three ESCO English CSV files.",
    )
    parser.add_argument(
        "--version",
        default=DEFAULT_VERSION,
        help="ESCO version label stored with every record (default: v1.2.1).",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    if not input_dir.exists():
        raise SystemExit(
            f"Input directory does not exist: {input_dir}\n"
            "Download the ESCO classification (CSV, English) from "
            "https://esco.ec.europa.eu/en/use-esco/download and extract "
            "occupations_en.csv, skills_en.csv and "
            "occupationSkillRelations_en.csv into data/esco_raw/."
        )

    print(f"Reading ESCO files from: {input_dir}")

    (
        occupation_path,
        skill_path,
        relation_path,
        occupations,
        skills,
        relations,
    ) = load_source_files(input_dir)

    print(f"Occupations file: {occupation_path.name}")
    print(f"Skills file: {skill_path.name}")
    print(f"Relationships file: {relation_path.name}")

    catalogue, report = build_catalogue(
        occupations, skills, relations, version=args.version
    )
    report["files_used"] = {
        "occupations": occupation_path.name,
        "skills": skill_path.name,
        "relationships": relation_path.name,
    }

    OUTPUT_CATALOGUE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_CATALOGUE.open("w", encoding="utf-8") as file:
        # One record per line keeps the file compact but still diff-friendly.
        file.write("[\n")
        for index, record in enumerate(catalogue):
            file.write(json.dumps(record, ensure_ascii=False))
            file.write(",\n" if index < len(catalogue) - 1 else "\n")
        file.write("]\n")

    with OUTPUT_REPORT.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print("\nImport completed.")
    print(json.dumps(report["cleaning"], indent=2))
    print(json.dumps(report["final_counts"], indent=2))
    print(f"\nCatalogue saved to: {OUTPUT_CATALOGUE}")
    print(f"Validation report saved to: {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
