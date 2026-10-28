"""Storage for user data: profile, saved careers and learning progress.

Two back-ends with the same interface:

* FileStore   - JSON files in a local folder (default: user_data/). Good for
                running CareerAI on your own computer.
* MemoryStore - keeps data in memory for one browser session only. Use it for
                a public deployment so visitors never see each other's data.

Choose with the CAREERAI_STORAGE environment variable ("file" or "session").
"""

from __future__ import annotations

import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_USER_DATA_DIR = PROJECT_ROOT / "user_data"

EMPTY_PROFILE = {"name": "", "skills": "", "interests": "", "education": ""}
PROFILE_FIELDS = tuple(EMPTY_PROFILE)
MAX_FIELD_LENGTH = 1000
MAX_SAVED_CAREERS = 200

SAVED_CAREER_FIELDS = (
    "id", "title", "description", "source", "esco_uri", "skills",
    "interests", "learning",
)


def clean_profile(profile) -> dict:
    if not isinstance(profile, dict):
        return dict(EMPTY_PROFILE)
    return {
        field: str(profile.get(field, "") or "").strip()[:MAX_FIELD_LENGTH]
        for field in PROFILE_FIELDS
    }


def clean_saved_career(career) -> dict | None:
    if not isinstance(career, dict) or not str(career.get("title", "")).strip():
        return None
    item = {field: deepcopy(career.get(field)) for field in SAVED_CAREER_FIELDS}
    item["title"] = str(item["title"]).strip()
    item["id"] = str(item.get("id") or f"title:{item['title'].casefold()}")
    item["description"] = str(item.get("description") or "")
    item["source"] = str(item.get("source") or "")
    item["esco_uri"] = str(item.get("esco_uri") or "")
    for field in ("skills", "interests", "learning"):
        value = item.get(field)
        item[field] = [str(v) for v in value] if isinstance(value, list) else []
    return item


def clean_progress(progress) -> dict:
    if not isinstance(progress, dict):
        return {}
    return {
        str(key): sorted({str(v) for v in value})
        for key, value in progress.items()
        if isinstance(value, list)
    }


class MemoryStore:
    """Per-session storage; `state` is any dict (e.g. st.session_state)."""

    def __init__(self, state: dict | None = None):
        self.state = state if state is not None else {}
        self.state.setdefault("_store_profile", dict(EMPTY_PROFILE))
        self.state.setdefault("_store_saved", [])
        self.state.setdefault("_store_progress", {})

    persistent = False

    def load_profile(self):
        return dict(self.state["_store_profile"])

    def save_profile(self, profile):
        self.state["_store_profile"] = clean_profile(profile)

    def load_saved_careers(self):
        return deepcopy(self.state["_store_saved"])

    def _write_saved(self, careers):
        self.state["_store_saved"] = careers

    def load_progress(self):
        return deepcopy(self.state["_store_progress"])

    def save_progress(self, progress):
        self.state["_store_progress"] = clean_progress(progress)

    # Shared helpers -------------------------------------------------------

    def save_career(self, career) -> bool:
        """Add a career; returns False if it was already saved."""
        item = clean_saved_career(career)
        if item is None:
            raise ValueError("Cannot save a career without a title.")
        careers = self.load_saved_careers()
        if any(saved["id"] == item["id"] for saved in careers):
            return False
        if len(careers) >= MAX_SAVED_CAREERS:
            raise ValueError(f"You can save up to {MAX_SAVED_CAREERS} careers.")
        careers.append(item)
        self._write_saved(careers)
        return True

    def remove_career(self, career_id) -> None:
        careers = [c for c in self.load_saved_careers() if c["id"] != career_id]
        self._write_saved(careers)

    def is_saved(self, career_id) -> bool:
        return any(c["id"] == career_id for c in self.load_saved_careers())


class FileStore(MemoryStore):
    """JSON files on disk, written atomically."""

    persistent = True

    def __init__(self, directory: Path | str = DEFAULT_USER_DATA_DIR):
        self.directory = Path(directory)
        self.profile_path = self.directory / "profile.json"
        self.saved_path = self.directory / "saved_items.json"
        self.progress_path = self.directory / "learning_progress.json"

    def _read(self, path: Path, default):
        try:
            with path.open("r", encoding="utf-8-sig") as file:
                return json.load(file)
        except FileNotFoundError:
            return default
        except (OSError, json.JSONDecodeError):
            # Keep the unreadable file for inspection instead of overwriting.
            try:
                path.replace(path.with_suffix(".corrupt.json"))
            except OSError:
                pass
            return default

    def _write(self, path: Path, data) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=self.directory, prefix=f".{path.stem}-", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                json.dump(data, file, indent=2, ensure_ascii=False)
            os.replace(tmp_name, path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise

    def load_profile(self):
        return clean_profile(self._read(self.profile_path, EMPTY_PROFILE))

    def save_profile(self, profile):
        self._write(self.profile_path, clean_profile(profile))

    def load_saved_careers(self):
        data = self._read(self.saved_path, {"careers": []})
        careers = data.get("careers", []) if isinstance(data, dict) else []
        cleaned = []
        seen = set()
        for career in careers:
            item = clean_saved_career(career)
            if item and item["id"] not in seen:
                cleaned.append(item)
                seen.add(item["id"])
        return cleaned

    def _write_saved(self, careers):
        self._write(self.saved_path, {"careers": careers})

    def load_progress(self):
        return clean_progress(self._read(self.progress_path, {}))

    def save_progress(self, progress):
        self._write(self.progress_path, clean_progress(progress))


def storage_mode() -> str:
    mode = os.environ.get("CAREERAI_STORAGE", "file").strip().lower()
    return mode if mode in {"file", "session"} else "file"


def user_data_dir() -> Path:
    configured = os.environ.get("CAREERAI_DATA_DIR", "").strip()
    if not configured:
        return DEFAULT_USER_DATA_DIR
    path = Path(configured).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path
