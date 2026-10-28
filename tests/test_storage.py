import json

import pytest

from src.storage import FileStore, MemoryStore


CAREER = {"id": "x1", "title": "Photographer", "description": "Takes photos.",
          "skills": ["photography"], "source": "ESCO"}


def test_save_list_and_remove_career(tmp_path):
    store = FileStore(tmp_path)

    assert store.save_career(CAREER) is True
    assert store.save_career(CAREER) is False  # no duplicates
    assert store.is_saved("x1")
    assert [c["title"] for c in FileStore(tmp_path).load_saved_careers()] == ["Photographer"]

    store.remove_career("x1")
    assert store.load_saved_careers() == []


def test_profile_round_trip_and_cleaning(tmp_path):
    store = FileStore(tmp_path)
    store.save_profile({"name": "  A  ", "skills": "python", "extra": "ignored"})

    assert store.load_profile() == {
        "name": "A", "skills": "python", "interests": "", "education": "",
    }


def test_missing_files_give_empty_defaults(tmp_path):
    store = FileStore(tmp_path / "does-not-exist")

    assert store.load_saved_careers() == []
    assert store.load_progress() == {}
    assert store.load_profile()["skills"] == ""


def test_corrupt_file_is_set_aside_not_crashing(tmp_path):
    (tmp_path / "saved_items.json").write_text("{broken", encoding="utf-8")
    store = FileStore(tmp_path)

    assert store.load_saved_careers() == []
    assert (tmp_path / "saved_items.corrupt.json").exists()
    store.save_career(CAREER)
    assert len(store.load_saved_careers()) == 1


def test_legacy_saved_items_without_ids_are_readable(tmp_path):
    legacy = {"careers": [{"title": "Software Developer", "skills": ["python"]}], "jobs": []}
    (tmp_path / "saved_items.json").write_text(json.dumps(legacy), encoding="utf-8")

    careers = FileStore(tmp_path).load_saved_careers()

    assert careers[0]["title"] == "Software Developer"
    assert careers[0]["id"] == "title:software developer"


def test_career_without_title_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        FileStore(tmp_path).save_career({"id": "y"})


def test_memory_store_keeps_sessions_apart():
    first, second = MemoryStore({}), MemoryStore({})
    first.save_career(CAREER)

    assert first.is_saved("x1")
    assert not second.is_saved("x1")
