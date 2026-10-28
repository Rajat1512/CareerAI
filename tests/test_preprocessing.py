
from src.preprocessing import (
    normalize_text,
    normalize_list,
    load_and_prepare_careers,
)


def test_normalize_text():
    assert normalize_text("  PYTHON   Developer  ") == "python developer"
    assert normalize_text(None) == ""


def test_normalize_list():
    result = normalize_list([" Python ", "python", "", "JAVA", "Java", None])
    assert result == ["python", "java"]


def test_load_and_prepare_careers():
    careers = load_and_prepare_careers()

    assert len(careers) >= 45
    assert careers[0]["title"] == "Software Developer"
    assert all(career["search_text"] for career in careers)


def test_required_fields_are_preserved():
    careers = load_and_prepare_careers()

    required_fields = {"title", "description", "skills", "interests", "learning"}

    for career in careers:
        assert required_fields.issubset(career.keys())
