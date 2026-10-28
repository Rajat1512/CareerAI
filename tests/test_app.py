"""Headless smoke tests of the Streamlit app using streamlit.testing.AppTest."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


APP = str(Path(__file__).resolve().parent.parent / "app.py")
PAGES = [
    "Home", "Career Explorer", "Job Opportunities",
    "Learning Path", "My Profile", "Saved Careers",
]


@pytest.fixture
def app(monkeypatch, tmp_path):
    # Never touch the real user_data/ folder during tests.
    monkeypatch.setenv("CAREERAI_STORAGE", "file")
    monkeypatch.setenv("CAREERAI_DATA_DIR", str(tmp_path))
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception
    return at


def all_text(at):
    parts = [m.value for m in at.markdown] + [e.value for e in at.error]
    parts += [i.value for i in at.info] + [w.value for w in at.warning]
    parts += [s.value for s in at.success] + [c.value for c in at.caption]
    parts += [t.value for t in at.title] + [s.value for s in at.subheader]
    return "\n".join(str(p) for p in parts)


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_without_errors(app, page):
    app.sidebar.radio[0].set_value(page).run()

    assert not app.exception
    assert not app.error, [e.value for e in app.error]


def test_main_user_journey(app, tmp_path):
    # Home search -> Career Explorer with results.
    app.text_input(key="home_skill").input("photography")
    app.button(key="home_explore").click().run()
    assert app.sidebar.radio[0].value == "Career Explorer"
    assert "1. Photographer" in all_text(app)
    assert "Why this career?" in all_text(app)

    # Save the first career.
    save_buttons = [b for b in app.button if b.key and b.key.startswith("save_")]
    save_buttons[0].click().run()
    assert not app.exception
    assert (tmp_path / "saved_items.json").exists()
    assert "Saved" in all_text(app)

    # Revisit it on the Saved Careers page.
    app.sidebar.radio[0].set_value("Saved Careers").run()
    assert "Photographer" in all_text(app)

    # Open its learning path.
    learn = [b for b in app.button if b.key and b.key.startswith("saved_learn_")]
    learn[0].click().run()
    assert app.sidebar.radio[0].value == "Learning Path"
    assert "Your Learning Roadmap" in all_text(app)

    # Remove it again.
    app.sidebar.radio[0].set_value("Saved Careers").run()
    remove = [b for b in app.button if b.key and b.key.startswith("remove_")]
    remove[0].click().run()
    assert "No saved careers yet" in all_text(app)


def test_profile_feeds_explorer(app):
    app.sidebar.radio[0].set_value("My Profile").run()
    app.text_input[1].input("sanskrit")
    app.button[0].click().run()  # form submit
    assert "saved successfully" in all_text(app)

    recommend = [b for b in app.button if b.label == "Recommend careers from my profile"]
    recommend[0].click().run()
    assert app.sidebar.radio[0].value == "Career Explorer"
    assert "1. Sanskrit Teacher" in all_text(app)


def test_empty_and_unknown_input_are_handled(app):
    app.sidebar.radio[0].set_value("Career Explorer").run()
    app.button(key="explore_careers_button").click().run()
    assert "Please enter at least one skill or interest" in all_text(app)

    app.text_input(key="_w_explorer_skills").input("qwxzvbnm")
    app.button(key="explore_careers_button").click().run()
    assert not app.exception
    assert "were not found in any career" in all_text(app)
    assert "No relevant careers found" in all_text(app)


def test_jobs_page_has_no_fake_listings(app):
    app.sidebar.radio[0].set_value("Job Opportunities").run()
    text = all_text(app)
    assert "does not have a job-vacancy data source" in text
    assert "Demo Listing" not in text
