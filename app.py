"""CareerAI - Streamlit application.

Run with:  streamlit run app.py
"""

import logging
from urllib.parse import quote_plus

import streamlit as st

from src.catalogue import DatasetError, load_import_report
from src.content_recommender import ContentRecommender
from src.storage import FileStore, MemoryStore, storage_mode, user_data_dir


logger = logging.getLogger("careerai")

PAGES = [
    "Home",
    "Career Explorer",
    "Job Opportunities",
    "Learning Path",
    "My Profile",
    "Saved Careers",
]

# A few hand-picked starting points from the original project. Every other
# skill gets a course search link instead of an invented resource.
LEARNING_RESOURCES = {
    "python": "https://docs.python.org/3/tutorial/",
    "java": "https://dev.java/learn/",
    "data structures": "https://www.geeksforgeeks.org/data-structures/",
    "algorithms": "https://www.geeksforgeeks.org/fundamentals-of-algorithms/",
    "git": "https://git-scm.com/doc",
    "sql": "https://www.w3schools.com/sql/",
    "excel": "https://support.microsoft.com/excel",
    "statistics": "https://www.khanacademy.org/math/statistics-probability",
    "linear algebra": "https://www.khanacademy.org/math/linear-algebra",
    "machine learning": "https://scikit-learn.org/stable/user_guide.html",
}


# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="CareerAI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------
# CUSTOM CSS
# --------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root { --ink:#0F1E3D; --muted:#5B6B86; --blue:#2563EB; --blue-dark:#1D4ED8; --sky:#E8F0FE; --line:#E2E8F0; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:#F7F9FC; color:var(--ink); overflow-x:hidden; }
.block-container { padding-top:1.8rem; padding-bottom:3rem; max-width:1320px; }
h1,h2,h3 { font-family:'Manrope','DM Sans',sans-serif !important; letter-spacing:-.03em; color:var(--ink); }
h1 { font-weight:800 !important; font-size:clamp(1.8rem,3vw,2.4rem) !important; }
h2 { font-size:clamp(1.4rem,2.5vw,1.8rem) !important; }
p, label, .stMarkdown { color:var(--ink); }
[data-testid="stSidebar"] { background:#FFFFFF; border-right:1px solid var(--line); }
[data-testid="stSidebar"] [data-testid="stRadio"] label { padding:.45rem .55rem; border-radius:9px; }
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background:var(--sky); }
.hero-label { color:var(--blue); font-size:12px; font-weight:800; letter-spacing:2px; margin-bottom:12px; }
.hero-title { color:var(--ink); font-size:clamp(30px,4vw,44px); font-weight:800; line-height:1.12; margin-bottom:12px; }
.hero-description { color:var(--muted); font-size:16px; line-height:1.7; max-width:720px; }
.section-heading { color:var(--ink); font-size:24px; font-weight:800; letter-spacing:-.025em; margin-top:24px; margin-bottom:12px; }
div.stButton > button, div[data-testid="stFormSubmitButton"] button { background:var(--blue); color:#fff !important; border:0; border-radius:10px; min-height:42px; font-weight:700; box-shadow:0 4px 12px #2563eb26; transition:transform .15s ease, background .15s ease; max-width:100%; white-space:normal; }
div.stButton > button p, div[data-testid="stFormSubmitButton"] button p { color:#fff !important; font-weight:700; }
div.stButton > button:hover, div[data-testid="stFormSubmitButton"] button:hover { background:var(--blue-dark); color:#fff !important; transform:translateY(-1px); }
[data-testid="stVerticalBlockBorderWrapper"] { background:#fff; border:1px solid var(--line); border-radius:16px; box-shadow:0 4px 18px #0f1e3d08; min-width:0; max-width:100%; }
[data-testid="stVerticalBlockBorderWrapper"]:hover { border-color:#BFD3FB; }
.stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div { background:#fff !important; color:var(--ink) !important; border:1px solid #D5DEEB !important; border-radius:10px !important; }
.stTextInput input:focus, .stTextArea textarea:focus { border-color:var(--blue) !important; box-shadow:0 0 0 2px #2563eb22 !important; }
.stTextInput input::placeholder { color:#8794AA !important; opacity:1 !important; }
.stTextInput label,.stSelectbox label,.stTextArea label { color:var(--ink) !important; font-weight:700 !important; }
[data-testid="stAlert"] { border-radius:12px; }
[data-testid="stMetric"] { background:#fff; border:1px solid var(--line); padding:16px; border-radius:14px; }
div[data-testid="stMarkdownContainer"] { overflow-wrap:anywhere; word-break:normal; }
@media(max-width:700px) { .block-container{padding:1rem 1rem 2rem;} }
</style>
""", unsafe_allow_html=True)


# --------------------------------
# SHARED RESOURCES AND STATE
# --------------------------------

@st.cache_resource(show_spinner="Loading the ESCO career catalogue (first start only)...")
def get_recommender():
    return ContentRecommender()


@st.cache_data
def get_import_report():
    return load_import_report()


def get_store():
    if storage_mode() == "session":
        return MemoryStore(st.session_state)
    return FileStore(user_data_dir())


def go_to(page):
    st.session_state["nav"] = page


def split_terms(text):
    return [term.strip() for term in str(text or "").split(",") if term.strip()]


def persistent_text_input(label, key, **kwargs):
    """Text input whose value survives switching pages."""
    widget_key = f"_w_{key}"
    if widget_key not in st.session_state:
        st.session_state[widget_key] = st.session_state.get(key, "")
    value = st.text_input(label, key=widget_key, **kwargs)
    st.session_state[key] = value
    return value


def set_explorer_inputs(skills, interests, autorun=True, navigate=True):
    """Callback: fill the Career Explorer and optionally run it."""
    for key, value in (("explorer_skills", skills), ("explorer_interests", interests)):
        st.session_state[key] = value
        st.session_state[f"_w_{key}"] = value
    if autorun:
        st.session_state["explorer_autorun"] = True
    if navigate:
        go_to("Career Explorer")


def home_search_callback():
    text = st.session_state.get("home_skill", "").strip()
    if text:
        set_explorer_inputs(text, "")
    else:
        st.session_state["home_empty_warning"] = True


def save_career_callback(career):
    try:
        added = get_store().save_career(career)
        st.session_state["flash"] = (
            f"Saved “{career['title']}”." if added
            else f"“{career['title']}” is already saved."
        )
    except (OSError, ValueError) as error:
        logger.exception("Saving a career failed")
        st.session_state["flash_error"] = f"Could not save the career: {error}"


def remove_career_callback(career_id, title):
    try:
        get_store().remove_career(career_id)
        st.session_state["flash"] = f"Removed “{title}”."
    except OSError:
        logger.exception("Removing a career failed")
        st.session_state["flash_error"] = "Could not remove the career. Please try again."


def show_flash_messages():
    if "flash" in st.session_state:
        st.success(st.session_state.pop("flash"))
    if "flash_error" in st.session_state:
        st.error(st.session_state.pop("flash_error"))


def source_caption(career):
    source = career.get("source", "")
    if source == "ESCO":
        return "Source: ESCO v1.2.1 occupation (European Commission)"
    if source == "ESCO + CareerAI curated":
        return "Source: ESCO v1.2.1 occupation, with extra skills from the CareerAI curated list"
    if source == "CareerAI curated":
        return "Source: CareerAI curated career (not an official ESCO occupation)"
    return f"Source: {source}" if source else ""


def resolve_saved(recommender, saved):
    """Find the catalogue record for a saved career (older saves have no id)."""
    career = recommender.get_career(saved["id"])
    if career is None:
        matches = [c for c in recommender.search_titles(saved["title"], limit=10)
                   if c["title"].casefold() == saved["title"].casefold()]
        career = matches[0] if matches else None
    return career


def candidate_careers(recommender, store):
    """Saved careers first, then the latest recommendations, without duplicates."""
    options = {}
    for saved in store.load_saved_careers():
        career = resolve_saved(recommender, saved)
        if career is not None:
            options[career["id"]] = career
    for result in st.session_state.get("career_results", []):
        career = recommender.get_career(result["id"])
        if career is not None:
            options.setdefault(career["id"], career)
    return list(options.values())


def choose_career(recommender, store, key):
    """Let the user pick a career from saved/recent ones or by searching."""
    careers = candidate_careers(recommender, store)
    query = st.text_input(
        "Search all careers by title",
        placeholder="e.g. data scientist, nurse, photographer",
        key=f"{key}_search",
    )
    if query.strip():
        found = recommender.search_titles(query, limit=25)
        if not found:
            st.info("No career titles match that search. Try a shorter or different word.")
        careers = found + [c for c in careers if c not in found]

    if not careers:
        st.info(
            "Search for a career above, or get recommendations in the "
            "Career Explorer and save the ones you like."
        )
        return None

    preselect = st.session_state.get(f"{key}_target")
    ids = [c["id"] for c in careers]
    index = ids.index(preselect) if preselect in ids else 0
    choice = st.selectbox(
        "Choose a career",
        careers,
        index=index,
        format_func=lambda c: c["title"],
        key=f"{key}_select",
    )
    st.session_state[f"{key}_target"] = choice["id"]
    return choice


# --------------------------------
# SIDEBAR
# --------------------------------

st.session_state.setdefault("nav", "Home")

with st.sidebar:
    st.title("🎓 CareerAI")
    st.caption("Explore • Learn • Grow")
    st.divider()
    page = st.radio("Navigation", PAGES, key="nav")
    st.divider()
    report = get_import_report()
    counts = report.get("final_counts", {})
    if counts:
        st.caption(
            f"Career data: ESCO {report.get('version', '')} — "
            f"{counts.get('unique_occupations', 0):,} occupations, "
            f"{counts.get('unique_skills_linked', 0):,} skills — "
            "plus the CareerAI curated list."
        )
    if storage_mode() == "session":
        st.caption("Your profile and saved careers are kept only for this browser session.")
    st.caption("Your skills can take you anywhere.")


# Load the catalogue once (cached). Stop with a clear message if it is missing.
try:
    recommender = get_recommender()
except DatasetError as error:
    st.error("The career dataset could not be loaded.")
    st.code(str(error), language="text")
    st.stop()
except Exception:
    logger.exception("Unexpected error while loading the recommender")
    st.error(
        "CareerAI could not start because of an unexpected error. "
        "Check the terminal/log output for details."
    )
    st.stop()

store = get_store()


# --------------------------------
# HOME PAGE
# --------------------------------

def page_home():
    with st.container(border=True):
        st.markdown('<div class="hero-label">CAREER DISCOVERY, REIMAGINED</div>', unsafe_allow_html=True)
        st.markdown('<div class="hero-title">Turn your curiosity into a career.</div>', unsafe_allow_html=True)
        st.markdown('<div class="hero-description">Explore career paths that connect your skills, interests, and ambitions. Get practical next steps to move from where you are to where you want to be.</div>', unsafe_allow_html=True)

    st.markdown('<p class="section-heading">What skills or interests do you have?</p>', unsafe_allow_html=True)

    skill = st.text_input(
        "Enter your skills",
        placeholder="e.g. Sanskrit, Python, Photography, Cooking (separate with commas)",
        label_visibility="collapsed",
        key="home_skill",
    )
    st.button("Explore Careers", width="stretch", key="home_explore",
              on_click=home_search_callback)
    if st.session_state.pop("home_empty_warning", False):
        st.warning("Please enter a skill or interest first.")

    st.write("")
    st.markdown('<p class="section-heading">How It Works</p>', unsafe_allow_html=True)

    steps = [
        ("01", "👤", "Build Profile", "Add your skills, interests and education."),
        ("02", "🎯", "Find Careers", "Discover careers matched to your profile."),
        ("03", "💼", "Explore Jobs", "Search job sites for the careers you like."),
        ("04", "📈", "Grow Skills", "Identify skill gaps and learning paths."),
    ]
    for column, step in zip(st.columns(4, gap="medium"), steps):
        with column:
            with st.container(border=True):
                st.caption(f"STEP {step[0]}")
                st.markdown(f"## {step[1]}")
                st.markdown(f"### {step[2]}")
                st.write(step[3])

    st.write("")
    left, right = st.columns(2, gap="medium")

    with left:
        st.markdown('<p class="section-heading">Recommended Careers</p>', unsafe_allow_html=True)
        home_results = st.session_state.get("career_results", [])
        if home_results:
            for result in home_results[:3]:
                with st.container(border=True):
                    st.markdown(f"### {result['title']}")
                    st.write(result["description"] or "No description available.")
                    st.caption(f"Relevance score: {result['score']:.2f}")
        else:
            st.info(
                "Your personalised career suggestions will appear here "
                "after you explore careers."
            )
            st.button("Explore careers", key="home_go_explorer", width="stretch",
                      on_click=go_to, args=("Career Explorer",))

    with right:
        st.markdown('<p class="section-heading">About the career data</p>', unsafe_allow_html=True)
        with st.container(border=True):
            report = get_import_report()
            counts = report.get("final_counts", {})
            st.write(
                f"CareerAI searches **{len(recommender.careers):,} careers**: "
                f"{counts.get('unique_occupations', 0):,} occupations from the "
                f"European Commission's **ESCO {report.get('version', '')}** "
                "classification, plus careers from the original CareerAI "
                "curated list."
            )
            st.caption(
                "ESCO describes occupations and the skills usually linked to "
                "them. It is a taxonomy, not a list of job vacancies, salaries "
                "or employers. Skill lists are general and may not match a "
                "specific employer or country. Curated careers were written "
                "for this project and are not official ESCO records."
            )

    st.divider()
    st.caption("CareerAI | Explore • Learn • Grow")


# --------------------------------
# CAREER EXPLORER
# --------------------------------

def render_result(result, rank):
    with st.container(border=True):
        title_col, save_col = st.columns([4, 1])
        with title_col:
            st.markdown(f"### {rank}. {result['title']}")
            st.caption(source_caption(result))
        with save_col:
            if store.is_saved(result["id"]):
                st.success("Saved")
            else:
                st.button(
                    "Save Career",
                    key=f"save_{result['id']}",
                    on_click=save_career_callback,
                    args=(result,),
                )

        st.write(result["description"] or "No description available.")
        st.caption(
            f"Relevance score {result['score']:.2f} "
            f"(explicit match {result['explicit_match_score']:.2f}, "
            f"text similarity {result['cosine_similarity']:.2f}) — "
            "used only to order results, not a probability."
        )
        st.info("**Why this career?** " + result["explanation"])

        matched = result["matched_skills"]
        st.write(
            "**Matched skills:** "
            + (", ".join(matched) if matched else "No listed skill matches your terms directly")
        )
        if result["matched_interests"]:
            st.write("**Matched interests:** " + ", ".join(result["matched_interests"]))

        missing = result["missing_essential_skills"]
        if missing:
            st.write("**Key skills to develop:** " + ", ".join(missing[:8]))
        optional_missing = [s for s in result["missing_skills"] if s not in missing]
        if len(missing) > 8 or optional_missing:
            with st.expander(f"All {len(result['missing_skills'])} skills you could develop"):
                if missing:
                    st.write("**Essential:** " + ", ".join(missing))
                if optional_missing:
                    st.write("**Optional:** " + ", ".join(optional_missing))

        if result["learning"]:
            st.write("**Suggested learning:** " + ", ".join(result["learning"]))

        links = st.columns(3)
        with links[0]:
            st.button("Learning path", key=f"learn_{result['id']}",
                      on_click=open_career_in, args=("Learning Path", result["id"]))
        with links[1]:
            st.button("Job searches", key=f"jobs_{result['id']}",
                      on_click=open_career_in, args=("Job Opportunities", result["id"]))
        with links[2]:
            if result["esco_uri"]:
                st.link_button("ESCO page ↗", result["esco_uri"])


def open_career_in(page, career_id):
    key = "learning" if page == "Learning Path" else "jobs"
    st.session_state[f"{key}_target"] = career_id
    st.session_state.pop(f"{key}_select", None)
    go_to(page)


def page_explorer():
    st.title("Career Explorer")
    st.write("Discover careers that match your skills and interests.")
    show_flash_messages()

    profile = store.load_profile()
    if profile["skills"] or profile["interests"]:
        st.button(
            "Use my saved profile",
            on_click=set_explorer_inputs,
            args=(profile["skills"], profile["interests"]),
            kwargs={"navigate": False},
        )

    skills = persistent_text_input(
        "Enter your skills (comma-separated)",
        "explorer_skills",
        placeholder="e.g., Python, communication, writing",
        max_chars=500,
    )
    interests = persistent_text_input(
        "What are your interests? (comma-separated)",
        "explorer_interests",
        placeholder="e.g., technology, teaching, research",
        max_chars=500,
    )
    top_k = st.select_slider("Number of suggestions", options=[5, 10, 15, 20], value=10)

    clicked = st.button("Explore Careers", key="explore_careers_button")
    autorun = st.session_state.pop("explorer_autorun", False)

    if clicked or autorun:
        skill_terms = split_terms(skills)
        interest_terms = split_terms(interests)
        if not (skill_terms or interest_terms):
            st.session_state.pop("career_results", None)
            st.session_state["career_search_performed"] = False
            st.warning("Please enter at least one skill or interest.")
        else:
            try:
                results = recommender.recommend(skill_terms, interest_terms, top_k=top_k)
                _, _, unmatched = recommender.analyse_profile(skill_terms, interest_terms)
                st.session_state["career_results"] = results
                st.session_state["career_unmatched"] = unmatched
                st.session_state["career_search_performed"] = True
            except ValueError as error:
                st.error(f"Could not generate recommendations: {error}")
            except Exception:
                logger.exception("Recommendation failed")
                st.error("Something went wrong while generating recommendations. Please try again.")

    if not st.session_state.get("career_search_performed"):
        return

    results = st.session_state.get("career_results", [])
    unmatched = st.session_state.get("career_unmatched", [])
    if unmatched:
        st.warning(
            "These terms were not found in any career and were ignored: "
            + ", ".join(unmatched)
            + ". Check the spelling or try a more common word."
        )

    st.subheader("Career Suggestions")
    if not results:
        st.info("No relevant careers found. Try different or more general skills and interests.")
        return

    st.write(
        "Careers are ranked by how many of your terms they match explicitly "
        "(title and listed skills count most), combined with TF-IDF text similarity."
    )
    for rank, result in enumerate(results, start=1):
        render_result(result, rank)


# --------------------------------
# JOB OPPORTUNITIES
# --------------------------------

def page_jobs():
    st.title("Job Opportunities")
    st.write("Find live vacancies for a career on established job sites.")
    st.info(
        "CareerAI does not have a job-vacancy data source. ESCO lists "
        "occupations and skills, not vacancies, salaries or employers. The "
        "buttons below open a search on an external job site; CareerAI does "
        "not check those results."
    )

    career = choose_career(recommender, store, "jobs")
    if career is None:
        return

    location = st.text_input(
        "Location (optional)", placeholder="e.g. Bengaluru, Remote, Berlin",
        max_chars=100, key="jobs_location",
    ).strip()

    title = career["title"]
    searches = {
        "LinkedIn Jobs": "https://www.linkedin.com/jobs/search/?keywords="
        + quote_plus(title) + ("&location=" + quote_plus(location) if location else ""),
        "Indeed": "https://www.indeed.com/jobs?q=" + quote_plus(title)
        + ("&l=" + quote_plus(location) if location else ""),
        "Google Jobs": "https://www.google.com/search?ibp=htl;jobs&q="
        + quote_plus(f"{title} jobs {location}".strip()),
    }

    with st.container(border=True):
        st.markdown(f"### {title}")
        st.caption(source_caption(career))
        alt_titles = career.get("alt_titles", [])[:8]
        if alt_titles:
            st.write("**Also advertised as:** " + ", ".join(alt_titles))
            st.caption("Try these alternative titles too — employers often use different names.")
        for column, (site, url) in zip(st.columns(len(searches)), searches.items()):
            with column:
                st.link_button(f"Search {site} ↗", url, width="stretch")


# --------------------------------
# LEARNING PATH
# --------------------------------

def page_learning():
    st.title("Personalized Learning Path")
    st.write("Identify the skills you need to develop for your target career.")

    career = choose_career(recommender, store, "learning")
    if career is None:
        return

    profile = store.load_profile()
    user_skills = st.text_input(
        "Your current skills (comma-separated)",
        value=profile["skills"],
        help="Pre-filled from My Profile. Changes here are not saved to your profile.",
        key=f"learning_skills_{career['id']}",
    )

    have, missing = recommender.skill_gap(career, user_skills)
    essential = set(career.get("curated_skills", [])) | set(career.get("essential_skills", []))
    missing_core = [s for s in missing if s in essential]
    missing_optional = [s for s in missing if s not in essential]

    st.caption(source_caption(career))
    st.divider()

    st.subheader("Your Skill Match")
    core_total = len(essential) or 1
    core_have = len([s for s in have if s in essential])
    st.progress(core_have / core_total)
    st.write(
        f"You match **{core_have} of {len(essential)}** key skills listed for this career "
        f"(plus {len(have) - core_have} optional)."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.success(f"Skills you have: {len(have)}")
        st.write(", ".join(have) if have else "No matching skills found yet.")
    with col2:
        st.warning(f"Key skills to develop: {len(missing_core)}")
        st.write(", ".join(missing_core) if missing_core else "You've covered all key skills!")

    st.divider()
    st.subheader("Your Learning Roadmap")

    if not missing_core:
        st.success(
            "You have all the key skills listed for this career. "
            "Consider building a project to demonstrate them."
        )
    else:
        progress = store.load_progress()
        done = set(progress.get(career["id"], progress.get(career["title"], [])))
        st.write("Tick each topic as you learn it. Your progress is saved automatically.")

        new_done = set()
        for index, skill in enumerate(missing_core, start=1):
            with st.container(border=True):
                checked = st.checkbox(
                    f"Step {index}: {skill}",
                    value=skill in done,
                    key=f"learn_{career['id']}_{skill}",
                )
                if checked:
                    new_done.add(skill)
                url = LEARNING_RESOURCES.get(skill.casefold())
                link_cols = st.columns(2)
                with link_cols[0]:
                    if url:
                        st.link_button("Recommended resource ↗", url)
                    else:
                        st.link_button(
                            "Find courses ↗",
                            "https://www.coursera.org/search?query=" + quote_plus(skill),
                        )
                with link_cols[1]:
                    st.link_button(
                        "Video tutorials ↗",
                        "https://www.youtube.com/results?search_query="
                        + quote_plus(f"learn {skill}"),
                    )

        if new_done != done & set(missing_core):
            progress[career["id"]] = sorted(new_done)
            progress.pop(career["title"], None)
            try:
                store.save_progress(progress)
            except OSError:
                logger.exception("Saving learning progress failed")
                st.error("Could not save your learning progress.")

        completed = len(new_done)
        st.progress(completed / len(missing_core))
        st.caption(
            f"Learning progress: {round(completed / len(missing_core) * 100)}% "
            f"({completed} of {len(missing_core)} topics completed)"
        )

    if missing_optional:
        with st.expander(f"Optional skills for this career ({len(missing_optional)})"):
            st.write(", ".join(missing_optional))

    if career.get("learning"):
        st.write("**Suggested learning topics:** " + ", ".join(career["learning"]))


# --------------------------------
# MY PROFILE
# --------------------------------

def page_profile():
    st.title("My Profile")
    st.write("Save your skills and interests to personalise your career recommendations.")
    st.caption(
        "Only skills and interests are used for recommendations. Name and "
        "education are optional. Do not enter sensitive personal information."
    )

    profile = store.load_profile()
    with st.form("profile_form"):
        name = st.text_input("Your name (optional)", value=profile["name"], max_chars=100)
        skills = st.text_input("Your skills (comma-separated)", value=profile["skills"], max_chars=1000)
        interests = st.text_input("Your interests (comma-separated)", value=profile["interests"], max_chars=1000)
        education = st.text_input("Education (optional)", value=profile["education"], max_chars=200)
        submitted = st.form_submit_button("Save Profile")

    if submitted:
        updated = {"name": name, "skills": skills, "interests": interests, "education": education}
        try:
            store.save_profile(updated)
            st.success("Your profile has been saved successfully!")
            profile = store.load_profile()
        except OSError:
            logger.exception("Saving the profile failed")
            st.error("Could not save your profile. Please try again.")

    if profile["skills"] or profile["interests"]:
        st.button(
            "Recommend careers from my profile",
            on_click=set_explorer_inputs,
            args=(profile["skills"], profile["interests"]),
        )


# --------------------------------
# SAVED CAREERS
# --------------------------------

def page_saved():
    st.title("Saved Careers")
    st.write("Your bookmarked careers.")
    show_flash_messages()

    saved = store.load_saved_careers()
    if not saved:
        st.info("No saved careers yet. Use “Save Career” in the Career Explorer.")
        st.button("Go to Career Explorer", on_click=go_to, args=("Career Explorer",))
        return

    for item in saved:
        career = resolve_saved(recommender, item) or item
        with st.container(border=True):
            st.markdown(f"### {item['title']}")
            st.caption(source_caption(career))
            st.write(career.get("description") or "No description available.")
            skills = item.get("skills") or []
            if skills:
                st.write("**Listed skills:** " + ", ".join(skills[:10])
                         + (f" … (+{len(skills) - 10} more)" if len(skills) > 10 else ""))
            cols = st.columns(3)
            with cols[0]:
                st.button("Learning path", key=f"saved_learn_{item['id']}",
                          on_click=open_career_in, args=("Learning Path", career.get("id", item["id"])))
            with cols[1]:
                st.button("Job searches", key=f"saved_jobs_{item['id']}",
                          on_click=open_career_in, args=("Job Opportunities", career.get("id", item["id"])))
            with cols[2]:
                st.button("Remove", key=f"remove_{item['id']}",
                          on_click=remove_career_callback, args=(item["id"], item["title"]))


# --------------------------------
# ROUTER
# --------------------------------

PAGE_RENDERERS = {
    "Home": page_home,
    "Career Explorer": page_explorer,
    "Job Opportunities": page_jobs,
    "Learning Path": page_learning,
    "My Profile": page_profile,
    "Saved Careers": page_saved,
}

try:
    PAGE_RENDERERS.get(page, page_home)()
except Exception:
    logger.exception("Page %s failed", page)
    st.error("Sorry, this page hit an unexpected error. Please try again or choose another page.")
