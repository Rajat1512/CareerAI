# CareerAI

CareerAI is a career recommendation app built with Python, Streamlit and
scikit-learn. You enter your skills and interests, and it:

- recommends matching careers and explains **why** each one was suggested,
- lists the skills you already match and the skills to develop,
- builds a learning checklist for a target career,
- links to job-site searches for a career,
- lets you save careers and come back to them.

## Career data

| Source | Records | What it is |
|---|---|---|
| **ESCO v1.2.1** (English), European Commission | 3,039 occupations, 13,475 linked skills, 126,051 occupation–skill links (67,600 essential, 58,451 optional) | Official European taxonomy of occupations and skills |
| **CareerAI curated list** (`data/careers.json`) | 49 careers | Hand-written for this project, *not* official ESCO records |

- When a curated career has **exactly the same title** as an ESCO occupation
  (16 cases, such as Photographer, Translator and Data Scientist), the two are
  **merged** into one record. The ESCO id, description and skills are kept, and
  the curated skills and interests are added in separate fields.
- The other 33 curated careers (such as Sanskrit Teacher and AI/ML Engineer)
  are added as separate records labelled *CareerAI curated*.
- **Total searchable careers: 3,072.** Every result shows its source.
- The app **requires** `data/careers_esco.json`. If the file is missing or
  invalid, the app stops with instructions for restoring it. It never silently
  falls back to the small curated list.

**Limitations of the data.** ESCO describes occupations and the skills usually
linked to them. It is **not** a database of job vacancies, salaries or
employers, and its skill lists are general European descriptions rather than
the requirements of a specific employer or country. CareerAI has no
job-vacancy data source: the *Job Opportunities* page only opens searches on
external job sites.

> **Note on earlier figures.** An earlier import report gave 126,407
> relationship rows. That number was inflated by duplicate concept rows in the
> ESCO CSV export: 4 occupations and 21 skills appear twice with different
> `modifiedDate` values. The importer now removes those duplicates. The
> correct count is 126,051, which equals the number of rows in
> `occupationSkillRelations_en.csv`.

ESCO is © European Union. Check the terms of use on the ESCO portal
(https://esco.ec.europa.eu) before redistributing the data.

## Project structure

```
app.py                         Streamlit app (all six pages)
src/
  catalogue.py                 loads, validates and merges ESCO + curated data
  content_recommender.py       recommendation engine
  storage.py                   profile / saved careers / progress storage
  import_esco.py               rebuilds data/careers_esco.json from ESCO CSVs
  run_evaluation.py            Precision@K / Recall@K / NDCG@K evaluation
  evaluation.py                metric functions
  preprocessing.py             loads the curated careers.json
  demo_queries.py              prints recommendations for example profiles
data/
  careers_esco.json            ESCO catalogue used by the app (6.5 MB, required)
  esco_import_report.json      counts and cleaning report from the import
  careers.json                 curated careers
  evaluation_profiles.json     labelled evaluation profiles
  esco_raw/                    original ESCO CSV files (only needed to re-import)
reports/                       evaluation output (generated)
tests/                         pytest test suite
user_data/                     your profile and saved careers (created by the app, git-ignored)
.streamlit/config.toml         theme, and hides stack traces from users
```

## Setup on Windows (PowerShell)

These steps need Python 3.11, which is the version this project was verified
with. Run them from the project folder:

```powershell
cd D:\DATA\car_ai
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements-dev.txt
```

(`requirements.txt` holds only what the app needs. `requirements-dev.txt`
adds pytest.)

### Run the app

```powershell
.venv\Scripts\python -m streamlit run app.py
```

Then open http://localhost:8501. The first page load builds the search index,
which takes about 5 seconds; after that the index is cached.

### Run the tests

```powershell
.venv\Scripts\python -m pytest -q
```

### Run the evaluation

```powershell
.venv\Scripts\python -m src.run_evaluation --ablation
```

This writes `reports/evaluation_report.csv` (per profile) and
`reports/evaluation_summary.json` (averages, ablation and settings).

### Rebuild the ESCO catalogue (only if needed)

The three ESCO v1.2.1 English CSV files must be in `data/esco_raw/`. If they
are missing, download the ESCO classification (CSV format, English) from
https://esco.ec.europa.eu/en/use-esco/download.

```powershell
.venv\Scripts\python -m src.import_esco
```

## How recommendations are ranked

1. **Normalising input.** Input is split on commas and matched
   case-insensitively. Duplicate terms are removed and synonyms are unified
   (for example *AI* → *artificial intelligence* and *coding* →
   *programming*). A light stemmer unifies word forms such as *photography* /
   *photographer* and *programming* / *programmer*. Matching uses whole words
   only, so *data* never matches *database*.
2. **Explicit matches (main signal, weight 0.55).** Each term is looked up as
   a whole phrase in the career's title and alternative titles, essential,
   optional and curated skills, curated interests and description. A title
   match counts more than a skill match, which counts more than a mention in
   the description. Each term is weighted by how rare it is in the catalogue
   (IDF), so *sanskrit* counts for more than *communication*. ESCO qualifiers
   such as "(computer programming)" are ignored when matching, so typing
   *programming* does not claim that you know Java.
3. **TF-IDF cosine similarity (weight 0.30).** This compares the wording of
   the profile with the career text, and also breaks ties.
4. **Related terms (weight 0.05).** Related terms from a small hand-written
   list (for example *sanskrit* → *languages*) add a small amount.

The final score is the weighted average, between 0 and 1. It is used only to
order results and is not a probability. Results are deterministic, and ties
are broken by similarity and then by title. Unknown terms are ignored and the
app tells the user which ones were not found. Missing skills come only from
the career's own record, listed in this order: curated, essential, optional.

## Evaluation

`data/evaluation_profiles.json` contains 13 profiles: photography,
Sanskrit/languages, software development, data analytics, machine learning,
data science, cybersecurity, cloud, gardening, writing, finance, nursing and
cooking. Each profile has a hand-labelled list of relevant careers taken from
the full 3,072-record catalogue. The labels were set by keyword-searching the
catalogue titles **before** the current ranking was run on them.

Results from the latest run (`python -m src.run_evaluation --ablation`, K = 5):

| Variant | Precision@5 | Recall@5 | NDCG@5 |
|---|---|---|---|
| TF-IDF cosine only | 0.785 | 0.342 | 0.823 |
| Explicit matching only | 0.769 | 0.342 | 0.810 |
| Hybrid without related terms | 0.785 | 0.353 | 0.836 |
| **Hybrid (default)** | **0.800** | **0.358** | **0.847** |

How to read these results:

- **This is a small regression set labelled by one person.** It is not a
  representative benchmark, and the numbers are not an accuracy figure.
  Accuracy is not a suitable metric for ranking anyway.
- Recall@5 is low **by construction**: most profiles have 6–19 relevant
  careers, so at most 5 can appear in the top 5.
- The variants differ by only one or two ranking changes across 13 profiles,
  so the differences are **not statistically meaningful**. The default hybrid
  was chosen because it is best or tied on every metric, and because the
  explicit matches give honest explanations. A fourth "depth" signal was
  tested and removed because it did not help.
- **Historical figures** (Precision@5 0.680, Recall@5 0.950, NDCG@5 0.905)
  came from an older version: 5 profiles, labelled only with curated titles,
  scored against the 49-career catalogue. They cannot be compared with the
  numbers above.

Weak spots seen in the evaluation:

- Broad finance profiles also surface management roles that list
  "financial management", such as Insurance agency manager.
- Python/programming profiles rank a few lecturer and engineer roles above
  some developer titles.

## Configuration

No API keys or passwords are needed. Two optional environment variables are
described in `.env.example`:

| Variable | Default | Meaning |
|---|---|---|
| `CAREERAI_STORAGE` | `file` | `file` saves your profile, saved careers and progress as JSON in `user_data/`. `session` keeps them only for the current browser session. |
| `CAREERAI_DATA_DIR` | `user_data` | Folder for those JSON files when using `file`. |

PowerShell example:

```powershell
$env:CAREERAI_STORAGE = "session"
```

## Deployment (recommended: Streamlit Community Cloud)

**Why Streamlit Community Cloud?** It runs Streamlit apps straight from a
GitHub repository and needs no server, database or paid service. The data
files fit comfortably: the app needs only `data/careers_esco.json` (6.5 MB),
and the largest raw CSV is 28 MB, below GitHub's 100 MB per-file limit.

> The app has **not** been deployed yet. Follow these steps when you are ready.

1. **Put the project on GitHub.** In PowerShell:

   ```powershell
   cd D:\DATA\car_ai
   git init
   git add .
   git status
   ```

   Check that `user_data/`, `.venv/`, `.env` and `_backup_original.zip` are
   **not** listed; `.gitignore` excludes them.

   Then commit:

   ```powershell
   git commit -m "CareerAI ready for deployment"
   ```

   Create an empty repository on github.com and push to it, following the
   commands GitHub shows you.
2. **Include these data files** (they are committed by default):
   - `data/careers_esco.json` and `data/careers.json` — **required**.
   - `data/esco_import_report.json` — shown in the app.
   - `data/evaluation_profiles.json` — for the evaluation.
   - `data/esco_raw/` — optional; only needed to re-run the import.
3. Go to https://share.streamlit.io and sign in with GitHub. Choose **Create
   app**, select your repository and branch, and set **Main file path** to
   `app.py`.
4. In **Advanced settings**, choose **Python 3.11**. Under **Secrets**, add:

   ```toml
   CAREERAI_STORAGE = "session"
   ```

   Root-level secrets are available to the app as environment variables.
   Session mode stops different visitors from seeing each other's profiles
   and saved careers, because the server's disk is shared and not permanent.
5. Choose **Deploy**, wait for the build, then open the URL and run the
   checklist below.

### Deployment checklist

- [ ] `python -m pytest -q` passes locally.
- [ ] `python -m src.run_evaluation` completes.
- [ ] `git status` shows no `user_data/`, `.env`, `.venv/` or zip files.
- [ ] `data/careers_esco.json` and `data/careers.json` are committed.
- [ ] The Python version is set to 3.11 and the secret `CAREERAI_STORAGE = "session"` is added.
- [ ] On the live URL:
  - [ ] Home → enter "photography" → Explore Careers shows Photographer first.
  - [ ] Save a career, open Saved Careers, then open its Learning path.
  - [ ] Job Opportunities shows the "no vacancy data source" notice and external search buttons.
  - [ ] The sidebar shows "ESCO v1.2.1 — 3,039 occupations".

## Known limitations

- The ranking is purely content-based: it learns nothing from user feedback.
  It handles synonyms only through a small hand-written list, so unusual
  wording may be ignored. The app shows a warning when that happens.
- The stemmer is a simple rule set made for this vocabulary. It is not a full
  linguistic stemmer.
- ESCO skill lists are long and general. "Skills to develop" can list 20–40
  items for some occupations.
- With file storage, all users of one running app share the same
  `user_data/` folder. That is fine on your own computer; use
  `CAREERAI_STORAGE=session` for public hosting.
- In session mode, saved careers disappear when the browser session ends.
- There is no job-vacancy data. The Job Opportunities page links to external
  searches, and their results are not checked.
