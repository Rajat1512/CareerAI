"""Content-based career recommender for CareerAI.

Ranking approach (see README "How recommendations are ranked")
---------------------------------------------------------------
Every career is scored against the user's skills and interests with two
signals, then combined:

1. Explicit match score (the main signal). Each user term ("python",
   "photography", "machine learning") is matched as a whole phrase against
   the career title, alternative titles, ESCO essential/optional skills,
   curated skills, curated interests and the description. Word variants are
   unified with a small suffix stemmer (photography ~ photographer,
   programming ~ programmer) and matching is done on whole words only, so
   "data" never matches "database". Each term is weighted by its inverse
   document frequency across the catalogue, so a rare, specific term such as
   "sanskrit" counts for more than a generic one such as "communication".
   The score is the weighted share of the user's terms that the career
   matches, where a title match counts more than a skill match, which counts
   more than a description mention.

2. TF-IDF cosine similarity between the profile and the career text (title,
   alternative titles, description, skills). This catches related wording
   that is not an explicit phrase match and breaks ties.

Related terms from QUERY_EXPANSIONS (e.g. sanskrit -> languages) add a
small amount. The weights were compared on data/evaluation_profiles.json
(python -m src.run_evaluation --ablation) rather than tuned per test.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from functools import lru_cache

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from src.catalogue import all_skills, load_default_catalogue
from src.preprocessing import normalize_list


SKILL_SYNONYMS = {
    "ai": "artificial intelligence",
    "artificial intelligence": "artificial intelligence",
    "ml": "machine learning",
    "machine learning": "machine learning",
    "coding": "programming",
    "programming": "programming",
    "software development": "software development",
    "photographer": "photography",
    "photography": "photography",
    "translator": "translation",
    "translation": "translation",
    "language": "languages",
    "languages": "languages",
    "sanskrit": "sanskrit",
    "python": "python",
    "sql": "sql",
    "data analysis": "data analysis",
    "analytics": "analytics",
    "predictive modeling": "prediction",
    "predictive modelling": "prediction",
    "forecasting": "forecasting",
    "phd": "doctoral research",
    "doctorate": "doctoral research",
    "doctoral student": "doctoral research",
    "research phd": "doctoral research",
    "predict": "prediction",
    "predictive analytics": "predictive analytics",
    "time series": "time series",
    "cyber security": "cybersecurity",
    "infosec": "cybersecurity",
    "information security": "cybersecurity",
    "dl": "deep learning",
    "nlp": "natural language processing",
    "js": "javascript",
    "ui": "user interface",
    "ux": "user experience",
    "hr": "human resources",
}


def normalize_term(value):
    """Normalize known equivalent terms while preserving unknown terms."""
    term = " ".join(str(value).strip().lower().split())
    return SKILL_SYNONYMS.get(term, term)


QUERY_EXPANSIONS = {
    "sanskrit": ["translation", "languages", "literature", "classical languages"],
    "photography": ["visual arts", "camera"],
    "artificial intelligence": ["machine learning"],
    "machine learning": ["artificial intelligence", "model evaluation"],
    "programming": ["software development", "computer programming"],
    "software development": ["programming"],
    "data analysis": ["analytics", "statistics"],
    "analytics": ["data analysis"],
    "prediction": [
        "predictive analytics", "forecasting", "statistics", "machine learning",
    ],
    "predictive analytics": ["prediction", "statistics", "forecasting"],
    "forecasting": ["prediction", "time series", "statistics"],
    "doctoral research": ["research", "academia", "scientific writing"],
    "research": ["research methods", "scientific research methodology"],
    "statistics": ["probability", "regression", "statistical modelling"],
    "time series": ["forecasting", "statistics"],
    "cybersecurity": ["cyber security", "information security"],
    "gardening": ["horticulture", "plants"],
    "cooking": ["food preparation", "culinary"],
    "writing": ["editing", "copywriting"],
    "nursing": ["patient care", "healthcare"],
}


def expand_profile_terms(values):
    """Return normalized input terms plus explicitly related terms."""
    expanded = set()
    concept_phrases = sorted(
        set(SKILL_SYNONYMS) | set(QUERY_EXPANSIONS),
        key=len,
        reverse=True,
    )

    for value in values:
        if value is None or not str(value).strip():
            continue

        raw_term = " ".join(str(value).strip().lower().split())
        term = normalize_term(raw_term)
        expanded.add(term)
        expanded.update(QUERY_EXPANSIONS.get(term, []))

        # Natural-language entries may contain several known concepts.
        padded = f" {normalize_match_text(raw_term)} "
        for phrase in concept_phrases:
            if phrase != raw_term and f" {phrase} " in padded:
                canonical = normalize_term(phrase)
                expanded.add(canonical)
                expanded.update(QUERY_EXPANSIONS.get(canonical, []))

    return sorted(expanded)


# --------------------------------------------------------------------------
# Text normalization and light stemming
# --------------------------------------------------------------------------

STOP_WORDS = frozenset(ENGLISH_STOP_WORDS) | {"etc", "eg", "ie", "using"}

# Compound words written both ways in the catalogue.
COMPOUNDS = {
    "cybersecurity": "cyber security",
    "e-learning": "elearning",
    "front-end": "frontend",
    "back-end": "backend",
}

_IRREGULAR_STEMS = {
    "analysis": "analys", "analyses": "analys", "analyse": "analys",
    "analysing": "analys", "analysed": "analys",
    "analytics": "analyt", "analytic": "analyt", "analytical": "analyt",
    "children": "child",
}

# Words whose ending looks like a suffix but is not one; stemming them would
# create false matches (engineer -> engine, community -> communication).
_PROTECTED = frozenset({
    "engineer", "computer", "water", "paper", "power", "flower", "layer",
    "tower", "master", "number", "member", "letter", "order", "career",
    "community", "music", "logic", "topic", "public", "basic", "tutor",
    "floor", "door", "motor", "mirror", "error", "sector", "factor", "vector",
    "major", "minor", "news", "series", "species", "business", "physics",
    "physician", "army", "monitor", "interior", "senior", "junior",
})

_SUFFIXES = (
    "ically", "ation", "ition", "ator", "ment", "ness", "ical", "ian",
    "ity", "ing", "ant", "ate", "ial", "ism", "ist", "ic", "ed", "er", "or",
)

_VOWELS = set("aeiou")


@lru_cache(maxsize=100_000)
def stem(token: str) -> str:
    """Conservative suffix stemmer used for matching (not linguistics)."""
    if token in _IRREGULAR_STEMS:
        return _IRREGULAR_STEMS[token]
    if len(token) <= 3 or not token.isalpha() or token in _PROTECTED:
        return token

    token = re.sub(r"([iy])z(e|es|ed|ing|ation|ations|er|ers)$", r"\1s\2", token)

    if token.endswith("ies") and len(token) > 4:
        token = token[:-3] + "y"
    elif token.endswith("sses"):
        token = token[:-2]
    elif token.endswith("s") and not token.endswith(("ss", "us", "is")):
        token = token[:-1]

    if token in _PROTECTED or token in _IRREGULAR_STEMS:
        return _IRREGULAR_STEMS.get(token, token)

    for _ in range(2):
        for suffix in _SUFFIXES:
            if token.endswith(suffix) and len(token) - len(suffix) >= 3:
                base = token[: -len(suffix)]
                # coding -> code, baker -> bake (consonant-vowel-consonant).
                if (
                    suffix in ("ing", "ed", "er")
                    and len(base) == 3
                    and base[0] not in _VOWELS
                    and base[1] in _VOWELS
                    and base[2] not in _VOWELS | {"w", "x", "y"}
                ):
                    base += "e"
                token = base
                break
        else:
            break
        if token in _PROTECTED:
            return token

    if len(token) >= 5 and token.endswith("e"):
        token = token[:-1]
    if len(token) >= 6 and token.endswith("y"):
        token = token[:-1]
    if (
        len(token) >= 4
        and token[-1] == token[-2]
        and token[-1] not in _VOWELS
    ):
        token = token[:-1]
    return token


def normalize_match_text(value):
    """Lower-case text, unify apostrophes and keep only word characters."""
    value = str(value).casefold().replace("’", "'")
    for compound, replacement in COMPOUNDS.items():
        value = value.replace(compound, replacement)
    value = re.sub(r"[^a-z0-9+#]+", " ", value)
    return " ".join(value.split())


@lru_cache(maxsize=200_000)
def text_stems(text: str) -> tuple[str, ...]:
    """Stemmed content words of a text, in order, without stop words."""
    return tuple(
        stem(token)
        for token in normalize_match_text(text).split()
        if token not in STOP_WORDS
    )


def contains_phrase(haystack: tuple, needle: tuple) -> bool:
    """True if `needle` occurs as a contiguous run inside `haystack`."""
    n = len(needle)
    if not n or n > len(haystack):
        return False
    first = needle[0]
    for i in range(len(haystack) - n + 1):
        if haystack[i] == first and haystack[i:i + n] == needle:
            return True
    return False


def skill_matches_term(skill, term):
    """Match a whole phrase (with word variants) inside a skill label."""
    return contains_phrase(text_stems(str(skill)), text_stems(str(term)))


def _analyzer(text: str) -> list[str]:
    stems = text_stems(text)
    return list(stems) + [f"{a} {b}" for a, b in zip(stems, stems[1:])]


# --------------------------------------------------------------------------
# Explanations
# --------------------------------------------------------------------------

def build_explanation(
    matched_skills,
    matched_interests,
    missing_skills,
    matched_terms=None,
    title_terms=None,
    related_terms=None,
):
    """Build a concise explanation from actual profile matches."""
    reasons = []

    if title_terms:
        reasons.append(
            "The job title matches your input: " + ", ".join(sorted(title_terms))
        )

    if matched_skills:
        reasons.append(
            "Your profile matches these listed skills: "
            + ", ".join(list(matched_skills)[:6])
        )

    if matched_interests:
        reasons.append(
            "Your interests align with: " + ", ".join(sorted(matched_interests))
        )

    if related_terms and not (matched_skills or matched_interests or title_terms):
        reasons.append(
            "Related to your input through: " + ", ".join(sorted(related_terms)[:4])
        )

    if matched_terms and not (matched_skills or matched_interests or title_terms):
        reasons.append(
            "The occupation description mentions: "
            + ", ".join(sorted(matched_terms))
        )

    if missing_skills:
        reasons.append(
            "Potential skills to develop: " + ", ".join(list(missing_skills)[:4])
        )

    if not any((title_terms, matched_skills, matched_interests,
                related_terms, matched_terms)):
        if missing_skills:
            reasons.insert(0, "This career has textual similarity to your profile")
        else:
            return "This career has textual similarity to your profile."

    return ". ".join(reasons) + "."


# --------------------------------------------------------------------------
# Recommender
# --------------------------------------------------------------------------

DEFAULT_WEIGHTS = {
    "explicit": 0.55,   # IDF-weighted share of user terms explicitly matched
    "cosine": 0.30,     # TF-IDF cosine similarity
    "related": 0.05,    # explicit matches of expansion terms
}
# The final score is the weighted sum divided by the sum of the weights, so
# it stays between 0 and 1. A fourth "depth" term (number of matched skills)
# was tried and removed: it did not improve any metric in the ablation.

# How strongly each field of a career counts as evidence for a term.
FIELD_STRENGTH = {
    "title": 1.0,
    "alt_title": 0.85,
    "curated_skill": 0.9,
    "essential_skill": 0.8,
    "optional_skill": 0.6,
    "interest": 0.7,
    "description": 0.4,
}

SKILL_FIELDS = ("curated_skill", "essential_skill", "optional_skill")
MIN_COSINE_WITHOUT_MATCH = 0.05
INTEREST_WEIGHT = 0.8


def _split_inputs(values):
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    parts = []
    for value in values:
        if value is None:
            continue
        for part in re.split(r"[,;\n]+", str(value)):
            part = " ".join(part.split())
            if part:
                parts.append(part)
    return parts


class _CatalogueIndex:
    """Pre-computed phrase index and TF-IDF matrix for one catalogue."""

    def __init__(self, careers):
        self.careers = careers
        # items[i] = (career_index, field, label, stems)
        self.items = []
        self.postings = defaultdict(set)
        self.titles = {}

        for c_index, career in enumerate(careers):
            fields = [("title", career["title"])]
            if career.get("esco_title") and career["esco_title"].casefold() != career["title"].casefold():
                fields.append(("alt_title", career["esco_title"]))
            fields += [("alt_title", label) for label in career.get("alt_titles", [])]
            fields += [("curated_skill", s) for s in career.get("curated_skills", [])]
            fields += [("essential_skill", s) for s in career.get("essential_skills", [])]
            fields += [("optional_skill", s) for s in career.get("optional_skills", [])]
            fields += [("interest", s) for s in career.get("interests", [])]
            if career.get("description"):
                fields.append(("description", career["description"]))
            if career.get("curated_summary"):
                fields.append(("description", career["curated_summary"]))

            for field, label in fields:
                # ESCO qualifiers such as "Java (computer programming)" name
                # a category, not the skill; matching them would claim that
                # "programming" matches Java.
                match_text = label
                if field in SKILL_FIELDS:
                    match_text = re.sub(r"\([^)]*\)", " ", label)
                stems = text_stems(match_text)
                if not stems:
                    continue
                item_index = len(self.items)
                self.items.append((c_index, field, label, stems))
                for token in set(stems):
                    self.postings[token].add(item_index)

        documents = [self._document(career) for career in careers]
        self.vectorizer = TfidfVectorizer(
            analyzer=_analyzer,
            sublinear_tf=True,
            min_df=1,
        )
        self.matrix = self.vectorizer.fit_transform(documents)

    @staticmethod
    def _document(career):
        parts = [career["title"], career["title"], career.get("esco_title", "")]
        parts += career.get("alt_titles", [])
        parts.append(career.get("description", ""))
        parts.append(career.get("curated_summary", ""))
        parts += all_skills(career)
        parts += career.get("interests", [])
        parts += career.get("learning", [])
        return " . ".join(p for p in parts if p)

    def match(self, stems):
        """Return {career_index: [(field, label), ...]} for a phrase."""
        if not stems:
            return {}
        candidate_sets = [self.postings.get(token) for token in set(stems)]
        if any(not s for s in candidate_sets):
            return {}
        candidates = set.intersection(*sorted(candidate_sets, key=len))
        hits = defaultdict(list)
        for item_index in candidates:
            c_index, field, label, item_stems = self.items[item_index]
            if contains_phrase(item_stems, stems):
                hits[c_index].append((field, label))
        return hits


@lru_cache(maxsize=2)
def _default_index(catalogue_key):
    careers, _report = load_default_catalogue()
    return _CatalogueIndex(careers)


def _prepare_provided(careers):
    """Accept records in the curated schema or the merged schema."""
    if not isinstance(careers, list) or not careers:
        raise ValueError("Career dataset cannot be empty.")
    prepared = []
    for position, career in enumerate(careers):
        if not isinstance(career, dict) or not str(career.get("title", "")).strip():
            raise ValueError("Every career must be an object with a title.")
        record = {
            "id": str(career.get("id") or f"provided:{position}"),
            "title": str(career["title"]).strip(),
            "esco_title": career.get("esco_title", ""),
            "alt_titles": normalize_list(career.get("alt_titles", [])),
            "description": str(career.get("description", "") or "").strip(),
            "essential_skills": list(career.get("essential_skills", []) or []),
            "optional_skills": list(career.get("optional_skills", []) or []),
            "curated_skills": list(career.get("curated_skills", []) or []),
            "interests": list(career.get("interests", []) or []),
            "learning": list(career.get("learning", []) or []),
            "source": career.get("source", "Provided catalogue"),
            "source_detail": career.get("source_detail", ""),
            "esco_uri": career.get("esco_uri", ""),
            "isco_group": career.get("isco_group", ""),
        }
        if not (record["essential_skills"] or record["optional_skills"]
                or record["curated_skills"]):
            record["curated_skills"] = list(career.get("skills", []) or [])
        prepared.append(record)
    return prepared


class ContentRecommender:
    """Recommend careers from the merged ESCO + curated catalogue."""

    def __init__(self, careers=None, weights=None):
        if careers is None:
            careers_list, report = load_default_catalogue()
            self._index = _default_index(id(careers_list))
            self.catalogue_report = report
            self.dataset_source = (
                f"ESCO v1.2.1 ({report['esco_records']} occupations) + "
                f"CareerAI curated ({report['curated_records']} careers, "
                f"{len(report['curated_merged_into_esco'])} merged into ESCO)"
            )
        else:
            self._index = _CatalogueIndex(_prepare_provided(careers))
            self.catalogue_report = {"total_records": len(self._index.careers)}
            self.dataset_source = "Provided catalogue"

        self.careers = self._index.careers
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights:
            unknown = set(weights) - set(DEFAULT_WEIGHTS)
            if unknown:
                raise ValueError(f"Unknown weight names: {sorted(unknown)}")
            self.weights.update(weights)
        self._by_id = {career["id"]: career for career in self.careers}
        self._df_cache = {}

    # ---- profile analysis -------------------------------------------------

    def _document_frequency(self, stems):
        if stems not in self._df_cache:
            self._df_cache[stems] = self._index.match(stems)
        return self._df_cache[stems]

    def analyse_profile(self, skills, interests=None):
        """Turn raw inputs into weighted concepts.

        Returns (concepts, related, unmatched_terms). Each concept is a dict
        with label, stems, weight and hits ({career_index: [(field, label)]}).
        """
        n_careers = len(self.careers)
        concepts = {}
        unmatched = []

        def add_concept(label, origin_weight):
            stems = text_stems(label)
            if not stems:
                return False
            hits = self._document_frequency(stems)
            if not hits:
                return False
            idf = math.log((n_careers + 1) / (len(hits) + 1)) + 1.0
            weight = idf * origin_weight
            existing = concepts.get(stems)
            if existing is None or existing["weight"] < weight:
                concepts[stems] = {
                    "label": label, "stems": stems, "weight": weight,
                    "hits": hits,
                }
            return True

        raw_inputs = (
            [(term, 1.0) for term in _split_inputs(skills)]
            + [(term, INTEREST_WEIGHT) for term in _split_inputs(interests)]
        )
        seen = set()
        for raw, origin_weight in raw_inputs:
            canonical = normalize_term(raw)
            if canonical in seen:
                continue
            seen.add(canonical)

            if add_concept(canonical, origin_weight):
                continue
            if canonical != raw.lower() and add_concept(raw.lower(), origin_weight):
                continue

            # A phrase or sentence with no exact match: use its known concepts
            # and individual content words instead.
            pieces = [
                normalize_term(p)
                for p in expand_profile_terms([raw])
                if normalize_term(p) in SKILL_SYNONYMS.values()
                or p in SKILL_SYNONYMS
            ]
            words = [
                normalize_term(w)
                for w in normalize_match_text(raw).split()
                if w not in STOP_WORDS and len(w) > 1
            ]
            matched_any = False
            if len(words) > 1 or pieces:
                for piece in dict.fromkeys(pieces + words):
                    matched_any |= add_concept(piece, origin_weight * 0.8)
            if not matched_any:
                unmatched.append(raw)

        original_labels = {c["label"] for c in concepts.values()}
        related = {}
        for term in expand_profile_terms([c["label"] for c in concepts.values()]):
            stems = text_stems(term)
            if not stems or stems in concepts or term in original_labels:
                continue
            hits = self._document_frequency(stems)
            if hits:
                idf = math.log((n_careers + 1) / (len(hits) + 1)) + 1.0
                related[stems] = {"label": term, "stems": stems,
                                  "weight": idf, "hits": hits}

        return list(concepts.values()), list(related.values()), unmatched

    # ---- recommendation ---------------------------------------------------

    @staticmethod
    def _validate_top_k(top_k):
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise ValueError("top_k must be a whole number.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        return top_k

    def recommend(self, skills, interests=None, top_k=5):
        """Return the top_k careers for a profile, best first."""
        top_k = self._validate_top_k(top_k)

        concepts, related, _unmatched = self.analyse_profile(skills, interests)
        raw_terms = _split_inputs(skills) + _split_inputs(interests)
        if not raw_terms:
            return []

        query_text = " . ".join(
            [normalize_term(t) for t in raw_terms]
            + [c["label"] for c in concepts]
            + [r["label"] for r in related]
        )
        query_vector = self._index.vectorizer.transform([query_text])
        cosine_scores = linear_kernel(query_vector, self._index.matrix).ravel()

        total_weight = sum(c["weight"] for c in concepts)
        related_weight = sum(r["weight"] for r in related)

        candidates = set()
        for concept in concepts + related:
            candidates.update(concept["hits"])
        candidates.update(
            int(i) for i in cosine_scores.nonzero()[0]
            if cosine_scores[i] >= MIN_COSINE_WITHOUT_MATCH
        )

        weights = self.weights
        weight_total = sum(weights.values()) or 1.0
        results = []

        for c_index in candidates:
            career = self.careers[c_index]
            cosine = float(min(max(cosine_scores[c_index], 0.0), 1.0))

            explicit = 0.0
            matched_skill_labels = []
            matched_interest_labels = []
            title_terms = []
            description_terms = []
            matched_terms = []

            for concept in concepts:
                hits = concept["hits"].get(c_index)
                if not hits:
                    continue
                strength = max(FIELD_STRENGTH[field] for field, _ in hits)
                explicit += concept["weight"] * strength
                matched_terms.append(concept["label"])
                for field, label in hits:
                    if field in SKILL_FIELDS:
                        matched_skill_labels.append(label)
                    elif field == "interest":
                        matched_interest_labels.append(label)
                    elif field in ("title", "alt_title"):
                        title_terms.append(concept["label"])
                    else:
                        description_terms.append(concept["label"])

            related_score = 0.0
            related_skill_labels = []
            related_terms = []
            for concept in related:
                hits = concept["hits"].get(c_index)
                if not hits:
                    continue
                strength = max(FIELD_STRENGTH[field] for field, _ in hits)
                related_score += concept["weight"] * strength
                related_terms.append(concept["label"])
                related_skill_labels += [
                    label for field, label in hits if field in SKILL_FIELDS
                ]

            explicit = explicit / total_weight if total_weight else 0.0
            related_score = related_score / related_weight if related_weight else 0.0

            if explicit == 0 and related_score == 0 and cosine < MIN_COSINE_WITHOUT_MATCH:
                continue

            matched_skill_labels = list(dict.fromkeys(matched_skill_labels))

            score = (
                weights["explicit"] * explicit
                + weights["cosine"] * cosine
                + weights["related"] * related_score
            ) / weight_total
            if not math.isfinite(score) or score <= 0:
                continue

            results.append((score, cosine, explicit, c_index, {
                "matched_skills": matched_skill_labels,
                "matched_interests": list(dict.fromkeys(matched_interest_labels)),
                "matched_terms": list(dict.fromkeys(matched_terms)),
                "title_terms": list(dict.fromkeys(title_terms)),
                "description_terms": list(dict.fromkeys(description_terms)),
                "related_terms": related_terms,
                "related_skills": [
                    s for s in dict.fromkeys(related_skill_labels)
                    if s not in matched_skill_labels
                ],
            }))

        # Deterministic order: score, then cosine, then title.
        results.sort(key=lambda r: (-r[0], -r[1], self.careers[r[3]]["title"].casefold()))

        return [
            self._format_result(score, cosine, explicit, self.careers[c_index], details)
            for score, cosine, explicit, c_index, details in results[:top_k]
        ]

    def _format_result(self, score, cosine, explicit, career, details):
        matched = set(details["matched_skills"])
        missing_curated = [s for s in career.get("curated_skills", []) if s not in matched]
        missing_essential = [s for s in career.get("essential_skills", []) if s not in matched]
        missing_optional = [s for s in career.get("optional_skills", []) if s not in matched]
        missing = missing_curated + missing_essential + missing_optional

        explanation = build_explanation(
            details["matched_skills"],
            details["matched_interests"],
            missing,
            matched_terms=details["description_terms"],
            title_terms=details["title_terms"],
            related_terms=details["related_terms"],
        )

        score = round(float(score), 6)
        return {
            "id": career["id"],
            "title": career["title"],
            "description": career.get("description", ""),
            "score": score,
            "ranking_score": score,
            "cosine_similarity": round(float(cosine), 6),
            "explicit_match_score": round(float(explicit), 6),
            "matched_terms": details["matched_terms"],
            "matched_skills": details["matched_skills"],
            "matched_interests": details["matched_interests"],
            "related_skills": details["related_skills"],
            "missing_skills": missing,
            "missing_essential_skills": missing_curated + missing_essential,
            "skills": all_skills(career),
            "essential_skills": list(career.get("essential_skills", [])),
            "optional_skills": list(career.get("optional_skills", [])),
            "curated_skills": list(career.get("curated_skills", [])),
            "interests": list(career.get("interests", [])),
            "learning": list(career.get("learning", [])),
            "explanation": explanation,
            "source": career.get("source", ""),
            "source_detail": career.get("source_detail", ""),
            "esco_uri": career.get("esco_uri", ""),
            "isco_group": career.get("isco_group", ""),
        }

    # ---- helpers for the UI ----------------------------------------------

    def get_career(self, career_id):
        return self._by_id.get(career_id)

    def search_titles(self, query, limit=20):
        """Find careers whose title or alternative titles contain the query."""
        stems = text_stems(query or "")
        if not stems:
            return []
        hits = self._index.match(stems)
        scored = []
        for c_index, fields in hits.items():
            best = max(
                (FIELD_STRENGTH[f] for f, _ in fields if f in ("title", "alt_title")),
                default=0,
            )
            if best:
                title = self.careers[c_index]["title"]
                extra_words = len(text_stems(title)) - len(stems)
                scored.append((-best, extra_words, title.casefold(), c_index))
        scored.sort()
        return [self.careers[item[-1]] for item in scored[:limit]]

    def skill_gap(self, career, user_skills):
        """Split a career's skills into those the user has and those to learn."""
        concepts = [text_stems(normalize_term(s)) for s in _split_inputs(user_skills)]
        concepts = [c for c in concepts if c]
        have, missing = [], []
        for label in all_skills(career):
            label_stems = text_stems(label)
            if any(contains_phrase(label_stems, c) for c in concepts):
                have.append(label)
            else:
                missing.append(label)
        return have, missing
