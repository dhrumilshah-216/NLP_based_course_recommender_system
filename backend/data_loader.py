"""
Data Loader Module
Reads the synthetic My College NLP dataset from CSV files and converts
the course rows into a format compatible with COURSES_DATABASE so that
ml_engine.py does not need to change its interface.

Key contract with ml_engine.py:
  - Every course dict MUST contain at minimum: course_id, title, description
  - description is used for semantic embedding - richer text = better search
  - All additional fields are returned as-is in search results
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional

_BACKEND_DIR = Path(__file__).parent
_DATASET_DIR = _BACKEND_DIR.parent / "my_college_nlp_dataset"

# Public path constants (used by validate_dataset.py and tests)
COURSES_CSV   = _DATASET_DIR / "courses.csv"
REVIEWS_CSV   = _DATASET_DIR / "reviews.csv"
ASPECTS_CSV   = _DATASET_DIR / "review_aspects.csv"
PROFILES_CSV  = _DATASET_DIR / "student_profiles.csv"
QUERIES_CSV   = _DATASET_DIR / "queries_and_relevance.csv"
AGGREGATES_CSV = _DATASET_DIR / "course_synthetic_aggregates.csv"


def _build_description(row: Dict[str, str]) -> str:
    """
    Build a rich natural-language description from CSV fields for semantic indexing.

    Combining title + syllabus_topics + tools_technologies + topic_cluster gives
    the embedding model enough vocabulary to match diverse student queries
    (e.g. "I want to learn SQL" → CSC301 Database Management Systems).
    """
    parts: List[str] = []

    title = row.get("course_title", "").strip()
    if title:
        parts.append(title + ".")

    syllabus = row.get("syllabus_topics", "").strip()
    if syllabus:
        parts.append(f"Topics covered: {syllabus}.")

    tools = row.get("tools_technologies", "").strip()
    if tools:
        parts.append(f"Tools and technologies: {tools}.")

    cluster = row.get("topic_cluster", "").strip()
    if cluster:
        parts.append(f"Subject area: {cluster}.")

    emphasis = row.get("teaching_emphasis", "").strip()
    if emphasis:
        parts.append(f"Teaching emphasis: {emphasis}.")

    prereqs = row.get("prerequisites", "").strip()
    if prereqs and prereqs.lower() not in ("", "none"):
        parts.append(f"Prerequisites: {prereqs}.")

    return " ".join(parts)


def _build_tags(row: Dict[str, str]) -> List[str]:
    """Build a deduplicated tag list from tools_technologies and topic_cluster."""
    tags: List[str] = []
    seen = set()

    for raw_tool in row.get("tools_technologies", "").split(";"):
        tag = raw_tool.strip().lower()
        if tag and tag not in seen:
            tags.append(tag)
            seen.add(tag)

    cluster = row.get("topic_cluster", "").strip().lower()
    if cluster and cluster not in seen:
        tags.append(cluster)

    return tags


def load_courses_from_csv() -> Optional[List[Dict]]:
    """
    Load courses from courses.csv.

    Returns a list of course dicts on success, or None if the file is missing.
    The dict keys are a superset of those in the original COURSES_DATABASE so
    existing callers (ml_engine, app.py search endpoint) continue to work.
    """
    if not COURSES_CSV.exists():
        return None

    courses: List[Dict] = []
    with open(COURSES_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            course_id = row.get("course_id", "").strip()
            if not course_id:
                continue  # skip blank rows (e.g. trailing newline)

            try:
                credits = int(row.get("total_credits", "3"))
            except (ValueError, TypeError):
                credits = 3

            courses.append({
                # ---- Core fields (required by ml_engine.py) ----
                "course_id":   course_id,
                "title":       row.get("course_title", "").strip(),
                "description": _build_description(row),
                "tags":        _build_tags(row),
                # ---- Extended metadata (returned in search results) ----
                "department":           row.get("department", "").strip(),
                "semester":             row.get("semester", "").strip(),
                "course_type":          row.get("course_type", "").strip(),
                "credits":              credits,
                "prerequisites":        row.get("prerequisites", "").strip(),
                "level":                row.get("difficulty_label", "").strip(),
                "math_complexity":      row.get("math_complexity", "").strip(),
                "workload_label":       row.get("workload_label", "").strip(),
                "teaching_emphasis":    row.get("teaching_emphasis", "").strip(),
                "syllabus_topics":      row.get("syllabus_topics", "").strip(),
                "tools_technologies":   row.get("tools_technologies", "").strip(),
                "topic_cluster":        row.get("topic_cluster", "").strip(),
                "practical_relevance":  row.get("practical_relevance", "").strip(),
                "career_relevance":     row.get("career_relevance", "").strip(),
                "interest_breadth":     row.get("interest_breadth", "").strip(),
                "conceptual_intensity": row.get("conceptual_intensity", "").strip(),
                # ---- Provenance (dataset integrity) ----
                "data_source_type": row.get("data_source_type", "").strip(),
                "is_synthetic":     True,
            })

    return courses if courses else None


def load_csv_raw(filepath: Path) -> List[Dict]:
    """
    Load any CSV file and return a list of row dicts (keyed by header names).
    Returns an empty list if the file does not exist.
    Used by validate_dataset.py and tests.
    """
    if not filepath.exists():
        return []
    with open(filepath, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
