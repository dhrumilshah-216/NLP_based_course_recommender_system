"""
Dataset Validation Module — My College NLP Dataset
====================================================
Validates the synthetic dataset for:
  - File presence
  - Required column completeness
  - Duplicate primary keys
  - Referential integrity (reviews → courses, aspects → reviews)
  - Valid categorical values (sentiment, aspect names, labels)
  - Numeric range checks (ratings 1-5, credits, priority fields)
  - Provenance flags (is_synthetic, data_source_type)

Usage
-----
  As a module (in tests or app.py endpoint):
      from validate_dataset import validate_all
      report = validate_all()
      print(report.summary())

  As a runnable script:
      python backend/validate_dataset.py
      Exit code 0 = all checks pass, 1 = errors found.
"""

from pathlib import Path
from typing import Dict, List, Optional, Set

# Import path constants from data_loader so they stay in one place
from data_loader import (
    COURSES_CSV,
    REVIEWS_CSV,
    ASPECTS_CSV,
    PROFILES_CSV,
    QUERIES_CSV,
    AGGREGATES_CSV,
    load_csv_raw,
)

# ---------------------------------------------------------------------------
# Expected schema values (from data_dictionary.csv and dataset README)
# ---------------------------------------------------------------------------
VALID_SENTIMENTS:      Set[str] = {"positive", "negative", "neutral", "mixed"}
VALID_ASPECTS:         Set[str] = {
    "difficulty", "workload", "scoring_potential",
    "teaching_quality", "practical_value", "career_relevance",
}
VALID_CATEGORY_LABELS: Set[str] = {"Low", "Medium", "High"}
RATING_MIN,  RATING_MAX  = 1, 5
PRIORITY_MIN, PRIORITY_MAX = 1, 5

REQUIRED_COLS: Dict[str, List[str]] = {
    "courses.csv":              ["course_id", "course_title", "department", "semester",
                                 "total_credits", "difficulty_label", "syllabus_topics"],
    "reviews.csv":              ["review_id", "course_id", "review_text",
                                 "overall_sentiment", "overall_rating_synthetic"],
    "review_aspects.csv":       ["review_id", "course_id", "aspect",
                                 "sentiment", "aspect_rating_synthetic"],
    "student_profiles.csv":     ["student_profile_id", "profile_name"],
    "queries_and_relevance.csv":["query_id", "student_query",
                                 "expected_relevant_course_ids"],
    "course_synthetic_aggregates.csv": ["course_id"],
}

PROFILE_PRIORITY_FIELDS = [
    "scoring_priority_1_to_5",
    "low_workload_priority_1_to_5",
    "career_priority_1_to_5",
    "practical_priority_1_to_5",
    "depth_priority_1_to_5",
]


# ---------------------------------------------------------------------------
# ValidationReport
# ---------------------------------------------------------------------------
class ValidationReport:
    """Accumulates errors and warnings from all validation checks."""

    def __init__(self) -> None:
        self.errors:   List[str] = []
        self.warnings: List[str] = []
        self.info:     List[str] = []

    def error(self,   msg: str) -> None: self.errors.append(f"ERROR:   {msg}")
    def warning(self, msg: str) -> None: self.warnings.append(f"WARNING: {msg}")
    def note(self,    msg: str) -> None: self.info.append(f"INFO:    {msg}")

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def summary(self) -> str:
        lines = [
            "=" * 68,
            "  MY COLLEGE NLP DATASET — VALIDATION REPORT",
            "=" * 68,
        ]
        lines += self.info
        if self.warnings:
            lines.append("")
            lines += self.warnings
        if self.errors:
            lines.append("")
            lines += self.errors
        lines.append("")
        status = "PASSED" if self.is_valid else f"FAILED ({len(self.errors)} error(s))"
        lines.append(
            f"  Status  : {status}"
            f"  |  Errors: {len(self.errors)}"
            f"  |  Warnings: {len(self.warnings)}"
        )
        lines.append("=" * 68)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict (used by the /api/validate endpoint)."""
        return {
            "valid":         self.is_valid,
            "error_count":   len(self.errors),
            "warning_count": len(self.warnings),
            "errors":        self.errors,
            "warnings":      self.warnings,
            "info":          self.info,
        }


# ---------------------------------------------------------------------------
# Per-file validators
# ---------------------------------------------------------------------------

def _check_required_cols(rows: List[Dict], filename: str,
                         report: ValidationReport) -> None:
    """Warn if any required column is completely absent from the file."""
    if not rows:
        return
    required = REQUIRED_COLS.get(filename, [])
    present  = set(rows[0].keys())
    for col in required:
        if col not in present:
            report.error(f"{filename}: required column '{col}' is missing from header")


def validate_courses(rows: Optional[List[Dict]],
                     report: ValidationReport) -> Set[str]:
    """
    Validate courses.csv.
    Returns the set of valid course_ids for downstream cross-file checks.
    """
    if rows is None:
        report.error("courses.csv: file not found")
        return set()

    report.note(f"courses.csv: {len(rows)} row(s) found")
    _check_required_cols(rows, "courses.csv", report)

    course_ids: Set[str] = set()
    for row_num, row in enumerate(rows, start=2):   # row 1 is header
        cid = row.get("course_id", "").strip()

        if not cid:
            report.error(f"courses.csv row {row_num}: blank course_id")
            continue
        if cid in course_ids:
            report.error(f"courses.csv: duplicate course_id '{cid}' (row {row_num})")
        course_ids.add(cid)

        # Required text fields
        for col in ("course_title", "department", "semester", "syllabus_topics"):
            if not row.get(col, "").strip():
                report.warning(f"courses.csv row {row_num} ({cid}): blank '{col}'")

        # Total credits must be a positive integer
        raw_credits = row.get("total_credits", "").strip()
        if raw_credits:
            try:
                c = int(raw_credits)
                if not (1 <= c <= 12):
                    report.warning(
                        f"courses.csv row {row_num} ({cid}): "
                        f"total_credits={c} outside expected range 1–12"
                    )
            except ValueError:
                report.error(
                    f"courses.csv row {row_num} ({cid}): "
                    f"total_credits '{raw_credits}' is not an integer"
                )

        # Categorical labels
        for col in ("difficulty_label", "math_complexity", "workload_label"):
            val = row.get(col, "").strip()
            if val and val not in VALID_CATEGORY_LABELS:
                report.warning(
                    f"courses.csv row {row_num} ({cid}): "
                    f"{col}='{val}' not in {sorted(VALID_CATEGORY_LABELS)}"
                )

        # Provenance
        dst = row.get("data_source_type", "").strip()
        if dst and dst != "synthetic_reference_inspired":
            report.warning(
                f"courses.csv row {row_num} ({cid}): "
                f"unexpected data_source_type='{dst}'"
            )

    report.note(f"courses.csv: {len(course_ids)} unique course IDs")
    return course_ids


def validate_reviews(rows: Optional[List[Dict]],
                     course_ids: Set[str],
                     report: ValidationReport) -> Set[str]:
    """
    Validate reviews.csv.
    Returns the set of valid review_ids for downstream cross-file checks.
    """
    if rows is None:
        report.error("reviews.csv: file not found")
        return set()

    report.note(f"reviews.csv: {len(rows)} row(s) found")
    _check_required_cols(rows, "reviews.csv", report)

    review_ids: Set[str] = set()
    orphans = 0

    for row_num, row in enumerate(rows, start=2):
        rid = row.get("review_id", "").strip()
        cid = row.get("course_id", "").strip()

        if not rid:
            report.error(f"reviews.csv row {row_num}: blank review_id")
            continue
        if rid in review_ids:
            report.error(f"reviews.csv: duplicate review_id '{rid}' (row {row_num})")
        review_ids.add(rid)

        # Referential integrity: course must exist
        if not cid:
            report.error(f"reviews.csv row {row_num} ({rid}): blank course_id")
        elif cid not in course_ids:
            report.error(
                f"reviews.csv row {row_num} ({rid}): "
                f"course_id '{cid}' not found in courses.csv"
            )
            orphans += 1

        # Sentiment
        sentiment = row.get("overall_sentiment", "").strip()
        if sentiment not in VALID_SENTIMENTS:
            report.error(
                f"reviews.csv row {row_num} ({rid}): "
                f"invalid overall_sentiment='{sentiment}' "
                f"(expected one of {sorted(VALID_SENTIMENTS)})"
            )

        # Rating
        raw_rating = row.get("overall_rating_synthetic", "").strip()
        if raw_rating:
            try:
                r = int(raw_rating)
                if not (RATING_MIN <= r <= RATING_MAX):
                    report.error(
                        f"reviews.csv row {row_num} ({rid}): "
                        f"overall_rating_synthetic={r} outside {RATING_MIN}–{RATING_MAX}"
                    )
            except ValueError:
                report.error(
                    f"reviews.csv row {row_num} ({rid}): "
                    f"overall_rating_synthetic '{raw_rating}' is not an integer"
                )

        # Provenance guard
        is_real = row.get("is_real_student_feedback", "").strip().lower()
        if is_real == "true":
            report.warning(
                f"reviews.csv row {row_num} ({rid}): "
                "is_real_student_feedback=True — data should be synthetic"
            )

    if orphans:
        report.note(
            f"reviews.csv: {orphans} review(s) reference course IDs not in courses.csv"
        )
    report.note(f"reviews.csv: {len(review_ids)} unique review IDs")
    return review_ids


def validate_aspects(rows: Optional[List[Dict]],
                     course_ids: Set[str],
                     review_ids: Set[str],
                     report: ValidationReport) -> None:
    """Validate review_aspects.csv for referential integrity and value correctness."""
    if rows is None:
        report.error("review_aspects.csv: file not found")
        return

    report.note(f"review_aspects.csv: {len(rows)} row(s) found")
    _check_required_cols(rows, "review_aspects.csv", report)

    bad_review_refs  = 0
    bad_course_refs  = 0

    for row_num, row in enumerate(rows, start=2):
        rid    = row.get("review_id", "").strip()
        cid    = row.get("course_id", "").strip()
        aspect = row.get("aspect", "").strip()
        sent   = row.get("sentiment", "").strip()

        # Referential integrity
        if rid not in review_ids:
            bad_review_refs += 1
            if bad_review_refs <= 5:   # cap output to first 5 to avoid flooding
                report.error(
                    f"review_aspects.csv row {row_num}: "
                    f"review_id '{rid}' not found in reviews.csv"
                )

        if cid not in course_ids:
            bad_course_refs += 1
            if bad_course_refs <= 5:
                report.error(
                    f"review_aspects.csv row {row_num}: "
                    f"course_id '{cid}' not found in courses.csv"
                )

        # Aspect name
        if aspect not in VALID_ASPECTS:
            report.error(
                f"review_aspects.csv row {row_num}: "
                f"invalid aspect='{aspect}' "
                f"(expected one of {sorted(VALID_ASPECTS)})"
            )

        # Sentiment
        if sent not in VALID_SENTIMENTS:
            report.error(
                f"review_aspects.csv row {row_num}: "
                f"invalid sentiment='{sent}'"
            )

        # Rating
        raw_rating = row.get("aspect_rating_synthetic", "").strip()
        if raw_rating:
            try:
                r = int(raw_rating)
                if not (RATING_MIN <= r <= RATING_MAX):
                    report.error(
                        f"review_aspects.csv row {row_num}: "
                        f"aspect_rating_synthetic={r} outside {RATING_MIN}–{RATING_MAX}"
                    )
            except ValueError:
                report.error(
                    f"review_aspects.csv row {row_num}: "
                    f"aspect_rating_synthetic '{raw_rating}' is not an integer"
                )

        # Provenance
        is_human = row.get("is_human_annotated", "").strip().lower()
        if is_human == "true":
            report.warning(
                f"review_aspects.csv row {row_num}: "
                "is_human_annotated=True — data should be synthetic"
            )

    if bad_review_refs > 5:
        report.error(
            f"review_aspects.csv: {bad_review_refs} total rows with "
            "invalid review_id (showing first 5 above)"
        )
    if bad_course_refs > 5:
        report.error(
            f"review_aspects.csv: {bad_course_refs} total rows with "
            "invalid course_id (showing first 5 above)"
        )


def validate_profiles(rows: Optional[List[Dict]],
                      report: ValidationReport) -> Set[str]:
    """Validate student_profiles.csv."""
    if rows is None:
        report.error("student_profiles.csv: file not found")
        return set()

    report.note(f"student_profiles.csv: {len(rows)} row(s) found")
    _check_required_cols(rows, "student_profiles.csv", report)

    profile_ids: Set[str] = set()
    for row_num, row in enumerate(rows, start=2):
        pid = row.get("student_profile_id", "").strip()
        if not pid:
            report.error(f"student_profiles.csv row {row_num}: blank student_profile_id")
            continue
        if pid in profile_ids:
            report.error(
                f"student_profiles.csv: duplicate student_profile_id '{pid}' (row {row_num})"
            )
        profile_ids.add(pid)

        if not row.get("profile_name", "").strip():
            report.warning(f"student_profiles.csv row {row_num} ({pid}): blank profile_name")

        # Priority integers 1–5
        for field in PROFILE_PRIORITY_FIELDS:
            val = row.get(field, "").strip()
            if val:
                try:
                    v = int(val)
                    if not (PRIORITY_MIN <= v <= PRIORITY_MAX):
                        report.warning(
                            f"student_profiles.csv row {row_num} ({pid}): "
                            f"{field}={v} outside {PRIORITY_MIN}–{PRIORITY_MAX}"
                        )
                except ValueError:
                    report.error(
                        f"student_profiles.csv row {row_num} ({pid}): "
                        f"{field}='{val}' is not an integer"
                    )

    report.note(f"student_profiles.csv: {len(profile_ids)} unique profile IDs")
    return profile_ids


def validate_queries(rows: Optional[List[Dict]],
                     course_ids: Set[str],
                     report: ValidationReport) -> None:
    """Validate queries_and_relevance.csv."""
    if rows is None:
        report.error("queries_and_relevance.csv: file not found")
        return

    report.note(f"queries_and_relevance.csv: {len(rows)} row(s) found")
    _check_required_cols(rows, "queries_and_relevance.csv", report)

    query_ids: Set[str] = set()
    for row_num, row in enumerate(rows, start=2):
        qid = row.get("query_id", "").strip()
        if not qid:
            report.error(f"queries_and_relevance.csv row {row_num}: blank query_id")
            continue
        if qid in query_ids:
            report.error(
                f"queries_and_relevance.csv: duplicate query_id '{qid}' (row {row_num})"
            )
        query_ids.add(qid)

        if not row.get("student_query", "").strip():
            report.error(
                f"queries_and_relevance.csv row {row_num} ({qid}): blank student_query"
            )

        # Validate every referenced course ID
        expected_raw = row.get("expected_relevant_course_ids", "").strip()
        if expected_raw:
            for ref_cid in expected_raw.split(";"):
                ref_cid = ref_cid.strip()
                if ref_cid and ref_cid not in course_ids:
                    report.warning(
                        f"queries_and_relevance.csv row {row_num} ({qid}): "
                        f"expected course_id '{ref_cid}' not found in courses.csv"
                    )

    report.note(f"queries_and_relevance.csv: {len(query_ids)} unique query IDs")


def validate_aggregates(rows: Optional[List[Dict]],
                        course_ids: Set[str],
                        report: ValidationReport) -> None:
    """Validate course_synthetic_aggregates.csv."""
    if rows is None:
        report.error("course_synthetic_aggregates.csv: file not found")
        return

    report.note(f"course_synthetic_aggregates.csv: {len(rows)} row(s) found")
    _check_required_cols(rows, "course_synthetic_aggregates.csv", report)

    agg_ids: Set[str] = set()
    for row_num, row in enumerate(rows, start=2):
        cid = row.get("course_id", "").strip()
        if not cid:
            report.error(
                f"course_synthetic_aggregates.csv row {row_num}: blank course_id"
            )
            continue
        if cid in agg_ids:
            report.error(
                f"course_synthetic_aggregates.csv: duplicate course_id '{cid}'"
            )
        agg_ids.add(cid)

        if cid not in course_ids:
            report.error(
                f"course_synthetic_aggregates.csv row {row_num}: "
                f"course_id '{cid}' not found in courses.csv"
            )

        # Provenance: aggregate_source must indicate synthetic origin
        src = row.get("aggregate_source", "").strip()
        if src and src != "synthetic_reviews_only":
            report.warning(
                f"course_synthetic_aggregates.csv row {row_num} ({cid}): "
                f"unexpected aggregate_source='{src}'"
            )

    # Coverage check: every course should have an aggregate row
    missing_agg = course_ids - agg_ids
    if missing_agg:
        report.warning(
            f"course_synthetic_aggregates.csv: "
            f"{len(missing_agg)} course(s) have no aggregate row: "
            f"{sorted(missing_agg)}"
        )

    report.note(f"course_synthetic_aggregates.csv: {len(agg_ids)} unique course IDs")


# ---------------------------------------------------------------------------
# Cross-file summary checks
# ---------------------------------------------------------------------------

def _cross_file_checks(course_rows: Optional[List[Dict]],
                       review_rows: Optional[List[Dict]],
                       report: ValidationReport) -> None:
    """Verify expected review count (30 per course) as a dataset integrity signal."""
    if course_rows and review_rows:
        expected = len(course_rows) * 30
        actual   = len(review_rows)
        if actual != expected:
            report.warning(
                f"Expected {expected} reviews "
                f"(30 × {len(course_rows)} courses), found {actual}. "
                "This is a dataset completeness warning, not a structural error."
            )
        else:
            report.note(f"Review count: {actual} = 30 per course (ok)")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def validate_all() -> ValidationReport:
    """
    Run all dataset validation checks and return a ValidationReport.
    This is the function called by the /api/validate endpoint and the tests.
    """
    report = ValidationReport()

    # Load all files upfront (load_csv_raw returns [] if file is missing)
    def _load(path):
        rows = load_csv_raw(path)
        return rows if rows else None

    course_rows = _load(COURSES_CSV)
    review_rows = _load(REVIEWS_CSV)
    aspect_rows = _load(ASPECTS_CSV)
    profile_rows = _load(PROFILES_CSV)
    query_rows   = _load(QUERIES_CSV)
    agg_rows     = _load(AGGREGATES_CSV)

    # Per-file validation
    course_ids = validate_courses(course_rows, report)
    review_ids = validate_reviews(review_rows,  course_ids, report)
    validate_aspects(aspect_rows, course_ids, review_ids, report)
    validate_profiles(profile_rows, report)
    validate_queries(query_rows, course_ids, report)
    validate_aggregates(agg_rows, course_ids, report)

    # Cross-file checks
    _cross_file_checks(course_rows, review_rows, report)

    return report


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    result = validate_all()
    print(result.summary())
    sys.exit(0 if result.is_valid else 1)
