"""
Tests for Phase 1 — Dataset Integration and Validation
=======================================================
Run from the project root with:
    python -m pytest backend/tests/test_data_validation.py -v
Or directly:
    python backend/tests/test_data_validation.py

These tests verify:
  1. CSV file presence
  2. data_loader functions return well-formed data
  3. COURSES_DATABASE is populated (29 CSV courses or 12 hardcoded fallback)
  4. Validation report logic: referential integrity, valid values, etc.
  5. Edge-case behaviour of ValidationReport
"""

import sys
import csv
import unittest
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — ensure backend/ is importable from any working directory
# ---------------------------------------------------------------------------
_BACKEND_DIR = Path(__file__).parent.parent          # .../backend/
_PROJECT_DIR = _BACKEND_DIR.parent                   # .../NLP Mini Project/
_DATASET_DIR = _PROJECT_DIR / "my_college_nlp_dataset"

if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from data_loader import (
    COURSES_CSV, REVIEWS_CSV, ASPECTS_CSV,
    PROFILES_CSV, QUERIES_CSV, AGGREGATES_CSV,
    load_courses_from_csv, load_csv_raw,
    _build_description, _build_tags,
)
from validate_dataset import (
    ValidationReport,
    validate_courses, validate_reviews,
    validate_aspects, validate_profiles,
    validate_queries, validate_aggregates,
    validate_all,
    VALID_ASPECTS, VALID_SENTIMENTS, VALID_CATEGORY_LABELS,
)


# ===========================================================================
# 1. File presence tests
# ===========================================================================
class TestFilePresence(unittest.TestCase):
    """Every dataset CSV must exist on disk."""

    def test_courses_csv_exists(self):
        self.assertTrue(COURSES_CSV.exists(), f"Missing: {COURSES_CSV}")

    def test_reviews_csv_exists(self):
        self.assertTrue(REVIEWS_CSV.exists(), f"Missing: {REVIEWS_CSV}")

    def test_aspects_csv_exists(self):
        self.assertTrue(ASPECTS_CSV.exists(), f"Missing: {ASPECTS_CSV}")

    def test_profiles_csv_exists(self):
        self.assertTrue(PROFILES_CSV.exists(), f"Missing: {PROFILES_CSV}")

    def test_queries_csv_exists(self):
        self.assertTrue(QUERIES_CSV.exists(), f"Missing: {QUERIES_CSV}")

    def test_aggregates_csv_exists(self):
        self.assertTrue(AGGREGATES_CSV.exists(), f"Missing: {AGGREGATES_CSV}")


# ===========================================================================
# 2. data_loader unit tests
# ===========================================================================
class TestDataLoader(unittest.TestCase):
    """Test data_loader helper functions independently."""

    def test_load_courses_from_csv_returns_list(self):
        result = load_courses_from_csv()
        self.assertIsNotNone(result, "load_courses_from_csv() returned None (CSV missing?)")
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

    def test_load_courses_count(self):
        """Dataset README states 29 courses."""
        result = load_courses_from_csv()
        if result is not None:
            self.assertEqual(len(result), 29,
                             f"Expected 29 courses from CSV, got {len(result)}")

    def test_each_course_has_required_keys(self):
        required = {"course_id", "title", "description"}
        result = load_courses_from_csv()
        if result is not None:
            for course in result:
                missing = required - set(course.keys())
                self.assertEqual(missing, set(),
                                 f"Course {course.get('course_id')} missing keys: {missing}")

    def test_description_is_nonempty_string(self):
        result = load_courses_from_csv()
        if result is not None:
            for course in result:
                self.assertIsInstance(course["description"], str)
                self.assertGreater(len(course["description"]), 0,
                                   f"Empty description for {course['course_id']}")

    def test_tags_is_list(self):
        result = load_courses_from_csv()
        if result is not None:
            for course in result:
                self.assertIsInstance(course["tags"], list,
                                      f"tags not a list for {course['course_id']}")

    def test_course_ids_are_unique(self):
        result = load_courses_from_csv()
        if result is not None:
            ids = [c["course_id"] for c in result]
            self.assertEqual(len(ids), len(set(ids)), "Duplicate course IDs found in CSV")

    def test_credits_is_integer(self):
        result = load_courses_from_csv()
        if result is not None:
            for course in result:
                self.assertIsInstance(course["credits"], int,
                                      f"credits not int for {course['course_id']}")
                self.assertGreater(course["credits"], 0)

    def test_is_synthetic_flag(self):
        result = load_courses_from_csv()
        if result is not None:
            for course in result:
                self.assertTrue(course.get("is_synthetic"),
                                f"is_synthetic not True for {course['course_id']}")

    def test_build_description_uses_title(self):
        row = {"course_title": "Test Course", "syllabus_topics": "NLP, ML",
               "tools_technologies": "Python", "topic_cluster": "AI",
               "teaching_emphasis": "Practical", "prerequisites": "None"}
        desc = _build_description(row)
        self.assertIn("Test Course", desc)
        self.assertIn("NLP, ML", desc)
        self.assertIn("Python", desc)

    def test_build_description_skips_none_prerequisites(self):
        row = {"course_title": "X", "prerequisites": "None",
               "syllabus_topics": "", "tools_technologies": "",
               "topic_cluster": "", "teaching_emphasis": ""}
        desc = _build_description(row)
        self.assertNotIn("Prerequisites: None", desc)

    def test_build_tags_deduplicates(self):
        row = {"tools_technologies": "Python; python; SQL",
               "topic_cluster": "python"}
        tags = _build_tags(row)
        self.assertEqual(len(tags), len(set(tags)), "Tags contain duplicates")

    def test_load_csv_raw_returns_list_for_valid_file(self):
        rows = load_csv_raw(COURSES_CSV)
        self.assertIsInstance(rows, list)
        self.assertGreater(len(rows), 0)

    def test_load_csv_raw_returns_empty_list_for_missing(self):
        rows = load_csv_raw(Path("/nonexistent/path.csv"))
        self.assertEqual(rows, [])


# ===========================================================================
# 3. courses_data integration test
# ===========================================================================
class TestCoursesDatabase(unittest.TestCase):
    """Test that COURSES_DATABASE is populated and has expected structure."""

    def setUp(self):
        from courses_data import COURSES_DATABASE, _COURSES_SOURCE
        self.db = COURSES_DATABASE
        self.source = _COURSES_SOURCE

    def test_courses_database_is_nonempty(self):
        self.assertGreater(len(self.db), 0, "COURSES_DATABASE is empty")

    def test_courses_database_uses_csv_source(self):
        self.assertEqual(self.source, "csv",
                         f"Expected CSV source, got: {self.source}")

    def test_courses_database_count_29(self):
        if self.source == "csv":
            self.assertEqual(len(self.db), 29,
                             f"Expected 29 courses from CSV, got {len(self.db)}")

    def test_all_courses_have_course_id(self):
        for c in self.db:
            self.assertIn("course_id", c)
            self.assertTrue(c["course_id"], "Found course with blank course_id")

    def test_all_courses_have_title(self):
        for c in self.db:
            self.assertIn("title", c)
            self.assertTrue(c["title"], f"Blank title for {c.get('course_id')}")

    def test_all_courses_have_description(self):
        for c in self.db:
            self.assertIn("description", c)
            self.assertGreater(len(c["description"]), 0,
                               f"Empty description for {c.get('course_id')}")


# ===========================================================================
# 4. ValidationReport unit tests
# ===========================================================================
class TestValidationReport(unittest.TestCase):
    """Test the ValidationReport helper class directly."""

    def test_empty_report_is_valid(self):
        r = ValidationReport()
        self.assertTrue(r.is_valid)

    def test_error_makes_invalid(self):
        r = ValidationReport()
        r.error("something broke")
        self.assertFalse(r.is_valid)

    def test_warning_does_not_make_invalid(self):
        r = ValidationReport()
        r.warning("minor concern")
        self.assertTrue(r.is_valid)

    def test_to_dict_keys(self):
        r = ValidationReport()
        d = r.to_dict()
        for key in ("valid", "errors", "warnings", "info", "error_count", "warning_count"):
            self.assertIn(key, d)

    def test_summary_contains_status(self):
        r = ValidationReport()
        self.assertIn("PASSED", r.summary())
        r.error("fatal")
        self.assertIn("FAILED", r.summary())

    def test_error_count(self):
        r = ValidationReport()
        r.error("e1")
        r.error("e2")
        self.assertEqual(r.to_dict()["error_count"], 2)


# ===========================================================================
# 5. Per-validator unit tests with fabricated data
# ===========================================================================
class TestValidateCourses(unittest.TestCase):

    def _make_row(self, **overrides):
        base = {
            "course_id": "CSC101", "course_title": "Test",
            "department": "CS", "semester": "1",
            "total_credits": "4", "difficulty_label": "Medium",
            "math_complexity": "Low", "workload_label": "Medium",
            "syllabus_topics": "Python, OOP",
            "data_source_type": "synthetic_reference_inspired",
        }
        base.update(overrides)
        return base

    def test_valid_row_produces_no_errors(self):
        r = ValidationReport()
        validate_courses([self._make_row()], r)
        self.assertTrue(r.is_valid, r.errors)

    def test_missing_course_id_is_error(self):
        r = ValidationReport()
        validate_courses([self._make_row(course_id="")], r)
        self.assertFalse(r.is_valid)

    def test_duplicate_course_id_is_error(self):
        r = ValidationReport()
        validate_courses([self._make_row(), self._make_row()], r)
        self.assertFalse(r.is_valid)

    def test_invalid_credits_string_is_error(self):
        r = ValidationReport()
        validate_courses([self._make_row(total_credits="three")], r)
        self.assertFalse(r.is_valid)

    def test_unexpected_difficulty_label_is_warning(self):
        r = ValidationReport()
        validate_courses([self._make_row(difficulty_label="Extreme")], r)
        self.assertTrue(r.is_valid)          # warning only, still valid
        self.assertGreater(len(r.warnings), 0)

    def test_none_rows_produces_error(self):
        r = ValidationReport()
        validate_courses(None, r)
        self.assertFalse(r.is_valid)


class TestValidateReviews(unittest.TestCase):

    COURSE_IDS = {"CSC101"}

    def _make_row(self, **overrides):
        base = {
            "review_id": "REV001", "course_id": "CSC101",
            "review_text": "Great course",
            "overall_sentiment": "positive",
            "overall_rating_synthetic": "4",
            "is_real_student_feedback": "False",
        }
        base.update(overrides)
        return base

    def test_valid_row(self):
        r = ValidationReport()
        validate_reviews([self._make_row()], self.COURSE_IDS, r)
        self.assertTrue(r.is_valid, r.errors)

    def test_orphan_course_id_is_error(self):
        r = ValidationReport()
        validate_reviews([self._make_row(course_id="NOTEXIST")], self.COURSE_IDS, r)
        self.assertFalse(r.is_valid)

    def test_invalid_sentiment_is_error(self):
        r = ValidationReport()
        validate_reviews(
            [self._make_row(overall_sentiment="great")], self.COURSE_IDS, r
        )
        self.assertFalse(r.is_valid)

    def test_rating_out_of_range_is_error(self):
        r = ValidationReport()
        validate_reviews(
            [self._make_row(overall_rating_synthetic="6")], self.COURSE_IDS, r
        )
        self.assertFalse(r.is_valid)

    def test_duplicate_review_id_is_error(self):
        r = ValidationReport()
        validate_reviews(
            [self._make_row(), self._make_row()], self.COURSE_IDS, r
        )
        self.assertFalse(r.is_valid)


class TestValidateAspects(unittest.TestCase):

    COURSE_IDS = {"CSC101"}
    REVIEW_IDS = {"REV001"}

    def _make_row(self, **overrides):
        base = {
            "review_id": "REV001", "course_id": "CSC101",
            "aspect": "difficulty", "sentiment": "positive",
            "aspect_rating_synthetic": "3",
            "is_human_annotated": "False",
        }
        base.update(overrides)
        return base

    def test_valid_row(self):
        r = ValidationReport()
        validate_aspects([self._make_row()], self.COURSE_IDS, self.REVIEW_IDS, r)
        self.assertTrue(r.is_valid, r.errors)

    def test_invalid_aspect_name(self):
        r = ValidationReport()
        validate_aspects(
            [self._make_row(aspect="grammar")],
            self.COURSE_IDS, self.REVIEW_IDS, r
        )
        self.assertFalse(r.is_valid)

    def test_invalid_sentiment(self):
        r = ValidationReport()
        validate_aspects(
            [self._make_row(sentiment="ok")],
            self.COURSE_IDS, self.REVIEW_IDS, r
        )
        self.assertFalse(r.is_valid)

    def test_orphan_review_id(self):
        r = ValidationReport()
        validate_aspects(
            [self._make_row(review_id="NOTEXIST")],
            self.COURSE_IDS, self.REVIEW_IDS, r
        )
        self.assertFalse(r.is_valid)

    def test_all_valid_aspects_accepted(self):
        from validate_dataset import VALID_ASPECTS
        for aspect in VALID_ASPECTS:
            r = ValidationReport()
            validate_aspects(
                [self._make_row(aspect=aspect)],
                self.COURSE_IDS, self.REVIEW_IDS, r
            )
            self.assertTrue(r.is_valid, f"Aspect '{aspect}' was rejected: {r.errors}")


class TestValidateProfiles(unittest.TestCase):

    def _make_row(self, **overrides):
        base = {
            "student_profile_id": "STU001",
            "profile_name": "Grade-Focused Student",
            "scoring_priority_1_to_5": "5",
            "low_workload_priority_1_to_5": "3",
            "career_priority_1_to_5": "2",
            "practical_priority_1_to_5": "2",
            "depth_priority_1_to_5": "1",
        }
        base.update(overrides)
        return base

    def test_valid_row(self):
        r = ValidationReport()
        validate_profiles([self._make_row()], r)
        self.assertTrue(r.is_valid, r.errors)

    def test_out_of_range_priority_is_warning(self):
        r = ValidationReport()
        validate_profiles([self._make_row(scoring_priority_1_to_5="6")], r)
        self.assertTrue(r.is_valid)   # warning, not error
        self.assertGreater(len(r.warnings), 0)

    def test_non_integer_priority_is_error(self):
        r = ValidationReport()
        validate_profiles([self._make_row(scoring_priority_1_to_5="high")], r)
        self.assertFalse(r.is_valid)

    def test_duplicate_profile_id_is_error(self):
        r = ValidationReport()
        validate_profiles([self._make_row(), self._make_row()], r)
        self.assertFalse(r.is_valid)


# ===========================================================================
# 6. Full integration test against the real dataset
# ===========================================================================
class TestFullDatasetValidation(unittest.TestCase):
    """Run validate_all() against the real dataset and assert it passes cleanly."""

    @classmethod
    def setUpClass(cls):
        cls.report = validate_all()

    def test_no_errors(self):
        self.assertTrue(
            self.report.is_valid,
            f"Dataset validation failed with {len(self.report.errors)} error(s):\n"
            + "\n".join(self.report.errors)
        )

    def test_report_has_info(self):
        self.assertGreater(len(self.report.info), 0,
                           "Validation produced no info messages")

    def test_error_count_matches(self):
        self.assertEqual(
            self.report.to_dict()["error_count"],
            len(self.report.errors)
        )

    def test_warning_count_matches(self):
        self.assertEqual(
            self.report.to_dict()["warning_count"],
            len(self.report.warnings)
        )


# ===========================================================================
# Script entry point
# ===========================================================================
if __name__ == "__main__":
    unittest.main(verbosity=2)
