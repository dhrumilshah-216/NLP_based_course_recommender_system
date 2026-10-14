"""Tests for synthetic course-review analysis and provenance separation."""

import sys
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from review_analysis import ASPECTS, CourseReviewAnalyzer, analyze_course_reviews


class TestCourseReviewAnalyzer(unittest.TestCase):
    def test_actual_dataset_returns_separate_synthetic_references_and_predictions(self):
        result = analyze_course_reviews("CSC101")

        self.assertEqual(result["review_count"], 30)
        self.assertEqual(result["data_provenance"]["type"], "synthetic_sample_reviews")
        self.assertEqual(
            sum(result["overall_sentiment_reference"]["sentiment_counts"].values()),
            result["review_count"],
        )
        self.assertEqual(set(result["aspects"]), set(ASPECTS))
        self.assertEqual(
            result["aspects"]["difficulty"]["synthetic_reference_labels"]["human_annotated"],
            False,
        )
        self.assertNotIn(
            "overall_sentiment_reference",
            result["aspects"]["difficulty"]["prediction"],
        )
        self.assertEqual(
            result["aspects"]["difficulty"]["prediction"]["method"],
            "keyword_phrase_rules",
        )
        self.assertIn("synthetic", result["data_provenance"]["notice"].lower())

    def test_rule_predictions_group_by_aspect_and_distinguish_sentiments(self):
        reviews = [
            {
                "review_id": "R1",
                "course_id": "C1",
                "review_text": "The learning curve was reasonable. The difficult sections needed extra resources.",
                "overall_sentiment": "positive",
            },
            {
                "review_id": "R2",
                "course_id": "C1",
                "review_text": "The workload felt steady rather than overwhelming.",
                "overall_sentiment": "mixed",
            },
            {
                "review_id": "R3",
                "course_id": "C1",
                "review_text": "The assignments piled up near deadlines.",
                "overall_sentiment": "negative",
            },
            {
                "review_id": "R4",
                "course_id": "C1",
                "review_text": "The workload was mentioned without an opinion.",
                "overall_sentiment": "neutral",
            },
        ]
        result = CourseReviewAnalyzer(reviews, []).analyze_course("C1")

        self.assertEqual(
            result["aspects"]["difficulty"]["prediction"]["sentiment_counts"],
            {"positive": 0, "negative": 0, "neutral": 0, "mixed": 1},
        )
        self.assertEqual(
            result["aspects"]["workload"]["prediction"]["sentiment_counts"],
            {"positive": 1, "negative": 1, "neutral": 1, "mixed": 0},
        )
        self.assertEqual(
            result["overall_sentiment_reference"]["sentiment_counts"],
            {"positive": 1, "negative": 1, "neutral": 1, "mixed": 1},
        )

    def test_aspect_labels_are_grouped_without_becoming_predictions(self):
        reviews = [{
            "review_id": "R1",
            "course_id": "C1",
            "review_text": "The rubric made it easier to understand where marks came from.",
            "overall_sentiment": "positive",
        }]
        labels = [
            {
                "review_id": "R1",
                "course_id": "C1",
                "aspect": "scoring_potential",
                "sentiment": "mixed",
            },
            {
                "review_id": "R1",
                "course_id": "C1",
                "aspect": "teaching_quality",
                "sentiment": "negative",
            },
        ]
        result = CourseReviewAnalyzer(reviews, labels).analyze_course("C1")

        self.assertEqual(
            result["aspects"]["scoring_potential"]["prediction"]["sentiment_counts"]["positive"],
            1,
        )
        self.assertEqual(
            result["aspects"]["scoring_potential"]["synthetic_reference_labels"]["sentiment_counts"]["mixed"],
            1,
        )
        self.assertEqual(
            result["aspects"]["teaching_quality"]["synthetic_reference_labels"]["sentiment_counts"]["negative"],
            1,
        )

    def test_duplicate_reviews_and_aspect_pairs_do_not_inflate_counts(self):
        review = {
            "review_id": "R1",
            "course_id": "C1",
            "review_text": "The practical tasks were useful.",
            "overall_sentiment": "positive",
        }
        label = {
            "review_id": "R1",
            "course_id": "C1",
            "aspect": "practical_value",
            "sentiment": "positive",
        }
        result = CourseReviewAnalyzer([review, review.copy()], [label, label.copy()]).analyze_course("C1")

        self.assertEqual(result["review_count"], 1)
        self.assertEqual(result["overall_sentiment_reference"]["sentiment_counts"]["positive"], 1)
        self.assertEqual(
            result["aspects"]["practical_value"]["prediction"]["sentiment_counts"]["positive"],
            1,
        )
        self.assertEqual(
            result["aspects"]["practical_value"]["synthetic_reference_labels"]["labeled_review_count"],
            1,
        )
        self.assertEqual(result["data_quality"]["duplicate_review_rows_excluded"], 1)
        self.assertEqual(result["data_quality"]["duplicate_aspect_label_rows_excluded"], 1)

    def test_empty_text_and_missing_aspect_labels_are_safe(self):
        reviews = [
            {"review_id": "R1", "course_id": "C1", "review_text": "", "overall_sentiment": "neutral"},
            {"review_id": "R2", "course_id": "C1", "review_text": None, "overall_sentiment": ""},
        ]
        result = CourseReviewAnalyzer(reviews, []).analyze_course("C1")

        self.assertEqual(result["review_count"], 2)
        self.assertEqual(result["data_quality"]["empty_review_text_count"], 2)
        self.assertEqual(
            result["aspects"]["difficulty"]["prediction"]["sentiment_counts"],
            {"positive": 0, "negative": 0, "neutral": 0, "mixed": 0},
        )
        self.assertEqual(
            result["aspects"]["difficulty"]["synthetic_reference_labels"]["labeled_review_count"],
            0,
        )

    def test_course_with_no_reviews_returns_empty_aspect_summaries(self):
        result = CourseReviewAnalyzer([], []).analyze_course("NO_REVIEWS")

        self.assertEqual(result["review_count"], 0)
        self.assertEqual(
            result["overall_sentiment_reference"]["sentiment_counts"],
            {"positive": 0, "negative": 0, "neutral": 0, "mixed": 0},
        )
        for summary in result["aspects"].values():
            self.assertEqual(summary["prediction"]["review_count"], 0)
            self.assertEqual(summary["synthetic_reference_labels"]["labeled_review_count"], 0)

    def test_orphan_aspect_labels_are_excluded_and_reported(self):
        labels = [{
            "review_id": "MISSING",
            "course_id": "C1",
            "aspect": "difficulty",
            "sentiment": "negative",
        }]
        result = CourseReviewAnalyzer([], labels).analyze_course("C1")

        self.assertEqual(result["data_quality"]["unmatched_aspect_label_rows_excluded"], 1)
        self.assertEqual(
            result["aspects"]["difficulty"]["synthetic_reference_labels"]["labeled_review_count"],
            0,
        )


if __name__ == "__main__":
    unittest.main()
