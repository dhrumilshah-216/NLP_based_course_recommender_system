"""Tests for personalized ranking using a deterministic semantic-search stub."""

import sys
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from recommendations import (
    PersonalizedCourseRecommender,
    RecommendationConfig,
)


class FakeSemanticEngine:
    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def search(self, query, top_k=5):
        self.calls.append(query)
        scores = self.scores.get(query, self.scores.get("*", {}))
        return [
            {
                "course_id": course_id,
                "title": course_id,
                "relevance_score": score,
            }
            for course_id, score in sorted(scores.items(), key=lambda item: -item[1])[:top_k]
        ]


def sample_courses():
    return [
        {
            "course_id": "HARD",
            "title": "Advanced Applied Analytics",
            "level": "High",
            "workload_label": "High",
            "teaching_emphasis": "Practical",
            "practical_relevance": "High",
            "career_relevance": "High",
            "math_complexity": "High",
            "prerequisites": "Foundations of Analytics",
        },
        {
            "course_id": "EASY",
            "title": "Foundations of Analytics",
            "level": "Low",
            "workload_label": "Low",
            "teaching_emphasis": "Theoretical",
            "practical_relevance": "Medium",
            "career_relevance": "Medium",
            "math_complexity": "Low",
            "prerequisites": "None",
        },
    ]


def sample_profiles():
    return [
        {
            "student_profile_id": "STU003",
            "profile_name": "Low workload",
            "interest_priority_1_to_5": "2",
            "scoring_priority_1_to_5": "4",
            "low_workload_priority_1_to_5": "1",
            "career_priority_1_to_5": "2",
            "practical_priority_1_to_5": "2",
            "depth_priority_1_to_5": "3",
            "profile_description": "Prefers lower difficulty and lower workload.",
        },
        {
            "student_profile_id": "STU006",
            "profile_name": "Hands-on learner",
            "interest_priority_1_to_5": "3",
            "scoring_priority_1_to_5": "3",
            "low_workload_priority_1_to_5": "4",
            "career_priority_1_to_5": "5",
            "practical_priority_1_to_5": "5",
            "depth_priority_1_to_5": "4",
            "profile_description": "Prefers practical exercises and projects.",
        },
    ]


class TestPersonalizedCourseRecommendations(unittest.TestCase):
    def setUp(self):
        self.courses = sample_courses()
        self.semantic = FakeSemanticEngine({
            "analytics query": {"HARD": 0.8, "EASY": 0.7},
            "career goal": {"HARD": 0.4, "EASY": 0.0},
            "*": {"HARD": 0.2, "EASY": 0.1},
        })
        self.recommender = PersonalizedCourseRecommender(
            self.semantic,
            self.courses,
            profiles=sample_profiles(),
            aggregates=[
                {
                    "course_id": "HARD",
                    "scoring_potential_positive_share": "0.9",
                    "scoring_potential_review_count": "20",
                    "aggregate_source": "synthetic_reviews_only",
                },
                {
                    "course_id": "EASY",
                    "scoring_potential_positive_share": "0.1",
                    "scoring_potential_review_count": "20",
                    "aggregate_source": "synthetic_reviews_only",
                },
            ],
        )

    def test_query_only_reuses_semantic_ranking(self):
        result = self.recommender.recommend(query="analytics query")

        self.assertEqual(result["recommendations"][0]["course_id"], "HARD")
        first = result["recommendations"][0]
        self.assertEqual(first["semantic_relevance"], 0.8)
        self.assertIsNone(first["preference_match"])
        self.assertEqual(set(first["components"]), {"semantic_relevance"})

    def test_difficulty_and_workload_preferences_change_ranking(self):
        low = self.recommender.recommend(
            query="analytics query",
            preferences={"difficulty": "low", "workload": "low"},
        )
        high = self.recommender.recommend(
            query="analytics query",
            preferences={"difficulty": "high", "workload": "high"},
        )

        self.assertEqual(low["recommendations"][0]["course_id"], "EASY")
        self.assertEqual(high["recommendations"][0]["course_id"], "HARD")
        self.assertIn("difficulty", low["recommendations"][0]["components"])
        self.assertTrue(low["recommendations"][0]["why_recommended"])

    def test_learning_style_and_profile_priorities_are_applied(self):
        result = self.recommender.recommend(
            query="analytics query",
            profile_id="STU006",
        )

        self.assertEqual(result["profile"]["student_profile_id"], "STU006")
        self.assertEqual(result["recommendations"][0]["course_id"], "HARD")
        self.assertIn("learning_style", result["recommendations"][0]["components"])
        self.assertEqual(
            result["recommendations"][0]["components"]["learning_style"]["score"],
            1.0,
        )
        self.assertTrue(result["data_notice"].startswith("Course attributes"))

    def test_incomplete_preferences_omit_unspecified_components(self):
        result = self.recommender.recommend(
            query="analytics query",
            preferences={"difficulty": "low"},
        )
        recommendation = result["recommendations"][0]

        self.assertEqual(set(recommendation["components"]), {"semantic_relevance", "difficulty"})
        self.assertNotIn("workload", recommendation["components"])
        self.assertNotIn("scoring_preference", recommendation["components"])
        self.assertNotIn("interest", recommendation["components"])

    def test_conflicting_preferences_expose_contributions_and_tradeoffs(self):
        result = self.recommender.recommend(
            query="analytics query",
            preferences={"difficulty": "low", "learning_style": "practical"},
        )
        easy = next(item for item in result["recommendations"] if item["course_id"] == "EASY")

        self.assertGreater(easy["components"]["difficulty"]["score"], 0.9)
        self.assertEqual(easy["components"]["learning_style"]["score"], 0.5)
        self.assertTrue(any("learning style" in item for item in easy["tradeoffs"]))
        self.assertIn("weight", easy["components"]["difficulty"])
        self.assertIn("contribution", easy["components"]["difficulty"])

    def test_empty_query_with_preferences_is_allowed_but_empty_request_is_rejected(self):
        result = self.recommender.recommend(preferences={"difficulty": "low"})
        self.assertEqual(result["query"], None)
        self.assertEqual(result["recommendations"][0]["course_id"], "EASY")

        with self.assertRaisesRegex(ValueError, "Provide a query"):
            self.recommender.recommend()

    def test_empty_query_with_profile_uses_profile_description(self):
        result = self.recommender.recommend(profile_id="STU003")

        self.assertEqual(result["recommendations"][0]["course_id"], "EASY")
        self.assertIn("profile_context", result["recommendations"][0]["components"])
        self.assertIn("difficulty", result["recommendations"][0]["components"])

    def test_unknown_profile_is_reported(self):
        with self.assertRaisesRegex(ValueError, "Unknown student profile"):
            self.recommender.recommend(query="analytics query", profile_id="UNKNOWN")

    def test_known_unmet_prerequisite_excludes_course(self):
        result = self.recommender.recommend(
            preferences={"difficulty": "high"},
            completed_courses=[],
            enforce_prerequisites=True,
        )

        self.assertEqual([item["course_id"] for item in result["recommendations"]], ["EASY"])

    def test_unmet_prerequisite_is_disclosed_without_existing_mandatory_policy(self):
        result = self.recommender.recommend(
            preferences={"difficulty": "high"},
            completed_courses=[],
        )

        hard = next(item for item in result["recommendations"] if item["course_id"] == "HARD")
        self.assertEqual(hard["prerequisite_status"], "unmet")
        self.assertTrue(any("Known unmet prerequisite" in item for item in hard["tradeoffs"]))
        self.assertFalse(result["prerequisite_policy"]["enforced"])

    def test_unknown_prerequisites_remain_eligible_with_tradeoff(self):
        self.courses[0]["prerequisites"] = "Advanced Statistics"
        result = self.recommender.recommend(
            preferences={"difficulty": "high"},
            completed_courses=["An unrelated course"],
        )

        hard = next(item for item in result["recommendations"] if item["course_id"] == "HARD")
        self.assertEqual(hard["prerequisite_status"], "unknown")
        self.assertTrue(any("could not be verified" in item for item in hard["tradeoffs"]))

    def test_synthetic_scoring_signal_is_capped_and_not_used_alone(self):
        result = self.recommender.recommend(
            query="analytics query",
            preferences={"scoring_preference": "high"},
        )
        component = result["recommendations"][0]["components"]["scoring_preference"]

        self.assertLessEqual(component["weight"], 0.05)
        self.assertEqual(component["source"], "synthetic_review_reference_labels")
        with self.assertRaisesRegex(ValueError, "cannot be the only ranking signal"):
            self.recommender.recommend(preferences={"scoring_preference": "high"})

    def test_unverified_aggregate_cannot_contribute_to_scoring_preference(self):
        recommender = PersonalizedCourseRecommender(
            self.semantic,
            self.courses,
            profiles=[],
            aggregates=[
                {"course_id": "HARD", "scoring_potential_positive_share": "1.0"},
                {"course_id": "EASY", "scoring_potential_positive_share": "0.0"},
            ],
        )
        result = recommender.recommend(
            query="analytics query",
            preferences={"scoring_preference": "high"},
        )

        for recommendation in result["recommendations"]:
            self.assertNotIn("scoring_preference", recommendation["components"])

    def test_prerequisite_status_is_unknown_when_student_record_not_supplied(self):
        result = self.recommender.recommend(
            query="analytics query",
            preferences={"difficulty": "high"},
        )
        hard = next(item for item in result["recommendations"] if item["course_id"] == "HARD")

        self.assertEqual(hard["prerequisite_status"], "unknown")
        self.assertTrue(any("completion status was not provided" in item for item in hard["tradeoffs"]))

    def test_configurable_weights_are_reported(self):
        config = RecommendationConfig(weights={
            "semantic_relevance": 0.8,
            "difficulty": 0.2,
        })
        recommender = PersonalizedCourseRecommender(
            self.semantic,
            self.courses,
            profiles=[],
            aggregates=[],
            config=config,
        )
        result = recommender.recommend(
            query="analytics query",
            preferences={"difficulty": "low"},
        )

        self.assertEqual(result["ranking_formula"]["configured_base_weights"], config.weights)

    def test_negative_weight_configuration_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite nonnegative"):
            RecommendationConfig(weights={"semantic_relevance": -0.1})


if __name__ == "__main__":
    unittest.main()
