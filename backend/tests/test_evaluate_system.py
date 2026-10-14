import sys
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from evaluate_system import _keyword_ranking, _rank_metrics, _supported_preferences


class EvaluationHelpersTests(unittest.TestCase):
    def test_binary_retrieval_metrics_use_expected_relevant_set(self):
        metrics = _rank_metrics(["A", "B", "C"], {"B", "C"}, 2)
        self.assertEqual(metrics["precision@2"], 0.5)
        self.assertEqual(metrics["recall@2"], 0.5)
        self.assertAlmostEqual(
            metrics["ndcg@2"],
            (1 / 1.584962500721156) / (1 + 1 / 1.584962500721156),
        )

    def test_keyword_baseline_is_deterministic_and_uses_course_text(self):
        courses = [
            {"course_id": "B", "title": "Python and machine learning"},
            {"course_id": "A", "title": "Python programming basics"},
            {"course_id": "C", "title": "Business communication"},
        ]
        self.assertEqual(_keyword_ranking("learn Python", courses), ["A", "B", "C"])
        self.assertEqual(_keyword_ranking("learn Python", courses), ["A", "B", "C"])

    def test_only_supported_preference_intents_are_mapped(self):
        row = {
            "intent_field_1": "difficulty",
            "intent_value_1": "Low",
            "intent_field_2": "scoring_potential",
            "intent_value_2": "High",
            "intent_field_3": "career_relevance",
            "intent_value_3": "High",
            "intent_field_4": "practical_relevance",
            "intent_value_4": "High",
        }
        self.assertEqual(
            _supported_preferences(row),
            {
                "difficulty": "low",
                "scoring_preference": "high",
                "learning_style": "practical",
            },
        )


if __name__ == "__main__":
    unittest.main()
