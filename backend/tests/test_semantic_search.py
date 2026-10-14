"""Unit tests for semantic course ranking without downloading a model."""

import sys
import tempfile
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml_engine import CourseRecommenderEngine, _course_search_text


class FakeSentenceModel:
    """Small deterministic encoder for exercising the ranking pipeline."""

    def __init__(self):
        self.course_encode_calls = 0

    @staticmethod
    def _encode_one(text):
        normalized = text.lower()
        if "python" in normalized or "machine learning" in normalized or "patterns from data" in normalized:
            return [1.0, 0.0, 0.0]
        if "analytics" in normalized or "data analysis" in normalized:
            return [0.0, 1.0, 0.0]
        if "practical" in normalized or "hands-on" in normalized:
            return [0.0, 0.0, 1.0]
        return [0.0, 0.0, 0.0]

    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            return self._encode_one(texts)
        self.course_encode_calls += 1
        return [self._encode_one(text) for text in texts]


def make_courses():
    return [
        {
            "course_id": "CS101",
            "title": "Python and Machine Learning",
            "description": "Build predictive models with Python.",
            "department": "Computer Science",
            "syllabus_topics": "Machine learning, model evaluation",
            "tools_technologies": "Python",
            "teaching_emphasis": "Practical",
            "practical_relevance": "High",
        },
        {
            "course_id": "BA201",
            "title": "Data Analytics",
            "description": "",
            "department": "Business",
            "syllabus_topics": "Data analysis and reporting",
            "tools_technologies": "SQL",
        },
        {"course_id": "EMPTY", "title": "", "description": ""},
    ]


class TestSemanticCourseSearch(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.cache_path = Path(self.temporary_directory.name) / "embeddings.json"

    def tearDown(self):
        self.temporary_directory.cleanup()

    def make_engine(self, courses=None):
        engine = CourseRecommenderEngine(
            courses=make_courses() if courses is None else courses,
            cache_path=self.cache_path,
        )
        engine.model = FakeSentenceModel()
        engine.reindex_courses()
        return engine

    def test_natural_language_queries_rank_matching_courses(self):
        engine = self.make_engine()

        python_result = engine.search("I want to learn Python and machine learning")
        paraphrase_result = engine.search("I want to learn patterns from data")
        career_result = engine.search("I want a career in data analytics")
        practical_result = engine.search("I prefer hands-on practical work")

        self.assertEqual(python_result[0]["course_id"], "CS101")
        self.assertEqual(paraphrase_result[0]["course_id"], "CS101")
        self.assertEqual(career_result[0]["course_id"], "BA201")
        self.assertEqual(practical_result[0]["course_id"], "CS101")
        self.assertEqual(python_result[0]["relevance_score"], 1.0)
        self.assertIn("Tools and technologies: Python", python_result[0]["match_explanation"])

    def test_result_has_ids_titles_and_raw_cosine_score(self):
        result = self.make_engine().search("Python")[0]

        self.assertEqual(result["course_id"], "CS101")
        self.assertEqual(result["title"], "Python and Machine Learning")
        self.assertIsInstance(result["relevance_score"], float)
        self.assertEqual(result["relevance_score"], 1.0)
        self.assertEqual(result["similarity_score"], 100.0)

    def test_missing_description_uses_other_course_fields(self):
        course = make_courses()[1]
        self.assertIn("Data Analytics", _course_search_text(course))
        self.assertIn("Data analysis and reporting", _course_search_text(course))

        results = self.make_engine().search("career in analytics")
        self.assertIn("BA201", [course["course_id"] for course in results])

    def test_empty_query_is_rejected(self):
        engine = self.make_engine()
        for query in ("", "   ", None):
            with self.subTest(query=query), self.assertRaises(ValueError):
                engine.search(query)

    def test_empty_course_catalogue_returns_no_results(self):
        engine = self.make_engine(courses=[])
        self.assertEqual(engine.search("an unfamiliar topic"), [])
        self.assertEqual(engine.model.course_encode_calls, 0)

    def test_courses_without_searchable_text_are_skipped(self):
        results = self.make_engine().search("an unfamiliar topic", top_k=10)
        self.assertNotIn("EMPTY", [course["course_id"] for course in results])

    def test_unknown_topic_and_ties_have_stable_order(self):
        engine = self.make_engine()
        first = engine.search("an unrelated astronomy topic", top_k=10)
        second = engine.search("an unrelated astronomy topic", top_k=10)

        self.assertEqual(
            [course["course_id"] for course in first],
            [course["course_id"] for course in second],
        )
        self.assertEqual(first[0]["relevance_score"], 0.0)

    def test_course_embeddings_are_persisted_and_reused(self):
        first_engine = self.make_engine()
        self.assertEqual(first_engine.model.course_encode_calls, 1)
        self.assertTrue(self.cache_path.exists())

        second_engine = self.make_engine()
        self.assertEqual(second_engine.model.course_encode_calls, 0)
        self.assertEqual(second_engine.search("Python")[0]["course_id"], "CS101")

    def test_changed_course_text_invalidates_cached_embeddings(self):
        self.make_engine()
        changed_courses = make_courses()
        changed_courses[0]["title"] = "Python for Data Science"
        changed_engine = self.make_engine(changed_courses)

        self.assertEqual(changed_engine.model.course_encode_calls, 1)


if __name__ == "__main__":
    unittest.main()
