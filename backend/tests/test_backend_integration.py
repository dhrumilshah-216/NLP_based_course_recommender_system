"""Endpoint integration checks for the existing FastAPI application."""

import asyncio
import sys
import unittest
from pathlib import Path

import httpx

BACKEND_DIR = Path(__file__).parent.parent
PROJECT_DIR = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import app as backend_app
from recommendations import get_recommendation_service


class BackendIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.original_engine_model = backend_app.engine.model
        self.original_engine_loaded = backend_app.ENGINE_LOADED
        self.original_nlp_loaded = backend_app.NLP_LOADED
        self.original_search = backend_app.engine.search
        self.original_nlp_functions = (
            backend_app.preprocess_text if backend_app.NLP_LOADED else None,
            backend_app.pos_tagging if backend_app.NLP_LOADED else None,
            backend_app.extract_entities if backend_app.NLP_LOADED else None,
            backend_app.tfidf_cosine if backend_app.NLP_LOADED else None,
        )
        backend_app.ENGINE_LOADED = True
        backend_app.engine.model = object()
        backend_app.engine.search = lambda query, top_k=5: [
            {
                "course_id": "CSC101",
                "title": "Programming Fundamentals",
                "description": "Programming logic and Python",
                "relevance_score": 0.72,
                "similarity_score": 72.0,
            },
            {
                "course_id": "CSC601",
                "title": "Machine Learning",
                "description": "Python and machine learning",
                "relevance_score": 0.55,
                "similarity_score": 55.0,
            },
        ][:top_k] if query else []
        backend_app.NLP_LOADED = True
        backend_app.preprocess_text = lambda text: {"nltk_tokens": [text[:20]]}
        backend_app.pos_tagging = lambda text: {"pos_tags": [[text[:4], "TEST"]]}
        backend_app.extract_entities = lambda text: {"entities": [{"text": text[:4]}]}
        backend_app.tfidf_cosine = lambda texts: {"cosine_similarities": [float(bool(texts))]}
        get_recommendation_service.cache_clear()

    def tearDown(self):
        backend_app.engine.model = self.original_engine_model
        backend_app.ENGINE_LOADED = self.original_engine_loaded
        backend_app.NLP_LOADED = self.original_nlp_loaded
        backend_app.engine.search = self.original_search
        if self.original_nlp_functions[0] is not None:
            (
                backend_app.preprocess_text,
                backend_app.pos_tagging,
                backend_app.extract_entities,
                backend_app.tfidf_cosine,
            ) = self.original_nlp_functions
        get_recommendation_service.cache_clear()

    async def _request(self, method, path, **kwargs):
        transport = httpx.ASGITransport(app=backend_app.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.request(method, path, **kwargs)

    def test_existing_frontend_and_catalogue_routes(self):
        root = asyncio.run(self._request("GET", "/"))
        static_script = asyncio.run(self._request("GET", "/static/script.js"))
        courses = asyncio.run(self._request("GET", "/api/courses"))

        self.assertEqual(root.status_code, 200)
        self.assertIn("/static/script.js", root.text)
        self.assertIn('id="prefLearningStyle"', root.text)
        self.assertIn('id="recommendBtn"', root.text)
        self.assertEqual(static_script.status_code, 200)
        self.assertIn("fetch('/api/search'", static_script.text)
        self.assertIn("fetch('/api/recommend'", static_script.text)
        self.assertIn("api/reviews/${encodeURIComponent(courseId)}/analysis", static_script.text)
        self.assertIn("Object.entries(data).filter", static_script.text)
        self.assertEqual(courses.status_code, 200)
        self.assertEqual(courses.json()["total"], 29)
        self.assertIn("synthetic_notice", courses.json())

    def test_semantic_search_contract_and_query_validation(self):
        response = asyncio.run(self._request(
            "POST",
            "/api/search",
            json={"query": "learn programming", "top_k": 1},
        ))
        blank = asyncio.run(self._request(
            "POST",
            "/api/search",
            json={"query": "   "},
        ))
        empty = asyncio.run(self._request(
            "POST",
            "/api/search",
            json={"query": ""},
        ))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            set(response.json()),
            {"query", "score_type", "results", "total_found"},
        )
        self.assertEqual(response.json()["total_found"], 1)
        self.assertEqual(response.json()["score_type"], "cosine_similarity")
        self.assertIn("relevance_score", response.json()["results"][0])
        self.assertEqual(blank.status_code, 400)
        self.assertEqual(empty.status_code, 422)

    def test_search_accepts_null_top_k_using_existing_default(self):
        response = asyncio.run(self._request(
            "POST",
            "/api/search",
            json={"query": "learn programming", "top_k": None},
        ))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["total_found"], 2)

    def test_search_and_recommendation_report_missing_model_without_reloading(self):
        backend_app.engine.model = None
        search = asyncio.run(self._request(
            "POST", "/api/search", json={"query": "data"},
        ))
        recommendation = asyncio.run(self._request(
            "POST", "/api/recommend", json={"query": "data"},
        ))

        self.assertEqual(search.status_code, 503)
        self.assertEqual(recommendation.status_code, 503)
        self.assertNotIn("Traceback", search.text)
        self.assertNotIn("Traceback", recommendation.text)

    def test_personalized_recommendation_contract_and_cached_service(self):
        payload = {
            "query": "learn programming",
            "preferences": {"difficulty": "low", "learning_style": "practical"},
            "completed_courses": [],
        }
        first = asyncio.run(self._request("POST", "/api/recommend", json=payload))
        second = asyncio.run(self._request("POST", "/api/recommend", json=payload))

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        body = first.json()
        self.assertEqual(
            set(body),
            {"query", "profile", "ranking_formula", "prerequisite_policy", "data_notice",
             "total_found", "recommendations"},
        )
        self.assertIn("recommendation_score", body["recommendations"][0])
        self.assertIn("components", body["recommendations"][0])
        self.assertGreaterEqual(get_recommendation_service.cache_info().hits, 1)

    def test_recommendation_invalid_values_and_empty_request(self):
        bad_preference = asyncio.run(self._request(
            "POST",
            "/api/recommend",
            json={"preferences": {"difficulty": "extreme"}},
        ))
        empty = asyncio.run(self._request("POST", "/api/recommend", json={}))
        unknown_profile = asyncio.run(self._request(
            "POST",
            "/api/recommend",
            json={"profile_id": "UNKNOWN", "query": "data"},
        ))

        self.assertEqual(bad_preference.status_code, 422)
        self.assertEqual(empty.status_code, 400)
        self.assertEqual(unknown_profile.status_code, 404)
        self.assertNotIn("Traceback", unknown_profile.text)

    def test_review_analysis_and_invalid_course_id(self):
        valid = asyncio.run(self._request("GET", "/api/reviews/CSC101/analysis"))
        missing = asyncio.run(self._request("GET", "/api/reviews/NOT_A_COURSE/analysis"))

        self.assertEqual(valid.status_code, 200)
        self.assertEqual(valid.json()["course_id"], "CSC101")
        self.assertEqual(valid.json()["review_count"], 30)
        self.assertEqual(missing.status_code, 404)

    def test_classical_analysis_retains_flat_response_for_frontend_fallback(self):
        response = asyncio.run(self._request(
            "POST",
            "/api/classical-analysis",
            json={"text": "some sample text"},
        ))

        self.assertEqual(response.status_code, 200)
        self.assertIn("1_preprocessing", response.json())
        self.assertNotIn("analysis", response.json())

    def test_health_and_validation_endpoints(self):
        health = asyncio.run(self._request("GET", "/api/health"))
        validation = asyncio.run(self._request("GET", "/api/validate"))

        self.assertEqual(health.status_code, 200)
        self.assertIn("model_loaded", health.json())
        self.assertTrue(health.json()["dataset_loaded"])
        self.assertEqual(validation.status_code, 200)
        self.assertIn("validation", validation.json())

    def test_unexpected_search_failure_is_logged_and_sanitized(self):
        def fail_search(query, top_k=5):
            if query and top_k:
                raise RuntimeError("private implementation detail")

        backend_app.engine.search = fail_search
        response = asyncio.run(self._request(
            "POST",
            "/api/search",
            json={"query": "data"},
        ))

        self.assertEqual(response.status_code, 500)
        self.assertNotIn("private implementation detail", response.text)
        self.assertEqual(response.json(), {"error": "Search failed. Check server logs."})


if __name__ == "__main__":
    unittest.main()
