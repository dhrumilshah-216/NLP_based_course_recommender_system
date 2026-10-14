"""
NLP Machine Learning Engine
Implements cached bi-encoder semantic course search.
"""

import hashlib
import json
import logging
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


sys.path.insert(0, str(Path(__file__).parent))
from courses_data import COURSES_DATABASE

BASE_MODEL_NAME = os.environ.get("BASE_MODEL_NAME", "all-MiniLM-L6-v2")
_PROJECT_DIR = Path(__file__).parent.parent
CUSTOM_MODEL_PATH = Path(
    os.environ.get("MODEL_PATH", str(_PROJECT_DIR / "models" / "trained_course_model"))
)
if not CUSTOM_MODEL_PATH.is_absolute():
    CUSTOM_MODEL_PATH = (_PROJECT_DIR / CUSTOM_MODEL_PATH).resolve()
EMBEDDINGS_CACHE_PATH = Path(__file__).parent / "checkpoints" / "course_embeddings.json"
logger = logging.getLogger(__name__)

_SEARCH_FIELDS: Tuple[Tuple[str, str], ...] = (
    ("title", "Course title"),
    ("syllabus_topics", "Topics"),
    ("prerequisites", "Prerequisites"),
    ("department", "Department"),
    ("topic_cluster", "Subject area"),
    ("teaching_emphasis", "Teaching emphasis"),
    ("tools_technologies", "Tools and technologies"),
    ("practical_relevance", "Practical relevance"),
    ("career_relevance", "Career relevance"),
    ("math_complexity", "Math complexity"),
    ("workload_label", "Workload"),
    ("level", "Level"),
    ("tags", "Tags"),
    ("description", "Description"),
)
_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)

TRAINING_DATA = [
    ("I want to build the robot for my senior capstone",
     "Build and program physical robots using ROS and C++. Implement SLAM, path planning, and sensor fusion for autonomous navigation."),
    ("How do I make a website that looks like Netflix?",
     "Master React, state management with Redux, and building complex streaming UIs like Netflix. Learn performance optimization."),
    ("I want to predict stock market crashes using data",
     "Apply LSTMs, ARIMA models, and reinforcement learning to financial market data. Build trading bots."),
    ("How do I hack into systems to test their security?",
     "Learn Kali Linux, network vulnerabilities, exploit development, and security auditing."),
    ("I want to make video game characters that actually think",
     "Implement pathfinding, behavior trees, procedural generation, and reinforcement learning in Unity."),
    ("How do I make an app that tracks my calories?",
     "Build native mobile applications using SwiftUI and CoreData. Learn app architecture and UI/UX design."),
    ("I want to build an AI assistant that reads my emails",
     "Build intelligent agents that can hold context-aware dialogues with users using speech-to-text, NLP, and large language models."),
    ("I need to learn the math behind neural networks",
     "Study vectors, matrices, eigenvalues, and singular value decomposition required for understanding machine learning algorithms."),
    ("I want to understand how ChatGPT works",
     "Master how to build models that understand, interpret, and generate human language using BERT, GPT, and modern Transformer architectures."),
    ("I want to analyze customer data for my business",
     "Use data to drive business decisions, A/B testing, customer segmentation, and ROI analysis. Learn SQL, Python pandas."),
]


def _field_text(value) -> str:
    """Convert a course field to clean searchable text."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return "; ".join(str(item).strip() for item in value if item is not None and str(item).strip())
    return str(value).strip()


def _course_search_fields(course: Dict) -> List[Tuple[str, str, str]]:
    fields = []
    for key, label in _SEARCH_FIELDS:
        value = _field_text(course.get(key))
        if value and value.lower() not in {"none", "n/a", "na"}:
            fields.append((key, label, value))
    return fields


def _course_search_text(course: Dict) -> str:
    return " ".join(f"{label}: {value}" for _, label, value in _course_search_fields(course))


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = sum(float(value) ** 2 for value in left) ** 0.5
    right_norm = sum(float(value) ** 2 for value in right) ** 0.5
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _as_vector_list(encoded) -> List[List[float]]:
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], (int, float)):
        encoded = [encoded]
    return [[float(value) for value in vector] for vector in encoded]


class CourseRecommenderEngine:
    """Sentence-Transformer search over the active course catalogue."""

    def __init__(
        self,
        courses: Optional[List[Dict]] = None,
        cache_path: Optional[Path] = None,
    ):
        self.model = None
        self.course_embeddings: List[List[float]] = []
        self.courses = COURSES_DATABASE if courses is None else courses
        self._indexed_course_indices: List[int] = []
        self._initialized = False
        self.cache_path = Path(cache_path) if cache_path else EMBEDDINGS_CACHE_PATH
        self._model_fingerprint = str(CUSTOM_MODEL_PATH.resolve())

    def load_or_train_model(self):
        """Load the existing local fine-tuned model without downloading or training."""
        if self._initialized:
            return
        if not CUSTOM_MODEL_PATH.is_dir() or not any(CUSTOM_MODEL_PATH.iterdir()):
            raise FileNotFoundError(
                f"Local Sentence-Transformer model not found at {CUSTOM_MODEL_PATH}. "
                "Place the trained model there or run model training explicitly."
            )

        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            str(CUSTOM_MODEL_PATH),
            local_files_only=True,
        )
        self._model_fingerprint = self._fingerprint_model_files()
        self.reindex_courses()
        self._initialized = True

    def _fingerprint_model_files(self) -> str:
        digest = hashlib.sha256(BASE_MODEL_NAME.encode("utf-8"))
        for path in sorted(CUSTOM_MODEL_PATH.rglob("*")):
            if not path.is_file():
                continue
            relative_path = str(path.relative_to(CUSTOM_MODEL_PATH))
            stat = path.stat()
            digest.update(f"{relative_path}:{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8"))
        return digest.hexdigest()

    def _index_signature(self, indexed: List[Tuple[int, Dict, str]]) -> str:
        payload = {
            "model": self._model_fingerprint,
            "courses": [
                [str(course.get("course_id", index)), text]
                for index, course, text in indexed
            ],
        }
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _load_cached_embeddings(self, signature: str) -> Optional[List[List[float]]]:
        if not self.cache_path.is_file():
            return None
        try:
            with self.cache_path.open(encoding="utf-8") as cache_file:
                cached = json.load(cache_file)
            if cached.get("signature") != signature:
                return None
            vectors = _as_vector_list(cached["embeddings"])
            if len(vectors) != len(self._indexed_course_indices):
                return None
            if vectors and any(len(vector) != len(vectors[0]) for vector in vectors):
                return None
            return vectors
        except (OSError, ValueError, TypeError, KeyError) as error:
            logger.warning("Ignoring invalid course embedding cache %s: %s", self.cache_path, error)
            return None

    def _save_cached_embeddings(self, signature: str, embeddings: List[List[float]]) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.cache_path.parent,
                prefix=self.cache_path.name + ".",
                suffix=".tmp",
                delete=False,
            ) as cache_file:
                temporary_path = Path(cache_file.name)
                json.dump(
                    {"signature": signature, "embeddings": embeddings},
                    cache_file,
                    separators=(",", ":"),
                )
            os.replace(temporary_path, self.cache_path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    def train_model(self, epochs: int = 15):
        """Explicitly fine-tune the base model; startup never downloads or trains."""
        from sentence_transformers import InputExample, SentenceTransformer, losses
        from torch.utils.data import DataLoader

        base_model = SentenceTransformer(BASE_MODEL_NAME)
        examples = [InputExample(texts=[query, description]) for query, description in TRAINING_DATA]
        data_loader = DataLoader(examples, shuffle=True, batch_size=2)
        train_loss = losses.MultipleNegativesRankingLoss(base_model)
        base_model.fit(
            train_objectives=[(data_loader, train_loss)],
            epochs=epochs,
            warmup_steps=10,
            output_path=str(CUSTOM_MODEL_PATH),
            show_progress_bar=True,
        )
        self.model = SentenceTransformer(
            str(CUSTOM_MODEL_PATH),
            local_files_only=True,
        )
        self._model_fingerprint = self._fingerprint_model_files()
        self.reindex_courses()
        self._initialized = True

    def reindex_courses(self):
        """Load valid cached course vectors or encode the current catalogue once."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_or_train_model() first.")

        indexed = [
            (index, course, _course_search_text(course))
            for index, course in enumerate(self.courses)
        ]
        indexed = [item for item in indexed if item[2]]
        self._indexed_course_indices = [index for index, _, _ in indexed]
        if not indexed:
            self.course_embeddings = []
            return

        signature = self._index_signature(indexed)
        cached_vectors = self._load_cached_embeddings(signature)
        if cached_vectors is not None:
            self.course_embeddings = cached_vectors
            return

        texts = [text for _, _, text in indexed]
        encoded = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        vectors = _as_vector_list(encoded)
        if len(vectors) != len(indexed):
            raise ValueError(
                f"Model returned {len(vectors)} course embeddings for {len(indexed)} courses."
            )
        if vectors and any(len(vector) != len(vectors[0]) for vector in vectors):
            raise ValueError("Model returned course embeddings with inconsistent dimensions.")

        self.course_embeddings = vectors
        self._save_cached_embeddings(signature, vectors)

    @staticmethod
    def _explain_match(query: str, course: Dict) -> List[str]:
        query_terms = set(_TOKEN_RE.findall(query.lower()))
        explanations = []
        for _, label, value in _course_search_fields(course):
            field_terms = set(_TOKEN_RE.findall(f"{label} {value}".lower()))
            if query_terms.intersection(field_terms):
                explanations.append(f"{label}: {value}")
            if len(explanations) == 3:
                break
        return explanations

    def search(self, query: str, top_k: int = 5):
        """Rank courses by raw cosine similarity, not a probability or preference score."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query cannot be empty.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if self.model is None:
            raise ValueError("Model not loaded. Call load_or_train_model() first.")
        if not self._indexed_course_indices:
            return []

        encoded_query = self.model.encode(query.strip(), convert_to_numpy=True)
        query_vector = _as_vector_list(encoded_query)[0]
        scored_courses = [
            (
                _cosine_similarity(query_vector, embedding),
                self._indexed_course_indices[index],
            )
            for index, embedding in enumerate(self.course_embeddings)
        ]
        scored_courses.sort(key=lambda result: (-result[0], result[1]))

        results = []
        for score, course_index in scored_courses[: min(top_k, len(scored_courses))]:
            course = self.courses[course_index].copy()
            course["relevance_score"] = round(score, 4)
            # Retain the prior field for API clients; it is a scaled cosine value, not a probability.
            course["similarity_score"] = round(score * 100, 2)
            course["match_explanation"] = self._explain_match(query, course)
            results.append(course)
        return results


engine = CourseRecommenderEngine()
