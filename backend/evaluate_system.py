"""Evaluate retrieval and synthetic review analysis against supplied development labels."""

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

BACKEND_DIR = Path(__file__).parent
sys.path.insert(0, str(BACKEND_DIR))

from data_loader import ASPECTS_CSV, QUERIES_CSV, REVIEWS_CSV, load_csv_raw
from ml_engine import _course_search_text, engine
from recommendations import PersonalizedCourseRecommender
from review_analysis import ASPECTS, SENTIMENTS, _normalize_sentiment, _predict_aspect_sentiment


STOP_WORDS = {
    "a", "an", "and", "are", "about", "for", "i", "in", "is", "it", "me",
    "my", "of", "on", "or", "the", "to", "want", "with", "you",
}


def _tokens(text: str) -> Set[str]:
    import re

    return {
        token for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in STOP_WORDS
    }


def _rank_metrics(ranked_ids: Sequence[str], relevant_ids: Set[str], k: int) -> Dict[str, float]:
    top_k = list(ranked_ids[:k])
    hits = sum(course_id in relevant_ids for course_id in top_k)
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, course_id in enumerate(top_k, start=1)
        if course_id in relevant_ids
    )
    ideal_hits = min(k, len(relevant_ids))
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return {
        f"precision@{k}": hits / k,
        f"recall@{k}": hits / len(relevant_ids) if relevant_ids else 0.0,
        f"ndcg@{k}": dcg / ideal_dcg if ideal_dcg else 0.0,
    }


def _mean(rows: Sequence[Dict[str, float]], key: str) -> Optional[float]:
    values = [row[key] for row in rows if key in row]
    return sum(values) / len(values) if values else None


def _keyword_ranking(query: str, courses: Sequence[Dict]) -> List[str]:
    query_terms = _tokens(query)
    scored = []
    for course in courses:
        course_id = str(course.get("course_id") or "")
        terms = _tokens(_course_search_text(course))
        score = len(query_terms & terms) / len(query_terms) if query_terms else 0.0
        scored.append((score, course_id))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [course_id for _, course_id in scored]


def _expected_ids(row: Dict[str, str]) -> Set[str]:
    return {
        item.strip()
        for item in row.get("expected_relevant_course_ids", "").split(";")
        if item.strip()
    }


def _supported_preferences(row: Dict[str, str]) -> Dict[str, str]:
    preferences = {}
    for index in range(1, 5):
        field = (row.get(f"intent_field_{index}") or "").strip().lower()
        value = (row.get(f"intent_value_{index}") or "").strip().lower()
        if field in {"difficulty", "workload", "math_complexity"} and value in {"low", "medium", "high"}:
            preferences[field] = value
        elif field == "scoring_potential" and value in {"low", "high"}:
            preferences["scoring_preference"] = value
        elif field == "teaching_emphasis" and value in {"practical", "theoretical"}:
            preferences["learning_style"] = value
        elif field == "practical_relevance" and value == "high":
            preferences["learning_style"] = "practical"
    return preferences


def evaluate_retrieval(k: int) -> Dict:
    with QUERIES_CSV.open(encoding="utf-8-sig", newline="") as source:
        queries = list(csv.DictReader(source))
    courses = engine.courses
    course_ids = {str(course.get("course_id") or "") for course in courses}
    semantic_metrics = []
    keyword_metrics = []
    personalization_rows = []
    query_details = []

    ranker = PersonalizedCourseRecommender(engine, courses)
    for row in queries:
        query = (row.get("student_query") or "").strip()
        relevant = _expected_ids(row)
        relevant_known = relevant & course_ids
        if not query or not relevant_known:
            continue

        semantic_results = engine.search(query, top_k=len(courses))
        semantic_ids = [str(result.get("course_id")) for result in semantic_results]
        keyword_ids = _keyword_ranking(query, courses)
        sem = _rank_metrics(semantic_ids, relevant_known, k)
        keyword = _rank_metrics(keyword_ids, relevant_known, k)
        semantic_metrics.append(sem)
        keyword_metrics.append(keyword)

        preferences = _supported_preferences(row)
        if preferences:
            personalized = ranker.recommend(
                query=query,
                top_k=len(courses),
                preferences=preferences,
            )["recommendations"]
            personalized_ids = [str(result["course_id"]) for result in personalized]
            semantic_top = set(semantic_ids[:k])
            personalized_top = set(personalized_ids[:k])
            common = set(semantic_ids) & set(personalized_ids)
            rank_sem = {course_id: rank for rank, course_id in enumerate(semantic_ids, start=1)}
            rank_personal = {course_id: rank for rank, course_id in enumerate(personalized_ids, start=1)}
            personalization_rows.append({
                "top1_changed": bool(semantic_ids and personalized_ids and semantic_ids[0] != personalized_ids[0]),
                "topk_overlap": len(semantic_top & personalized_top) / k,
                "mean_absolute_rank_shift": (
                    sum(abs(rank_sem[course_id] - rank_personal[course_id]) for course_id in common) / len(common)
                    if common else 0.0
                ),
            })
        query_details.append({
            "query_id": row.get("query_id"),
            "semantic": sem,
            "keyword": keyword,
            "supported_preferences": preferences,
        })

    return {
        "label_notice": (
            "Expected relevance labels are illustrative synthetic development judgments, "
            "not human-validated ground truth."
        ),
        "query_count": len(query_details),
        "k": k,
        "semantic_search": {
            key: _mean(semantic_metrics, key)
            for key in (f"precision@{k}", f"recall@{k}", f"ndcg@{k}")
        },
        "keyword_baseline": {
            key: _mean(keyword_metrics, key)
            for key in (f"precision@{k}", f"recall@{k}", f"ndcg@{k}")
        },
        "personalization_behavior_not_accuracy": {
            "queries_with_supported_preference_intents": len(personalization_rows),
            "top1_changed_count": sum(row["top1_changed"] for row in personalization_rows),
            "mean_topk_overlap": _mean(personalization_rows, "topk_overlap"),
            "mean_absolute_rank_shift": _mean(personalization_rows, "mean_absolute_rank_shift"),
        },
        "per_query": query_details,
    }


def _macro_f1(expected: Sequence[str], predicted: Sequence[str]) -> Optional[float]:
    if not expected:
        return None
    observed_labels = sorted(set(expected))
    f1_scores = []
    for label in observed_labels:
        true_positive = sum(actual == label and guess == label for actual, guess in zip(expected, predicted))
        false_positive = sum(actual != label and guess == label for actual, guess in zip(expected, predicted))
        false_negative = sum(actual == label and guess != label for actual, guess in zip(expected, predicted))
        denominator = 2 * true_positive + false_positive + false_negative
        f1_scores.append(2 * true_positive / denominator if denominator else 0.0)
    return sum(f1_scores) / len(f1_scores)


def evaluate_review_aspects() -> Dict:
    reviews = load_csv_raw(REVIEWS_CSV)
    aspect_labels = load_csv_raw(ASPECTS_CSV)
    reviews_by_id = {}
    for review in reviews:
        review_id = str(review.get("review_id") or "").strip()
        course_id = str(review.get("course_id") or "").strip()
        if review_id and course_id and review_id not in reviews_by_id:
            reviews_by_id[review_id] = review

    unique_labels = {}
    for row in aspect_labels:
        review_id = str(row.get("review_id") or "").strip()
        course_id = str(row.get("course_id") or "").strip()
        aspect = str(row.get("aspect") or "").strip().lower()
        sentiment = _normalize_sentiment(row.get("sentiment"))
        review = reviews_by_id.get(review_id)
        key = (review_id, course_id, aspect)
        if (
            review_id and course_id and aspect in ASPECTS and sentiment
            and review and str(review.get("course_id") or "").strip() == course_id
        ):
            unique_labels.setdefault(key, (sentiment, review))

    expected = []
    predicted_covered = []
    aspect_coverage = {aspect: {"labeled": 0, "predicted": 0} for aspect in ASPECTS}
    for (_, _, aspect), (sentiment, review) in unique_labels.items():
        aspect_coverage[aspect]["labeled"] += 1
        prediction = _predict_aspect_sentiment(str(review.get("review_text") or ""), aspect)
        if prediction in SENTIMENTS:
            expected.append(sentiment)
            predicted_covered.append(prediction)
            aspect_coverage[aspect]["predicted"] += 1

    accuracy = (
        sum(actual == guess for actual, guess in zip(expected, predicted_covered)) / len(expected)
        if expected else None
    )
    return {
        "label_notice": (
            "Aspect reference labels are synthetic and not human-annotated. "
            "Overall sentiment labels are not evaluated as aspect predictions."
        ),
        "method": "Compare rule-based aspect predictions with matching review_id/course_id/aspect labels; omit unlabeled or unpredicted pairs.",
        "unique_labeled_review_aspect_pairs": len(unique_labels),
        "prediction_coverage": len(expected) / len(unique_labels) if unique_labels else None,
        "accuracy_on_covered_pairs": accuracy,
        "macro_f1_on_covered_pairs": _macro_f1(expected, predicted_covered),
        "covered_prediction_count": len(expected),
        "coverage_by_aspect": {
            aspect: {
                **counts,
                "prediction_coverage": counts["predicted"] / counts["labeled"] if counts["labeled"] else None,
            }
            for aspect, counts in aspect_coverage.items()
        },
        "limitations": [
            "Illustrative labels are not ground truth and overlap with synthetic review-generation rules.",
            "Coverage excludes reviews with no aspect prediction, so covered-pair scores may be optimistic.",
            "No independently verified student feedback or outcome labels are available.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=5, help="Retrieval cutoff for Precision/Recall/nDCG")
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k must be at least 1")

    engine.load_or_train_model()
    report = {
        "synthetic_data": True,
        "retrieval": evaluate_retrieval(args.k),
        "review_aspect_sentiment": evaluate_review_aspects(),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
