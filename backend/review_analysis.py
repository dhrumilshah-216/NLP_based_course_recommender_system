"""Rule-based analysis and synthetic-reference summaries for course reviews."""

import re
from collections import Counter
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Set

from data_loader import ASPECTS_CSV, REVIEWS_CSV, load_csv_raw

ASPECTS = (
    "difficulty",
    "workload",
    "scoring_potential",
    "teaching_quality",
    "practical_value",
    "career_relevance",
)
SENTIMENTS = ("positive", "negative", "neutral", "mixed")
SYNTHETIC_DATA_NOTICE = (
    "All reviews and labels are synthetic sample data, not actual student experiences. "
    "These summaries are not evidence of real teaching quality, workload, or scoring outcomes."
)
ANALYSIS_METHOD = (
    "Aspect sentiment predictions use transparent keyword/phrase rules over review sentences; "
    "they are not produced by a trained sentiment model."
)
ANALYSIS_LIMITATIONS = [
    "Phrase rules can miss paraphrases, sarcasm, context, and nuanced negation.",
    "A prediction is counted only when a sentence contains a cue for that aspect; "
    "otherwise that review is unassessed for the aspect.",
    "Synthetic reference labels are illustrative, not human-annotated ground truth.",
]

_ASPECT_CUES = {
    "difficulty": (
        "difficulty", "learning curve", "approachable", "manageable", "easy",
        "hard", "difficult", "challenging", "prerequisite", "foundations",
        "abstract", "keep up", "extra resources",
    ),
    "workload": (
        "workload", "assignments", "studied a lot", "busy", "overwhelming",
        "weekend", "quieter weeks", "time commitment",
    ),
    "scoring_potential": (
        "scoring", "scored", "marks", "grading", "grade", "rubric",
        "assessment", "evaluation criteria", "marking criteria",
    ),
    "teaching_quality": (
        "teaching", "classes", "class", "explanations", "explanation",
        "questions were", "instructor", "organized",
    ),
    "practical_value": (
        "hands-on", "practical", "project", "portfolio", "realistic examples",
        "routine exercises", "concrete", "job skill",
    ),
    "career_relevance": (
        "career", "job-relevant", "job relevant", "job-focused", "job focused",
        "placement", "internship", "industry",
    ),
}

_POSITIVE_CUES = {
    "difficulty": (
        "approachable", "reasonable", "manageable", "easy to follow",
        "concepts felt manageable", "kept up each week",
    ),
    "workload": (
        "short enough", "steady rather than overwhelming", "quieter weeks",
        "manageable workload",
    ),
    "scoring_potential": (
        "rubric made it easier", "clear criteria", "criteria were clear",
        "could score well", "preparation translated into marks",
    ),
    "teaching_quality": (
        "organized", "questions were usually addressed", "helped to follow",
        "made difficult ideas easier", "teaching was adequate",
    ),
    "practical_value": (
        "hands-on work helped", "useful", "reuse in a project",
        "something concrete", "portfolio piece",
    ),
    "career_relevance": (
        "job-relevant", "job relevant", "career value", "useful background",
        "direct job skill", "career use",
    ),
}

_NEGATIVE_CUES = {
    "difficulty": (
        "hard to keep up", "difficult", "struggled", "needed extra resources",
        "moved too quickly", "move too quickly", "prerequisites mattered more",
        "took several rounds", "abstract",
    ),
    "workload": (
        "assignments piled up", "studied a lot", "very busy", "overwhelming",
    ),
    "scoring_potential": (
        "grading unpredictable", "scoring seemed to depend", "subjective",
        "not an easy-scoring", "harder than understanding",
    ),
    "teaching_quality": (
        "needed extra resources", "explanations moved too quickly",
        "hard to keep up", "teach myself", "could improve",
        "not especially memorable",
    ),
    "practical_value": (
        "wanted more realistic", "fewer routine exercises", "felt routine",
        "need another project", "more academic than job-focused",
    ),
    "career_relevance": (
        "struggled to see a direct career use", "not job-focused",
        "more academic than job-focused", "depends on the role",
        "rather than a direct job skill",
    ),
}

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_REVIEW_ID = "review_id"
_COURSE_ID = "course_id"


def _contains_phrase(text: str, phrase: str) -> bool:
    pattern = r"(?<!\w)" + re.escape(phrase.lower()) + r"(?!\w)"
    return re.search(pattern, text.lower()) is not None


def _matching_phrases(text: str, phrases: Iterable[str]) -> Set[str]:
    return {phrase for phrase in phrases if _contains_phrase(text, phrase)}


def _predict_aspect_sentiment(text: str, aspect: str) -> Optional[str]:
    """Classify one review/aspect only where its aspect is mentioned."""
    relevant_sentences = [
        sentence.strip()
        for sentence in _SENTENCE_SPLIT.split(text.strip())
        if sentence.strip() and _matching_phrases(sentence, _ASPECT_CUES[aspect])
    ]
    if not relevant_sentences:
        return None

    positive = any(
        _matching_phrases(sentence, _POSITIVE_CUES[aspect])
        for sentence in relevant_sentences
    )
    negative = False
    for sentence in relevant_sentences:
        negative_cues = _matching_phrases(sentence, _NEGATIVE_CUES[aspect])
        if aspect == "workload" and _contains_phrase(sentence, "rather than overwhelming"):
            negative_cues.discard("overwhelming")
        negative = negative or bool(negative_cues)
    if positive and negative:
        return "mixed"
    if positive:
        return "positive"
    if negative:
        return "negative"
    return "neutral"


def _empty_sentiment_counts() -> Dict[str, int]:
    return {sentiment: 0 for sentiment in SENTIMENTS}


def _normalize_sentiment(value) -> Optional[str]:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in SENTIMENTS else None


class CourseReviewAnalyzer:
    """Summarize review text and reference labels without conflating sources."""

    def __init__(self, reviews: List[Dict], aspect_labels: List[Dict]):
        self.reviews = reviews
        self.aspect_labels = aspect_labels

    def _unique_reviews(self):
        unique = {}
        duplicate_counts = Counter()
        for row in self.reviews:
            review_id = str(row.get(_REVIEW_ID) or "").strip()
            course_id = str(row.get(_COURSE_ID) or "").strip()
            if not review_id or not course_id:
                continue
            if review_id in unique:
                duplicate_counts[course_id] += 1
                continue
            unique[review_id] = row
        return unique, duplicate_counts

    def _unique_aspect_labels(self, reviews_by_id):
        unique = {}
        duplicate_counts = Counter()
        unmatched_counts = Counter()
        for row in self.aspect_labels:
            review_id = str(row.get(_REVIEW_ID) or "").strip()
            course_id = str(row.get(_COURSE_ID) or "").strip()
            aspect = str(row.get("aspect") or "").strip().lower()
            key = (review_id, course_id, aspect)
            if not review_id or not course_id or aspect not in ASPECTS:
                continue
            review = reviews_by_id.get(review_id)
            if review is None or str(review.get(_COURSE_ID) or "").strip() != course_id:
                unmatched_counts[course_id] += 1
                continue
            if key in unique:
                duplicate_counts[course_id] += 1
                continue
            unique[key] = row
        return unique, duplicate_counts, unmatched_counts

    def analyze_course(self, course_id: str) -> Dict:
        course_id = str(course_id).strip()
        reviews_by_id, duplicate_reviews = self._unique_reviews()
        labels, duplicate_labels, unmatched_labels = self._unique_aspect_labels(reviews_by_id)
        course_reviews = [
            row for row in reviews_by_id.values()
            if str(row.get(_COURSE_ID) or "").strip() == course_id
        ]
        overall_references = _empty_sentiment_counts()
        empty_text_count = 0
        for review in course_reviews:
            sentiment = _normalize_sentiment(review.get("overall_sentiment"))
            if sentiment:
                overall_references[sentiment] += 1
            if not str(review.get("review_text") or "").strip():
                empty_text_count += 1

        aspect_summaries = {}
        for aspect in ASPECTS:
            prediction_counts = _empty_sentiment_counts()
            reference_counts = _empty_sentiment_counts()
            for review in course_reviews:
                text = str(review.get("review_text") or "").strip()
                if not text:
                    continue
                predicted = _predict_aspect_sentiment(text, aspect)
                if predicted:
                    prediction_counts[predicted] += 1

            labeled_review_ids = set()
            for (review_id, label_course_id, label_aspect), label in labels.items():
                if label_course_id != course_id or label_aspect != aspect:
                    continue
                reference = _normalize_sentiment(label.get("sentiment"))
                if reference:
                    reference_counts[reference] += 1
                    labeled_review_ids.add(review_id)

            aspect_summaries[aspect] = {
                "prediction": {
                    "method": "keyword_phrase_rules",
                    "review_count": sum(prediction_counts.values()),
                    "sentiment_counts": prediction_counts,
                },
                "synthetic_reference_labels": {
                    "source": "review_aspects.csv",
                    "human_annotated": False,
                    "labeled_review_count": len(labeled_review_ids),
                    "sentiment_counts": reference_counts,
                },
            }

        return {
            "course_id": course_id,
            "data_provenance": {
                "type": "synthetic_sample_reviews",
                "notice": SYNTHETIC_DATA_NOTICE,
            },
            "analysis_method": {
                "aspect_prediction": ANALYSIS_METHOD,
                "overall_sentiment": (
                    "Reported separately from aspect predictions as the synthetic "
                    "overall_sentiment reference labels in reviews.csv."
                ),
                "limitations": ANALYSIS_LIMITATIONS,
            },
            "review_count": len(course_reviews),
            "overall_sentiment_reference": {
                "source": "reviews.csv",
                "human_annotated": False,
                "sentiment_counts": overall_references,
            },
            "aspects": aspect_summaries,
            "data_quality": {
                "empty_review_text_count": empty_text_count,
                "duplicate_review_rows_excluded": duplicate_reviews[course_id],
                "duplicate_aspect_label_rows_excluded": duplicate_labels[course_id],
                "unmatched_aspect_label_rows_excluded": unmatched_labels[course_id],
                "reviews_without_text_or_aspect_opinion_are_not_counted_as_predictions": True,
            },
        }


@lru_cache(maxsize=1)
def _load_review_analyzer() -> CourseReviewAnalyzer:
    reviews = load_csv_raw(REVIEWS_CSV)
    aspect_labels = load_csv_raw(ASPECTS_CSV)
    return CourseReviewAnalyzer(reviews, aspect_labels)


def analyze_course_reviews(course_id: str) -> Dict:
    """Analyze the cached dataset loaded once per process."""
    return _load_review_analyzer().analyze_course(course_id)
