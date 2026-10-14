"""Transparent personalized ranking layered over the existing semantic search."""

import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

from data_loader import AGGREGATES_CSV, PROFILES_CSV, load_csv_raw

SYNTHETIC_NOTICE = (
    "Course attributes, student profiles, and review-derived scoring signals are synthetic "
    "demonstration data. Recommendation rankings are not evidence of real-world accuracy."
)

_PROFILE_WEIGHT_FIELDS = {
    "interests": "interest_priority_1_to_5",
    "difficulty": "depth_priority_1_to_5",
    "workload": "low_workload_priority_1_to_5",
    "learning_style": "practical_priority_1_to_5",
    "career_goal": "career_priority_1_to_5",
    "scoring_preference": "scoring_priority_1_to_5",
    "math_complexity": "depth_priority_1_to_5",
    "profile_context": "interest_priority_1_to_5",
}
_LEVELS = {"low": 0.0, "medium": 0.5, "high": 1.0}
_SEPARATOR_RE = re.compile(r"[;,]")
_DEFAULT_WEIGHTS = {
    "semantic_relevance": 0.60,
    "interests": 0.14,
    "career_goal": 0.12,
    "difficulty": 0.10,
    "workload": 0.10,
    "learning_style": 0.12,
    "math_complexity": 0.08,
    "scoring_preference": 0.04,
    "profile_context": 0.10,
}


@dataclass
class RecommendationConfig:
    """Configurable base weights; active signal weights are normalized per request."""

    weights: Dict[str, float] = field(default_factory=lambda: dict(_DEFAULT_WEIGHTS))

    def __post_init__(self):
        for name, weight in self.weights.items():
            if not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
                raise ValueError(f"Recommendation weight for {name} must be a finite nonnegative number.")


def _clean(value) -> str:
    return str(value or "").strip()


def _normalize_label(value) -> Optional[str]:
    normalized = _clean(value).lower()
    return normalized if normalized in _LEVELS else None


def _normalized_cosine(value) -> float:
    try:
        return max(0.0, min(1.0, (float(value) + 1.0) / 2.0))
    except (TypeError, ValueError):
        return 0.5


def _split_prerequisites(value: str) -> List[str]:
    return [item.strip() for item in _SEPARATOR_RE.split(value) if item.strip() and item.strip().lower() != "none"]


class PersonalizedCourseRecommender:
    """Compose semantic relevance with only preferences that are actually supplied."""

    def __init__(
        self,
        semantic_engine,
        courses: List[Dict],
        profiles: Optional[List[Dict]] = None,
        aggregates: Optional[List[Dict]] = None,
        config: Optional[RecommendationConfig] = None,
    ):
        self.semantic_engine = semantic_engine
        self.courses = courses
        self.profiles = profiles if profiles is not None else load_csv_raw(PROFILES_CSV)
        self.aggregates = aggregates if aggregates is not None else load_csv_raw(AGGREGATES_CSV)
        self.config = config or RecommendationConfig()
        self._courses_by_id = {
            _clean(course.get("course_id")): course
            for course in courses if _clean(course.get("course_id"))
        }
        self._profiles_by_id = {
            _clean(profile.get("student_profile_id")): profile
            for profile in self.profiles if _clean(profile.get("student_profile_id"))
        }
        self._aggregate_by_id = {
            _clean(aggregate.get("course_id")): aggregate
            for aggregate in self.aggregates if _clean(aggregate.get("course_id"))
        }
        self._course_aliases = {}
        for course in courses:
            course_id = _clean(course.get("course_id"))
            title = _clean(course.get("title") or course.get("course_title"))
            for alias in (course_id, title):
                if alias:
                    self._course_aliases[alias.casefold()] = course_id

    def _profile_for(self, profile_id: Optional[str]) -> Optional[Dict]:
        if not profile_id:
            return None
        profile = self._profiles_by_id.get(profile_id.strip())
        if profile is None:
            raise ValueError(f"Unknown student profile: {profile_id}")
        return profile

    @staticmethod
    def _profile_preferences(profile: Optional[Dict]) -> Dict[str, str]:
        if not profile:
            return {}
        description = _clean(profile.get("profile_description")).lower()
        derived = {}
        if "lower difficulty" in description or "math avoider" in _clean(profile.get("profile_name")).lower():
            derived["difficulty"] = "low"
        if "lower workload" in description:
            derived["workload"] = "low"
        if "practical" in description or "hands-on" in description or "projects" in description:
            derived["learning_style"] = "practical"
        elif "theoretical" in description or "conceptual depth" in description or "foundations" in description:
            derived["learning_style"] = "theoretical"
        if "scoring potential" in description or "predictable assessment" in description:
            derived["scoring_preference"] = "high"
        if "career" in description or "placement" in description:
            derived["career_goal"] = description
        if "interested in" in description or "prefers management" in description or "data, analytics" in description:
            derived["interests"] = description
        if "lower mathematical complexity" in description:
            derived["math_complexity"] = "low"
        return derived

    def _semantic_scores(self, query: str) -> Dict[str, float]:
        if not query:
            return {}
        results = self.semantic_engine.search(query, top_k=max(len(self.courses), 1))
        return {
            _clean(result.get("course_id")): _normalized_cosine(result.get("relevance_score"))
            for result in results
            if _clean(result.get("course_id"))
        }

    def _prerequisite_status(
        self,
        course: Dict,
        completed_courses: Optional[Sequence[str]],
    ) -> Tuple[str, List[str], List[str]]:
        prerequisites = _split_prerequisites(_clean(course.get("prerequisites")))
        if not prerequisites:
            return "not_required", [], []
        if completed_courses is None:
            return "unknown", [], prerequisites

        completed_ids = set()
        for completed in completed_courses:
            normalized = _clean(completed).casefold()
            if normalized:
                completed_ids.add(self._course_aliases.get(normalized, normalized))

        unmet = []
        unknown = []
        for prerequisite in prerequisites:
            normalized = prerequisite.casefold()
            known_course_id = self._course_aliases.get(normalized)
            if known_course_id is None:
                unknown.append(prerequisite)
            elif known_course_id not in completed_ids:
                unmet.append(prerequisite)
        if unmet:
            return "unmet", unmet, unknown
        if unknown:
            return "unknown", [], unknown
        return "met", [], []

    def _preference_score(self, course: Dict, signal: str, preference: str) -> Optional[float]:
        preference = preference.strip().lower()
        if signal in {"difficulty", "workload", "math_complexity"}:
            target = _normalize_label(preference)
            field_name = {
                "difficulty": "level",
                "workload": "workload_label",
                "math_complexity": "math_complexity",
            }[signal]
            observed = _normalize_label(course.get(field_name))
            if target is None or observed is None:
                return None
            return 1.0 - abs(_LEVELS[target] - _LEVELS[observed])

        if signal == "learning_style":
            emphasis = _clean(course.get("teaching_emphasis")).lower()
            practical = _normalize_label(course.get("practical_relevance"))
            if preference == "practical":
                if emphasis in {"practical", "applied", "case-based"}:
                    return 1.0
                if emphasis in {"balanced", "mixed"}:
                    return 0.5
                return _LEVELS[practical] if practical is not None else None
            if preference == "theoretical":
                if emphasis == "theoretical":
                    return 1.0
                if emphasis in {"balanced", "mixed"}:
                    return 0.5
                if emphasis in {"practical", "applied", "case-based"}:
                    return 0.0
                return None
            return None

        if signal == "scoring_preference":
            if preference not in {"high", "low"}:
                return None
            aggregate = self._aggregate_by_id.get(_clean(course.get("course_id")), {})
            if aggregate.get("aggregate_source") != "synthetic_reviews_only":
                return None
            try:
                if int(aggregate.get("scoring_potential_review_count", 0)) <= 0:
                    return None
            except (TypeError, ValueError):
                return None
            positive_share = aggregate.get("scoring_potential_positive_share")
            try:
                score = float(positive_share)
            except (TypeError, ValueError):
                return None
            score = max(0.0, min(1.0, score))
            return score if preference == "high" else 1.0 - score

        return None

    def recommend(
        self,
        query: Optional[str] = None,
        top_k: int = 5,
        profile_id: Optional[str] = None,
        preferences: Optional[Dict[str, str]] = None,
        completed_courses: Optional[Sequence[str]] = None,
        enforce_prerequisites: bool = False,
    ) -> Dict:
        query = _clean(query)
        explicit = {
            key: _clean(value)
            for key, value in (preferences or {}).items()
            if _clean(value)
        }
        profile = self._profile_for(profile_id)
        inferred = self._profile_preferences(profile)
        merged = dict(inferred)
        merged.update(explicit)
        interests_query = merged.get("interests", "")
        career_query = merged.get("career_goal", "")
        profile_context = (
            " ".join(
                value for value in (
                    _clean(profile.get("profile_name")),
                    _clean(profile.get("profile_description")),
                ) if value
            )
            if profile else ""
        )
        if not any((query, interests_query, career_query, profile_context, merged)):
            raise ValueError("Provide a query, a student profile, or at least one preference.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        semantic_by_course = self._semantic_scores(query)
        signal_scores = {}
        if interests_query:
            signal_scores["interests"] = self._semantic_scores(interests_query)
        if career_query:
            signal_scores["career_goal"] = self._semantic_scores(career_query)
        if profile_context and "interests" not in signal_scores:
            signal_scores["profile_context"] = self._semantic_scores(profile_context)

        active_preference_signals = {}
        for signal in ("difficulty", "workload", "learning_style", "scoring_preference", "math_complexity"):
            if signal in merged:
                active_preference_signals[signal] = merged[signal]
        if set(merged) == {"scoring_preference"} and not (query or interests_query or career_query or profile_context):
            raise ValueError(
                "A scoring preference cannot be the only ranking signal; provide a query, "
                "another preference, or a student profile."
            )

        candidates = self.courses
        if query:
            candidates = [
                self._courses_by_id[course_id]
                for course_id in semantic_by_course
                if course_id in self._courses_by_id
            ]

        profile_priorities = {}
        if profile:
            for signal, field_name in _PROFILE_WEIGHT_FIELDS.items():
                try:
                    profile_priorities[signal] = max(
                        0.0, min(1.0, float(profile.get(field_name, 3)) / 5.0)
                    )
                except (TypeError, ValueError):
                    profile_priorities[signal] = 0.6

        ranked = []
        for course in candidates:
            course_id = _clean(course.get("course_id"))
            prerequisite_status, unmet_prerequisites, unknown_prerequisites = self._prerequisite_status(
                course, completed_courses
            )
            if prerequisite_status == "unmet" and enforce_prerequisites:
                continue

            components = {}
            if query:
                components["semantic_relevance"] = semantic_by_course.get(course_id, 0.5)
            for signal, scores in signal_scores.items():
                if course_id in scores:
                    components[signal] = scores[course_id]
            for signal, preference in active_preference_signals.items():
                score = self._preference_score(course, signal, preference)
                if score is not None:
                    components[signal] = score

            if not components:
                continue

            base_weights = {
                signal: self.config.weights.get(signal, 0.0)
                for signal in components
            }
            if profile:
                for signal in components:
                    if signal in profile_priorities and signal != "semantic_relevance":
                        base_weights[signal] *= profile_priorities[signal]
            scoring_weight = base_weights.pop("scoring_preference", 0.0)
            weight_total = sum(base_weights.values())
            if weight_total <= 0:
                continue
            normalized_weights = {
                signal: weight / weight_total * (1.0 - min(scoring_weight, 0.05))
                for signal, weight in base_weights.items()
            }
            if scoring_weight > 0:
                normalized_weights["scoring_preference"] = min(scoring_weight, 0.05)
            contributions = {
                signal: round(components[signal] * normalized_weights[signal], 6)
                for signal in components
            }
            final_score = sum(contributions.values())

            reasons = []
            tradeoffs = []
            for signal, score in components.items():
                if signal == "semantic_relevance":
                    reasons.append(f"Query relevance {score:.2f}")
                elif signal in {"interests", "career_goal", "profile_context"}:
                    if score >= 0.55:
                        reasons.append(f"Strong {signal.replace('_', ' ')} alignment ({score:.2f})")
                    else:
                        tradeoffs.append(f"Limited {signal.replace('_', ' ')} alignment ({score:.2f})")
                elif score >= 0.75:
                    reasons.append(f"Matches {signal.replace('_', ' ')} preference")
                elif score <= 0.5:
                    tradeoffs.append(f"Trade-off: weaker {signal.replace('_', ' ')} fit")

            if prerequisite_status == "unknown" and completed_courses is None:
                tradeoffs.append("Prerequisites are listed; completion status was not provided")
            elif unknown_prerequisites:
                tradeoffs.append(
                    "Prerequisites could not be verified: " + ", ".join(unknown_prerequisites)
                )
            if prerequisite_status == "unmet":
                tradeoffs.append(
                    "Known unmet prerequisite(s): " + ", ".join(unmet_prerequisites)
                )

            scoring_signal = components.get("scoring_preference")
            if scoring_signal is not None:
                tradeoffs.append(
                    "Scoring fit uses a low-weight synthetic review-label share, not verified outcomes"
                )

            ranked.append({
                "course_id": course_id,
                "title": _clean(course.get("title") or course.get("course_title")),
                "recommendation_score": round(final_score, 4),
                "semantic_relevance": (
                    round((semantic_by_course[course_id] * 2.0) - 1.0, 4)
                    if query and course_id in semantic_by_course else None
                ),
                "preference_match": round(
                    sum(
                        components[key] * normalized_weights[key]
                        for key in components if key != "semantic_relevance"
                    ) / sum(
                        normalized_weights[key]
                        for key in components if key != "semantic_relevance"
                    ),
                    4,
                ) if any(key != "semantic_relevance" for key in components) else None,
                "components": {
                    signal: {
                        "score": round(score, 4),
                        "weight": round(normalized_weights[signal], 4),
                        "contribution": contributions[signal],
                        "source": (
                            "synthetic_review_reference_labels"
                            if signal == "scoring_preference"
                            else "semantic_search"
                            if signal in {"semantic_relevance", "interests", "career_goal", "profile_context"}
                            else "synthetic_course_metadata"
                        ),
                    }
                    for signal, score in components.items()
                },
                "why_recommended": reasons,
                "tradeoffs": tradeoffs,
                "prerequisite_status": prerequisite_status,
            })

        ranked.sort(key=lambda result: (-result["recommendation_score"], result["course_id"]))
        return {
            "query": query or None,
            "profile": (
                {
                    "student_profile_id": _clean(profile.get("student_profile_id")),
                    "profile_name": _clean(profile.get("profile_name")),
                    "is_synthetic": True,
                }
                if profile else None
            ),
            "ranking_formula": {
                "description": (
                    "Sum(component_score × normalized_active_weight); weights are normalized "
                    "over available signals for each course. Semantic cosine scores are mapped "
                    "from [-1, 1] to [0, 1]. Missing preferences are omitted."
                ),
                "configured_base_weights": dict(self.config.weights),
                "weight_adjustment": "Profile priority fields scale matching preference weights.",
                "profile_priority_multipliers": profile_priorities,
                "scoring_preference_max_weight": 0.05,
            },
            "prerequisite_policy": {
                "enforced": enforce_prerequisites,
                "note": (
                    "Known unmet prerequisites are excluded only when enforcement is requested; "
                    "the current product has no mandatory-prerequisite policy."
                ),
            },
            "data_notice": SYNTHETIC_NOTICE,
            "total_found": min(len(ranked), top_k),
            "recommendations": ranked[:top_k],
        }


def recommend_courses(
    semantic_engine,
    courses: List[Dict],
    query: Optional[str] = None,
    top_k: int = 5,
    profile_id: Optional[str] = None,
    preferences: Optional[Dict[str, str]] = None,
    completed_courses: Optional[Sequence[str]] = None,
    enforce_prerequisites: bool = False,
) -> Dict:
    """Build a request-scoped ranker over shared course/profile data."""
    recommender = PersonalizedCourseRecommender(semantic_engine, courses)
    return recommender.recommend(
        query=query,
        top_k=top_k,
        profile_id=profile_id,
        preferences=preferences,
        completed_courses=completed_courses,
        enforce_prerequisites=enforce_prerequisites,
    )


@lru_cache(maxsize=1)
def get_recommendation_service(semantic_engine) -> PersonalizedCourseRecommender:
    """Reuse the process-wide recommender and its loaded profile/aggregate data."""
    from courses_data import get_all_courses

    return PersonalizedCourseRecommender(semantic_engine, get_all_courses())
