"""
FastAPI Backend - Ultimate Safe Mode Version
"""

import os
import sys
import json
import logging
from pathlib import Path
from contextlib import asynccontextmanager

BACKEND_DIR = Path(__file__).parent
sys.path.insert(0, str(BACKEND_DIR))

from fastapi import FastAPI,HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import uvicorn

# Try to import engine
try:
    from ml_engine import engine
    ENGINE_LOADED = True
    ENGINE_ERROR = None
except Exception:
    logging.getLogger(__name__).exception("Core course-search engine failed to import")
    ENGINE_LOADED = False
    ENGINE_ERROR = "Model dependencies failed to load; check server logs."

try:
    from classical_nlp_engine import preprocess_text, pos_tagging, extract_entities, tfidf_cosine
    NLP_LOADED = True
except Exception:
    logging.getLogger(__name__).exception("Classical NLP dependencies failed to load")
    NLP_LOADED = False

# Phase 1 — Dataset integration helpers
try:
    from courses_data import get_all_courses, _COURSES_SOURCE
    from validate_dataset import validate_all as _validate_all
    DATASET_LOADED = True
except Exception:
    logging.getLogger(__name__).exception("Course dataset helpers failed to import")
    DATASET_LOADED = False
    _COURSES_SOURCE = "unavailable"

try:
    from review_analysis import analyze_course_reviews
    REVIEW_ANALYSIS_LOADED = True
except Exception:
    logging.getLogger(__name__).exception("Course review analysis failed to import")
    REVIEW_ANALYSIS_LOADED = False

try:
    from recommendations import get_recommendation_service
    RECOMMENDATION_LOADED = True
except Exception:
    logging.getLogger(__name__).exception("Personalized recommendations failed to import")
    RECOMMENDATION_LOADED = False

@asynccontextmanager
async def lifespan(_app: FastAPI):
    global ENGINE_ERROR
    print("\n" + "=" * 60)
    print(" Starting Hybrid NLP Course Recommendation System")
    print("=" * 60)
    if ENGINE_LOADED and engine.model is None:
        try:
            engine.load_or_train_model()
            ENGINE_ERROR = None
        except Exception:
            ENGINE_ERROR = "Model initialization failed; check server logs."
            logging.getLogger(__name__).exception("Semantic search initialization failed")
    if DATASET_LOADED and RECOMMENDATION_LOADED:
        try:
            get_recommendation_service(engine)
        except Exception:
            logging.getLogger(__name__).exception("Personalized recommendation data failed to initialize")
    yield

app = FastAPI(title="Hybrid NLP Course Recommender", version="2.0.0", lifespan=lifespan)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

FRONTEND_DIR = BACKEND_DIR.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: Optional[int] = Field(default=5, ge=1, le=20)

class ClassicalRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)

class RecommendationPreferences(BaseModel):
    interests: Optional[str] = Field(default=None, max_length=500)
    career_goal: Optional[str] = Field(default=None, max_length=500)
    difficulty: Optional[str] = Field(default=None, pattern="^(low|medium|high)$")
    workload: Optional[str] = Field(default=None, pattern="^(low|medium|high)$")
    learning_style: Optional[str] = Field(default=None, pattern="^(practical|theoretical)$")
    math_complexity: Optional[str] = Field(default=None, pattern="^(low|medium|high)$")
    scoring_preference: Optional[str] = Field(default=None, pattern="^(high|low)$")

class RecommendationRequest(BaseModel):
    query: Optional[str] = Field(default=None, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    profile_id: Optional[str] = Field(default=None, max_length=32)
    preferences: RecommendationPreferences = Field(default_factory=RecommendationPreferences)
    completed_courses: Optional[List[str]] = Field(default=None, max_length=100)
    enforce_prerequisites: bool = False

@app.get("/")
async def root():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "API Running"}

@app.post("/api/search")
async def search_courses(request: SearchRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    if not ENGINE_LOADED:
        return JSONResponse(
            status_code=503,
            content={"error": f"AI model unavailable: {ENGINE_ERROR or 'Model dependencies failed to load; check server logs.'}"},
        )
    if engine.model is None:
        return JSONResponse(
            status_code=503,
            content={"error": f"AI model unavailable: {ENGINE_ERROR or 'Model is not initialized; check server logs.'}"},
        )

    try:
        results = engine.search(request.query.strip(), top_k=request.top_k or 5)
        return {
            "query": request.query,
            "score_type": "cosine_similarity",
            "results": results,
            "total_found": len(results),
        }
    except Exception:
        logging.getLogger(__name__).exception("Course search processing failed")
        return JSONResponse(status_code=500, content={"error": "Search failed. Check server logs."})

@app.post("/api/recommend")
async def personalized_recommendations(request: RecommendationRequest):
    if not DATASET_LOADED or not RECOMMENDATION_LOADED:
        return JSONResponse(
            status_code=503,
            content={"error": "Course and student profile data unavailable. Check server logs."},
        )
    if not ENGINE_LOADED or engine.model is None:
        return JSONResponse(
            status_code=503,
            content={"error": f"AI model unavailable: {ENGINE_ERROR or 'Model initialization failed; check server logs.'}"},
        )

    try:
        supplied_preferences = {
            key: value
            for key, value in request.preferences.model_dump(exclude_none=True).items()
            if value.strip()
        }
        service = get_recommendation_service(engine)
        return service.recommend(
            query=request.query,
            top_k=request.top_k,
            profile_id=request.profile_id,
            preferences=supplied_preferences,
            completed_courses=request.completed_courses,
            enforce_prerequisites=request.enforce_prerequisites,
        )
    except ValueError as error:
        status_code = 404 if str(error).startswith("Unknown student profile:") else 400
        return JSONResponse(status_code=status_code, content={"error": str(error)})
    except Exception:
        logging.getLogger(__name__).exception("Personalized course recommendation failed")
        return JSONResponse(
            status_code=500,
            content={"error": "Recommendation failed. Check server logs."},
        )
    
@app.post("/api/classical-analysis")
async def classical_analysis(request: ClassicalRequest):
    text = request.text.strip()
    
    if not NLP_LOADED:
        return JSONResponse(status_code=500, content={"error": "Classical NLP module failed to load."})

    try:
        results = {
            "1_preprocessing": preprocess_text(text),
            "6_pos_tagging": pos_tagging(text),
            "8_ner": extract_entities(text),
            "9_tfidf_cosine": tfidf_cosine([text, "This is a demo course about AI and machine learning."])
        }
        
        # THE MAGIC FIX: Force standard JSON serialization. 
        # If ANY object is weird, it converts it to a string instead of crashing.
        safe_json_str = json.dumps(results, default=str)
        return json.loads(safe_json_str)
        
    except Exception:
        logging.getLogger(__name__).exception("Classical NLP analysis failed")
        return JSONResponse(
            status_code=500, 
            content={
                "status": "error",
                "message": "Analysis failed. Check server logs.",
                "traceback": "Error details are recorded in server logs.",
            }
        )

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": ENGINE_LOADED and engine.model is not None,
        "model_error": ENGINE_ERROR,
        "nlp_loaded": NLP_LOADED,
        "dataset_loaded": DATASET_LOADED,
        "courses_source": _COURSES_SOURCE,
    }


@app.get("/api/courses")
async def list_courses():
    """
    Return the full course catalogue.
    Reflects whichever source is active (CSV or hardcoded fallback).
    """
    if not DATASET_LOADED:
        return JSONResponse(
            status_code=503,
            content={"error": "Course catalogue unavailable. Check server logs."}
        )
    try:
        courses = get_all_courses()
        return {
            "source": _COURSES_SOURCE,
            "total": len(courses),
            "is_synthetic": True,
            "synthetic_notice": (
                "All course data originates from the 'My College' synthetic dataset "
                "and is fictional. It must not be presented as real institutional data."
            ),
            "courses": courses,
        }
    except Exception:
        logging.getLogger(__name__).exception("Course catalogue request failed")
        return JSONResponse(
            status_code=500, content={"error": "Failed to retrieve courses. Check server logs."}
        )


@app.get("/api/reviews/{course_id}/analysis")
async def course_review_analysis(course_id: str):
    """Return synthetic-reference and rule-based sentiment summaries for a course."""
    if not DATASET_LOADED or not REVIEW_ANALYSIS_LOADED:
        return JSONResponse(
            status_code=503,
            content={"error": "Review analysis unavailable. Check server logs."}
        )
    try:
        if not any(course.get("course_id") == course_id for course in get_all_courses()):
            raise HTTPException(status_code=404, detail="Course not found")
        return analyze_course_reviews(course_id)
    except HTTPException:
        raise
    except Exception:
        logging.getLogger(__name__).exception("Course review analysis failed")
        return JSONResponse(
            status_code=500, content={"error": "Review analysis failed. Check server logs."}
        )


@app.get("/api/validate")
async def validate_dataset():
    """
    Run full dataset validation checks and return the report.
    Checks referential integrity, required fields, valid categorical values,
    rating ranges, and provenance flags across all 6 dataset files.
    """
    if not DATASET_LOADED:
        return JSONResponse(
            status_code=503,
            content={"error": "Validation module unavailable. Check server logs."}
        )
    try:
        report = _validate_all()
        return {
            "validation": report.to_dict(),
            "synthetic_notice": (
                "Validation targets the 'My College' synthetic dataset. "
                "All data is fictional and intended for prototype testing only."
            ),
        }
    except Exception:
        logging.getLogger(__name__).exception("Dataset validation request failed")
        return JSONResponse(
            status_code=500, content={"error": "Validation failed. Check server logs."}
        )

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        reload=False,
    )