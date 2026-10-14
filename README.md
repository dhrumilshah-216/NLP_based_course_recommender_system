# 🎓 AI Course Recommendation System

A full-stack, production-grade course recommendation engine powered by **semantic NLP** and **contrastive learning** using a fine-tuned Transformer model.

## 🌟 Overview

This system solves the "vocabulary mismatch" problem in course discovery by understanding user intent and context rather than relying on exact keyword matching.

**Example:** A query like *"I want to build an AI assistant that reads emails"* can rank *"Conversational AI & Chatbot Development"* highly despite little keyword overlap. Semantic relevance scores are cosine similarities, not confidence probabilities.

## 🛠️ Tech Stack

- **Backend:** FastAPI + Uvicorn
- **NLP/AI:** Sentence-Transformers + PyTorch
- **Frontend:** Vanilla HTML/CSS/JS with glassmorphism design
- **Architecture:** Bi-Encoder semantic search with contrastive learning

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Application

```bash
cd backend
python app.py
```

### 3. Open Your Browser

- **Frontend:** http://localhost:8000
- **API Docs (Swagger):** http://localhost:8000/docs

At startup, the backend loads the existing local fine-tuned Sentence-Transformer checkpoint.
Course embeddings are persisted in `backend/checkpoints/course_embeddings.json` and are
reused until the model or indexed course text changes. Startup does not download a model
or train automatically; restore the checkpoint or explicitly run model training if it is
missing.

The required runtime assets are the course/review CSV files under
`my_college_nlp_dataset/`, the local model checkpoint under `models/trained_course_model/`,
and the packages listed in `requirements.txt`. The checkpoint is present in the project,
but semantic search and personalized semantic ranking require the Sentence-Transformers
and PyTorch packages to be installed. No API keys are required.

Optional settings are documented in `.env.example`: `HOST` (default `0.0.0.0`), `PORT`
(default `8000`), `BASE_MODEL_NAME` (default `all-MiniLM-L6-v2`), and `MODEL_PATH`
(default `./models/trained_course_model`, resolved from the project root). The app does
not load `.env` files automatically; export these variables in the shell before starting
the backend if overriding the defaults.

## 📁 Project Structure

```
course_recommender/
├── backend/
│   ├── app.py              # FastAPI server
│   ├── ml_engine.py        # NLP pipeline
│   └── courses_data.py     # Course database
├── frontend/
│   ├── index.html          # UI
│   ├── style.css           # Glassmorphism styling
│   └── script.js           # Frontend logic
├── models/                 # Trained model (auto-created)
├── requirements.txt
├── .env.example
└── README.md
```

## 🔬 How It Works

1. **User Query** → Natural language describing a learning goal
2. **Semantic Encoding** → Fine-tuned Transformer converts the query and course fields
   (title, syllabus, prerequisites, department, tools, and related metadata) to vectors
3. **Vector Search** → Cosine similarity computed against cached course vectors
4. **Ranking** → Top-K courses returned with raw cosine relevance scores and matching
   course details; scores are not probabilities or personalized preference scores

## 📚 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/search` | POST | Semantic course search |
| `/api/recommend` | POST | Personalized recommendations using semantic relevance and supplied preferences |
| `/api/courses` | GET | List all courses |
| `/api/reviews/{course_id}/analysis` | GET | Synthetic review reference and rule-based aspect sentiment summaries |
| `/api/health` | GET | System health check |

`POST /api/search` accepts `{"query": "...", "top_k": 5}` and returns
`query`, `score_type`, `results`, and `total_found`. `POST /api/recommend` accepts the
optional query/profile/preferences fields documented below. `GET /api/courses` retains
the existing catalogue response. Request validation failures return FastAPI `422`
responses; unknown course/profile IDs return `404`, empty recommendation requests return
`400`, and unavailable models or datasets return `503`. Unexpected processing failures
are logged server-side and return a generic error without a traceback.

### Synthetic Review Analysis

`GET /api/reviews/CSC101/analysis` returns separate overall sentiment reference-label
counts from `reviews.csv`, rule-based aspect predictions from review text, and synthetic
aspect reference-label counts from `review_aspects.csv`. The aspect predictions are
keyword/phrase heuristics, not a trained model; reviews without an aspect cue are not
counted for that aspect. All dataset reviews and labels are synthetic, not actual student
experiences or evidence of teaching quality or scoring outcomes. Synthetic rating fields
are not used in these summaries.

The existing semantic-search screen also offers preference fields and a personalized
recommendation action. It displays each API-provided ranking score, component contribution,
reason, trade-off, and available course metadata. The review-insights disclosure on a result
loads the existing course review endpoint on demand. Review insights are labeled as
synthetic sample data; aspect predictions use keyword/phrase rules, while reference labels
are shown separately.

### Personalized Recommendations

`POST /api/recommend` accepts an optional `query`, `profile_id`, preference object
(`interests`, `career_goal`, `difficulty`, `workload`, `learning_style`,
`math_complexity`, and `scoring_preference`), optional `completed_courses`, and `top_k`.
The response keeps raw semantic relevance, preference fit, component contributions,
reasons, trade-offs, and prerequisite status separate. Active component weights are
normalized for each course and returned with the ranking formula; omitted preferences
do not count against a course. Prerequisite status is reported when completed courses are
supplied. Since the existing product has no mandatory-prerequisite policy, known unmet
requirements are disclosed as trade-offs by default and excluded only when the caller
explicitly sets `enforce_prerequisites` to `true`. Unmapped prerequisite names remain
eligible with an explicit verification trade-off.

Scoring preference matching is optional and capped at 5% of the final weight. It uses only
the synthetic `scoring_potential_positive_share` where the aggregate identifies its
`synthetic_reviews_only` source and has a nonzero sample count. It cannot be the only
ranking signal and is not a verified scoring outcome.

### Evaluation

With the project dependencies and local model checkpoint available, run
`python backend/evaluate_system.py --k 5` from the project root. It compares the semantic
ranker with a deterministic lexical-overlap baseline on the supplied
`queries_and_relevance.csv` labels using Precision@K, Recall@K, and binary nDCG@K. The
report also gives rank-change diagnostics for supported preference intents separately;
these describe ranking behavior, not recommendation accuracy. Aspect-review sentiment is
compared only against matching synthetic `review_aspects.csv` labels and reports coverage,
accuracy, and macro-F1 on covered review/aspect pairs. The supplied relevance and aspect
labels are synthetic and illustrative, not human-validated ground truth; the reported
figures must not be interpreted as real-world system validation.

## 🧪 Demo Scenarios

Try these queries to see semantic matching in action:

- *"I want to build an AI assistant that reads emails"* → Conversational AI
- *"How do I make a mobile app for my startup?"* → iOS Development
- *"I need to learn the math behind neural networks"* → Linear Algebra for ML
- *"I want to predict stock market trends"* → Algorithmic Trading

## 📖 Research References

1. Reimers & Gurevych (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*
2. Reimers & Gurevych (2020). *Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation*

## 📝 License

MIT License