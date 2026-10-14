"""
Course Database Module
----------------------
Primary source : my_college_nlp_dataset/courses.csv  (29 synthetic courses)
Fallback source: hardcoded list below                (12 demo courses)

On import, the module attempts to load courses from the CSV dataset so that
ml_engine.py gets a richer, more realistic catalogue for semantic search.
If the CSV is absent or unreadable for any reason, COURSES_DATABASE silently
falls back to the original 12-course hardcoded list — no existing API endpoint
or frontend feature is broken.

Note: all CSV data is 100 % synthetic ("My College" fictional institution).
"""

COURSES_DATABASE = [
    {
        "course_id": "CS101",
        "title": "Introduction to Python Programming",
        "description": "Learn the fundamentals of Python including variables, data types, control flow, functions, and basic object-oriented programming. Perfect for absolute beginners.",
        "instructor": "Dr. Sarah Chen",
        "credits": 3,
        "level": "Beginner",
        "tags": ["python", "programming", "basics"]
    },
    {
        "course_id": "CS204",
        "title": "Natural Language Processing with Transformers",
        "description": "Master how to build models that understand, interpret, and generate human language using BERT, GPT, and modern Transformer architectures. Covers tokenization, embeddings, attention mechanisms, and fine-tuning.",
        "instructor": "Dr. Michael Rodriguez",
        "credits": 4,
        "level": "Advanced",
        "tags": ["nlp", "transformers", "bert", "gpt", "ai"]
    },
    {
        "course_id": "CS305",
        "title": "Advanced Deep Learning & Neural Networks",
        "description": "Deep dive into backpropagation, CNNs, RNNs, LSTMs, and PyTorch for complex pattern recognition. Build image classifiers, sequence models, and generative networks.",
        "instructor": "Dr. Aisha Patel",
        "credits": 4,
        "level": "Advanced",
        "tags": ["deep learning", "neural networks", "pytorch", "cnn", "rnn"]
    },
    {
        "course_id": "BUS110",
        "title": "Marketing Analytics & Data-Driven Decisions",
        "description": "Use data to drive business decisions, A/B testing, customer segmentation, and ROI analysis. Learn SQL, Python pandas, and visualization with Tableau.",
        "instructor": "Prof. James Wilson",
        "credits": 3,
        "level": "Intermediate",
        "tags": ["marketing", "analytics", "business", "data"]
    },
    {
        "course_id": "CS401",
        "title": "Conversational AI & Chatbot Development",
        "description": "Build intelligent agents that can hold context-aware dialogues with users using speech-to-text, NLP, and large language models. Deploy chatbots for customer service and virtual assistants.",
        "instructor": "Dr. Emily Zhang",
        "credits": 4,
        "level": "Advanced",
        "tags": ["chatbot", "conversational ai", "dialogue", "llm"]
    },
    {
        "course_id": "MATH202",
        "title": "Linear Algebra for Machine Learning",
        "description": "Study vectors, matrices, eigenvalues, and singular value decomposition required for understanding machine learning algorithms and neural network mathematics.",
        "instructor": "Dr. Robert Kim",
        "credits": 3,
        "level": "Intermediate",
        "tags": ["math", "linear algebra", "machine learning"]
    },
    {
        "course_id": "CS490",
        "title": "Autonomous Robotics Lab",
        "description": "Build and program physical robots using ROS (Robot Operating System) and C++. Implement SLAM, path planning, and sensor fusion for autonomous navigation.",
        "instructor": "Dr. Lisa Thompson",
        "credits": 4,
        "level": "Advanced",
        "tags": ["robotics", "ros", "autonomous", "hardware"]
    },
    {
        "course_id": "CS310",
        "title": "Advanced Frontend Engineering",
        "description": "Master React, state management with Redux, and building complex streaming UIs like Netflix. Learn performance optimization, testing, and modern JavaScript.",
        "instructor": "Prof. David Park",
        "credits": 3,
        "level": "Intermediate",
        "tags": ["frontend", "react", "javascript", "web"]
    },
    {
        "course_id": "FIN405",
        "title": "Algorithmic Trading & Time Series Analysis",
        "description": "Apply LSTMs, ARIMA models, and reinforcement learning to financial market data. Build trading bots and analyze stock market patterns.",
        "instructor": "Dr. Jennifer Lee",
        "credits": 4,
        "level": "Advanced",
        "tags": ["finance", "trading", "time series", "lstm"]
    },
    {
        "course_id": "CYB300",
        "title": "Ethical Hacking & Penetration Testing",
        "description": "Learn Kali Linux, network vulnerabilities, exploit development, and security auditing. Practice on real-world scenarios in a controlled lab environment.",
        "instructor": "Prof. Marcus Johnson",
        "credits": 4,
        "level": "Advanced",
        "tags": ["cybersecurity", "hacking", "penetration testing", "security"]
    },
    {
        "course_id": "CS420",
        "title": "AI for Game Development",
        "description": "Implement pathfinding, behavior trees, procedural generation, and reinforcement learning in Unity. Create game characters that make intelligent decisions.",
        "instructor": "Dr. Kevin Brown",
        "credits": 4,
        "level": "Advanced",
        "tags": ["game dev", "ai", "unity", "reinforcement learning"]
    },
    {
        "course_id": "MOB200",
        "title": "iOS Development with Swift",
        "description": "Build native mobile applications using SwiftUI and CoreData. Learn app architecture, UI/UX design, and App Store deployment.",
        "instructor": "Prof. Amanda Garcia",
        "credits": 3,
        "level": "Intermediate",
        "tags": ["mobile", "ios", "swift", "app development"]
    }
]

# ---------------------------------------------------------------------------
# CSV-first loading  (Phase 1 — Dataset Integration)
# ---------------------------------------------------------------------------
# Try to replace the hardcoded list with the richer 29-course CSV catalogue.
# Any import or parsing failure is caught and the hardcoded list is kept,
# so no existing feature is broken.
_COURSES_SOURCE = "hardcoded"   # tracks which source is active (used by /api/health)
try:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).parent))
    from data_loader import load_courses_from_csv as _load_csv
    _csv_courses = _load_csv()
    if _csv_courses:
        COURSES_DATABASE = _csv_courses
        _COURSES_SOURCE  = "csv"
except Exception as _e:
    # Intentionally silent — hardcoded fallback stays in place
    _COURSES_SOURCE = f"hardcoded (csv load failed: {_e})"


def get_all_courses():
    """Return all courses from the active database source."""
    return COURSES_DATABASE


def get_course_by_id(course_id):
    """Return a single course by ID, or None if not found."""
    for course in COURSES_DATABASE:
        if course["course_id"] == course_id:
            return course
    return None