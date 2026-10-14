"""
Classical NLP Engine - 100% JSON Serializable Safe Version
"""

import nltk
import spacy
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK data quietly
for pkg in ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']:
    try:
        nltk.download(pkg, quiet=True)
    except:
        pass

# Load SpaCy model
try:
    nlp_spacy = spacy.load("en_core_web_sm")
except OSError:
    nlp_spacy = None

def safe_serialize(obj):
    """Recursively convert numpy types and tuples to native Python lists/dicts."""
    if isinstance(obj, dict):
        return {k: safe_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [safe_serialize(v) for v in obj]
    elif isinstance(obj, tuple):
        return [safe_serialize(v) for v in obj] # Convert tuple to list for JSON
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return str(obj) # Fallback to string for anything else

def preprocess_text(text):
    try:
        stop_words = set(stopwords.words('english'))
        stemmer = PorterStemmer()
        lemmatizer = WordNetLemmatizer()
        
        tokens = nltk.word_tokenize(text.lower())
        tokens_no_stop = [w for w in tokens if w.isalnum() and w not in stop_words]
        
        return safe_serialize({
            "nltk_tokens": tokens_no_stop[:20],
            "stemmed": [stemmer.stem(w) for w in tokens_no_stop[:20]],
            "lemmatized": [lemmatizer.lemmatize(w) for w in tokens_no_stop[:20]]
        })
    except Exception as e:
        return {"error": str(e)}

def pos_tagging(text):
    try:
        tokens = nltk.word_tokenize(text)
        tags = nltk.pos_tag(tokens)
        # Convert tuples to lists to be perfectly safe for JSON
        return safe_serialize({"pos_tags": [[t[0], t[1]] for t in tags[:15]]})
    except Exception as e:
        return {"error": str(e)}

def extract_entities(text):
    try:
        if nlp_spacy is None:
            return {"error": "SpaCy not loaded"}
        doc = nlp_spacy(text)
        entities = [{"text": str(ent.text), "label": str(ent.label_)} for ent in doc.ents]
        return safe_serialize({"entities": entities if entities else [{"note": "No entities found"}]})
    except Exception as e:
        return {"error": str(e)}

def tfidf_cosine(texts):
    try:
        tfidf = TfidfVectorizer()
        matrix = tfidf.fit_transform(texts)
        sim = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
        
        # Ensure features is a native Python list of strings
        features = list(tfidf.get_feature_names_out())[:10]
        # Ensure similarities is a native Python list of floats
        similarities = [round(float(x), 4) for x in sim]
        
        return safe_serialize({
            "features": features,
            "cosine_similarities": similarities
        })
    except Exception as e:
        return {"error": str(e)}