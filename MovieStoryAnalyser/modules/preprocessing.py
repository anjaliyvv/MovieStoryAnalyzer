"""
Movie Story Analyser - NLP Text Preprocessing Module
Responsible Team Member: Member 2 (NLP Text Processing & Summarization)
Description:
    Provides systematic multi-stage NLP text transformation pipeline:
    sentence segmentation, lowercasing, punctuation stripping, tokenization,
    stopword filtering, and lemmatization. Exposes intermediate stages for explainability.
"""

import re
import string
from typing import Dict, List, Any

# Curated high-frequency English stopwords (standard NLTK English stopwords subset)
STANDARD_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll",
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's",
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on",
    "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
    "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some",
    "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we",
    "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's",
    "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with",
    "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your",
    "yours", "yourself", "yourselves"
}

# Simple suffix lemmatization dictionary rules for robust offline execution
IRREGULAR_LEMMAS = {
    "ran": "run", "running": "run", "runs": "run",
    "went": "go", "gone": "go", "going": "go", "goes": "go",
    "saw": "see", "seen": "see", "seeing": "see", "sees": "see",
    "took": "take", "taken": "take", "taking": "take", "takes": "take",
    "made": "make", "making": "make", "makes": "make",
    "gave": "give", "given": "give", "giving": "give", "gives": "give",
    "became": "become", "becoming": "become", "becomes": "become",
    "killed": "kill", "killing": "kill", "kills": "kill",
    "fought": "fight", "fighting": "fight", "fights": "fight",
    "loved": "love", "loving": "love", "loves": "love",
    "escaped": "escape", "escaping": "escape", "escapes": "escape",
    "discovered": "discover", "discovering": "discover", "discovers": "discover",
    "saved": "save", "saving": "save", "saves": "save",
    "helped": "help", "helping": "help", "helps": "help",
    "found": "find", "finding": "find", "finds": "find",
    "lived": "live", "living": "live", "lives": "life",
    "died": "die", "dying": "die", "dies": "die"
}


def segment_sentences(text: str) -> List[str]:
    """Splits a narrative plot into individual sentence units."""
    if not text:
        return []
    # Regex split on sentence terminal punctuation followed by space or quote
    raw_sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'‘“])", text.strip())
    cleaned = []
    for s in raw_sentences:
        s_clean = s.strip()
        if len(s_clean) > 0:
            cleaned.append(s_clean)
    return cleaned if cleaned else [text.strip()]


def tokenize_words(text: str) -> List[str]:
    """Extracts alphanumeric word tokens."""
    if not text:
        return []
    return re.findall(r"\b[a-zA-Z0-9']+\b", text)


def simple_lemmatize(word: str) -> str:
    """Lightweight morphological lemmatizer covering irregular forms and plurals/past tense."""
    w = word.lower()
    if w in IRREGULAR_LEMMAS:
        return IRREGULAR_LEMMAS[w]
    # Plural nouns
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("sses") or w.endswith("shes") or w.endswith("ches") or w.endswith("xes"):
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    # Past tense regular verbs
    if w.endswith("ed") and len(w) > 4:
        if w.endswith("ied"):
            return w[:-3] + "y"
        return w[:-2] if not w.endswith("eed") else w[:-1]
    # Present participles
    if w.endswith("ing") and len(w) > 5:
        return w[:-3]
    return w


def preprocess_text(text: str) -> Dict[str, Any]:
    """
    Executes full pipeline and records step-by-step state for transparency.
    Returns:
        sentences: List of raw sentences
        tokens: All word tokens
        cleaned_tokens: Lowercased, punctuation-free tokens
        filtered_tokens: Stopwords removed
        lemmas: Lemmatized tokens
        processed_text_string: Cleaned space-separated string of lemmas
    """
    if not text:
        return {
            "sentences": [],
            "tokens": [],
            "cleaned_tokens": [],
            "filtered_tokens": [],
            "lemmas": [],
            "processed_text_string": "",
            "stats": {"raw_chars": 0, "token_count": 0, "vocab_size": 0}
        }

    sentences = segment_sentences(text)
    tokens = tokenize_words(text)
    cleaned_tokens = [t.lower() for t in tokens if t not in string.punctuation]
    filtered_tokens = [t for t in cleaned_tokens if t not in STANDARD_STOPWORDS and not t.isnumeric()]
    lemmas = [simple_lemmatize(t) for t in filtered_tokens]

    return {
        "raw_text": text,
        "sentences": sentences,
        "tokens": tokens,
        "cleaned_tokens": cleaned_tokens,
        "filtered_tokens": filtered_tokens,
        "lemmas": lemmas,
        "processed_text_string": " ".join(lemmas),
        "stats": {
            "raw_chars": len(text),
            "sentence_count": len(sentences),
            "token_count": len(tokens),
            "filtered_token_count": len(filtered_tokens),
            "vocab_size": len(set(lemmas))
        }
    }
