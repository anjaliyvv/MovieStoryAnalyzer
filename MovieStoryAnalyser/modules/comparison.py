"""
Movie Story Analyser - Dual Movie Comparison Module
Responsible Team Member: Member 4 (Keywords, Themes & Comparative Analytics)
Description:
    Computes comparative NLP metrics between any two selected movie storylines:
    TF-IDF Vector Space Cosine Similarity, shared vs unique keyword overlap,
    sentiment differential, emotional convergence, and narrative thematic alignment.
"""

import math
from typing import Dict, List, Any, Set
from modules.preprocessing import preprocess_text
from modules.summarizer import textrank_summarize
from modules.sentiment import analyze_storyline_sentiment
from modules.emotion import detect_emotions
from modules.keywords import extract_keywords_tfidf
from modules.themes import detect_themes


def compute_tfidf_cosine_similarity(text1: str, text2: str) -> float:
    """Calculates cosine similarity between two storyline text vectors."""
    p1 = preprocess_text(text1)
    p2 = preprocess_text(text2)

    lemmas1 = p1["lemmas"]
    lemmas2 = p2["lemmas"]

    if not lemmas1 or not lemmas2:
        return 0.0

    # Build joint vocabulary
    joint_vocab = sorted(list(set(lemmas1).union(set(lemmas2))))
    v_map = {w: i for i, w in enumerate(joint_vocab)}

    # Document frequencies across the 2 documents
    d1_set = set(lemmas1)
    d2_set = set(lemmas2)

    # Compute TF-IDF vectors
    vec1 = [0.0] * len(joint_vocab)
    vec2 = [0.0] * len(joint_vocab)

    for w in lemmas1:
        idx = v_map[w]
        # tf * idf
        tf = 1.0 + math.log(lemmas1.count(w))
        df = (1 if w in d1_set else 0) + (1 if w in d2_set else 0)
        idf = math.log((2.0 + 1.0) / (df + 1.0)) + 1.0
        vec1[idx] = tf * idf

    for w in lemmas2:
        idx = v_map[w]
        tf = 1.0 + math.log(lemmas2.count(w))
        df = (1 if w in d1_set else 0) + (1 if w in d2_set else 0)
        idf = math.log((2.0 + 1.0) / (df + 1.0)) + 1.0
        vec2[idx] = tf * idf

    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    similarity = dot / (norm1 * norm2)
    return max(0.0, min(1.0, float(similarity)))


def compare_two_movies(movie_a: Dict[str, Any], movie_b: Dict[str, Any]) -> Dict[str, Any]:
    """
    Performs end-to-end comparative analysis between Movie A and Movie B.
    All metrics and comparison tables are dynamically computed from actual storylines.
    """
    story_a = movie_a.get("storyline", "")
    story_b = movie_b.get("storyline", "")

    # 1. Textual Cosine Similarity
    sim_score = compute_tfidf_cosine_similarity(story_a, story_b)
    sim_percent = round(sim_score * 100, 1)

    # 2. Summaries
    sum_a = textrank_summarize(story_a, top_n=2)
    sum_b = textrank_summarize(story_b, top_n=2)

    # 3. Sentiment
    sent_a = analyze_storyline_sentiment(story_a)
    sent_b = analyze_storyline_sentiment(story_b)

    # 4. Emotions
    emo_a = detect_emotions(story_a)
    emo_b = detect_emotions(story_b)

    # 5. Keywords
    kw_a = extract_keywords_tfidf(story_a, top_n=12)
    kw_b = extract_keywords_tfidf(story_b, top_n=12)

    terms_a = [k["keyword"] for k in kw_a["keywords"]]
    terms_b = [k["keyword"] for k in kw_b["keywords"]]

    set_a = set(terms_a)
    set_b = set(terms_b)

    shared_keywords = sorted(list(set_a.intersection(set_b)))
    unique_to_a = sorted(list(set_a - set_b))
    unique_to_b = sorted(list(set_b - set_a))

    # 6. Themes
    thm_a = detect_themes(story_a)
    thm_b = detect_themes(story_b)

    active_themes_a = {t["theme"] for t in thm_a["themes"]}
    active_themes_b = {t["theme"] for t in thm_b["themes"]}

    shared_themes = sorted(list(active_themes_a.intersection(active_themes_b)))
    unique_themes_a = sorted(list(active_themes_a - active_themes_b))
    unique_themes_b = sorted(list(active_themes_b - active_themes_a))

    # 7. Word count & narrative density
    words_a = len(story_a.split())
    words_b = len(story_b.split())

    return {
        "similarity_score": round(sim_score, 4),
        "similarity_percentage": sim_percent,
        "movie_a": {
            "title": movie_a.get("title", "Movie A"),
            "year": movie_a.get("year", "N/A"),
            "genres": movie_a.get("genres", []),
            "word_count": words_a,
            "summary": sum_a["short_summary"],
            "sentiment": sent_a["overall_sentiment"],
            "compound": sent_a["compound_score"],
            "dominant_emotion": emo_a["dominant_emotion"],
            "dominant_theme": thm_a["dominant_theme"],
            "top_keywords": terms_a[:8],
            "emotions": emo_a["percentages"],
            "theme_names": [t["theme"] for t in thm_a["themes"][:4]]
        },
        "movie_b": {
            "title": movie_b.get("title", "Movie B"),
            "year": movie_b.get("year", "N/A"),
            "genres": movie_b.get("genres", []),
            "word_count": words_b,
            "summary": sum_b["short_summary"],
            "sentiment": sent_b["overall_sentiment"],
            "compound": sent_b["compound_score"],
            "dominant_emotion": emo_b["dominant_emotion"],
            "dominant_theme": thm_b["dominant_theme"],
            "top_keywords": terms_b[:8],
            "emotions": emo_b["percentages"],
            "theme_names": [t["theme"] for t in thm_b["themes"][:4]]
        },
        "shared_keywords": shared_keywords,
        "unique_keywords_a": unique_to_a,
        "unique_keywords_b": unique_to_b,
        "shared_themes": shared_themes,
        "unique_themes_a": unique_themes_a,
        "unique_themes_b": unique_themes_b
    }
