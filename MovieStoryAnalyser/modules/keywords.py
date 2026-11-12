"""
Movie Story Analyser - Keyword Extraction Module
Responsible Team Member: Member 4 (Keywords, Themes & Comparative Analytics)
Description:
    Extracts high-salience unigram and bigram keywords and key phrases using TF-IDF
    (Term Frequency - Inverse Document Frequency).
    Provides relevance scores, frequency statistics, and Plotly visualization builders.
"""

import math
from collections import Counter
from typing import Dict, List, Tuple, Any, Optional
from modules.preprocessing import preprocess_text, STANDARD_STOPWORDS


def extract_ngrams(tokens: List[str], n: int = 2) -> List[str]:
    """Generates contiguous n-grams from a sequence of tokens."""
    if len(tokens) < n:
        return []
    return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def extract_keywords_tfidf(
    text: str,
    corpus: Optional[List[str]] = None,
    top_n: int = 10,
    include_bigrams: bool = True
) -> Dict[str, Any]:
    """
    Computes TF-IDF relevance weights for unigrams and bigrams in a storyline.
    If an external corpus (e.g. all movie overviews) is provided, uses genuine corpus IDF.
    Otherwise, computes sentence-level document frequencies as document sub-units.
    """
    prep = preprocess_text(text)
    lemmas = prep["lemmas"]
    raw_sentences = prep["sentences"]

    if not lemmas:
        return {
            "keywords": [],
            "unigrams": [],
            "bigrams": [],
            "method": "TF-IDF N-gram Extraction"
        }

    # Generate candidate unigrams
    unigrams = [w for w in lemmas if len(w) > 2 and w not in STANDARD_STOPWORDS]

    # Generate candidate bigrams
    bigrams = []
    if include_bigrams:
        # Extract bigrams from raw words to keep grammatical collocations, then clean
        for s in raw_sentences:
            s_prep = preprocess_text(s)
            s_lemmas = [w for w in s_prep["lemmas"] if len(w) > 2]
            bigrams.extend(extract_ngrams(s_lemmas, 2))

    all_candidates = unigrams + bigrams
    if not all_candidates:
        return {
            "keywords": [],
            "unigrams": [],
            "bigrams": [],
            "method": "TF-IDF N-gram Extraction"
        }

    # Calculate Term Frequencies (TF) with sublinear scaling
    counts = Counter(all_candidates)
    max_tf = max(counts.values()) if counts else 1

    # Calculate Inverse Document Frequency (IDF)
    # If corpus given, use corpus doc counts. Else use sentences as document collection.
    doc_collection = corpus if corpus and len(corpus) > 1 else raw_sentences
    total_docs = max(1, len(doc_collection))

    # Pre-tokenize docs for quick matching
    doc_token_sets = []
    for d in doc_collection:
        d_lemmas = set(preprocess_text(d)["lemmas"])
        doc_token_sets.append(d_lemmas)

    tfidf_scores = {}
    for term, count in counts.items():
        # Sublinear TF: 1 + log(tf)
        tf = 1.0 + math.log(count)

        # Count documents containing this term
        parts = term.split()
        if len(parts) == 1:
            df = sum(1 for d_set in doc_token_sets if parts[0] in d_set)
        else:
            df = sum(1 for d_set in doc_token_sets if parts[0] in d_set and parts[1] in d_set)

        # Smooth IDF: log((N + 1) / (df + 1)) + 1
        idf = math.log((total_docs + 1.0) / (df + 1.0)) + 1.0

        # Bigram bonus for narrative richness
        weight_boost = 1.25 if len(parts) > 1 else 1.0
        score = tf * idf * weight_boost

        tfidf_scores[term] = round(score, 4)

    # Sort descending by score
    sorted_terms = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)

    top_keywords = []
    top_unigrams = []
    top_bigrams = []

    seen = set()
    for term, score in sorted_terms:
        if term in seen:
            continue
        seen.add(term)
        is_bigram = " " in term

        record = {
            "keyword": term,
            "type": "Bigram" if is_bigram else "Unigram",
            "score": score,
            "frequency": counts[term]
        }
        top_keywords.append(record)

        if is_bigram and len(top_bigrams) < top_n:
            top_bigrams.append(record)
        elif not is_bigram and len(top_unigrams) < top_n:
            top_unigrams.append(record)

        if len(top_keywords) >= top_n * 2:
            break

    return {
        "keywords": top_keywords[:top_n],
        "unigrams": top_unigrams[:top_n],
        "bigrams": top_bigrams[:top_n],
        "total_extracted": len(top_keywords),
        "method": "Sublinear TF-IDF with Sentence Document Fallback"
    }


def create_keywords_chart(keywords_data: List[Dict[str, Any]], movie_title: str = "Movie"):
    """
    Renders an interactive horizontal bar chart showing top keywords and their TF-IDF relevance.
    """
    try:
        import plotly.graph_objects as go
        from config import COLORS

        if not keywords_data:
            return None

        # Sort ascending for horizontal bar display
        items = list(reversed(keywords_data[:10]))
        terms = [item["keyword"] for item in items]
        scores = [item["score"] for item in items]
        types = [item["type"] for item in items]

        colors = [
            COLORS["accent"] if t == "Bigram" else COLORS["secondary_accent"]
            for t in types
        ]

        fig = go.Figure(data=[
            go.Bar(
                x=scores,
                y=terms,
                orientation="h",
                marker=dict(
                    color=colors,
                    line=dict(color="#28333E", width=1)
                ),
                text=[f"{s:.2f}" for s in scores],
                textposition="outside"
            )
        ])

        fig.update_layout(
            title=f"Top TF-IDF Storyline Keywords: {movie_title}",
            plot_bgcolor=COLORS["card_background"],
            paper_bgcolor=COLORS["secondary_background"],
            font=dict(color=COLORS["main_text"], family="sans-serif"),
            xaxis=dict(
                title="TF-IDF Relevance Score",
                gridcolor="#28333E",
                color=COLORS["main_text"]
            ),
            yaxis=dict(
                color=COLORS["main_text"],
                autorange=True
            ),
            margin=dict(l=100, r=40, t=50, b=40),
            height=380
        )
        return fig
    except Exception:
        return None
