"""
Movie Story Analyser - Story Summarization Module
Responsible Team Member: Member 2 (NLP Text Processing & Summarization)
Description:
    Graph-based extractive text summarization utilizing TextRank.
    Computes sentence-to-sentence cosine similarity matrices, builds a
    weighted sentence graph with NetworkX, executes PageRank centrality scoring,
    and extracts key storyline sentences based on narrative relevance.
"""

import math
from typing import Dict, List, Tuple, Any
from collections import Counter
from modules.preprocessing import segment_sentences, preprocess_text


def compute_sentence_vector(tokens: List[str], vocab: Dict[str, int]) -> List[float]:
    """Generates normalized term frequency vector over shared vocabulary."""
    counts = Counter(tokens)
    vec = [0.0] * len(vocab)
    for word, idx in vocab.items():
        if word in counts:
            vec[idx] = float(counts[word])
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        return [v / norm for v in vec]
    return vec


def cosine_sim(vec1: List[float], vec2: List[float]) -> float:
    """Computes dot product of normalized vectors."""
    return sum(a * b for a, b in zip(vec1, vec2))


def textrank_summarize(
    text: str,
    top_n: int = 3,
    damping: float = 0.85,
    max_iter: int = 50,
    tol: float = 1e-4
) -> Dict[str, Any]:
    """
    Executes TextRank algorithm on story text.
    Returns:
        short_summary: Top 1-2 sentences
        detailed_summary: Top n sentences in narrative chronological order
        ranked_sentences: List of (rank, score, sentence)
    """
    sentences = segment_sentences(text)
    if not sentences:
        return {
            "short_summary": "",
            "detailed_summary": "",
            "sentence_count": 0,
            "ranked_sentences": [],
            "method": "TextRank Graph Centrality"
        }

    # If storyline has only 1 or 2 sentences, return them directly
    if len(sentences) <= 2:
        return {
            "short_summary": sentences[0],
            "detailed_summary": " ".join(sentences),
            "sentence_count": len(sentences),
            "ranked_sentences": [{"sentence": s, "score": 1.0, "rank": i + 1} for i, s in enumerate(sentences)],
            "method": "Direct Extraction (Brief Narrative)"
        }

    # Tokenize each sentence
    sentence_tokens = []
    vocab_set = set()
    for s in sentences:
        p = preprocess_text(s)
        lemmas = p["lemmas"]
        sentence_tokens.append(lemmas)
        vocab_set.update(lemmas)

    vocab = {w: i for i, w in enumerate(sorted(list(vocab_set)))}
    if not vocab:
        return {
            "short_summary": sentences[0],
            "detailed_summary": " ".join(sentences[:min(len(sentences), top_n)]),
            "sentence_count": len(sentences),
            "ranked_sentences": [{"sentence": s, "score": 1.0, "rank": i + 1} for i, s in enumerate(sentences)],
            "method": "Fallback Chronological"
        }

    # Vectorize sentences
    vectors = [compute_sentence_vector(st, vocab) for st in sentence_tokens]
    n = len(sentences)

    # Build adjacency matrix using cosine similarity
    sim_matrix = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                sim = cosine_sim(vectors[i], vectors[j])
                # Filter weak ties to reduce noise in graph
                sim_matrix[i][j] = sim if sim > 0.05 else 0.0

    # Try using NetworkX if available; otherwise use pure Python PageRank power iteration
    scores = {}
    try:
        import networkx as nx
        nx_graph = nx.Graph()
        for i in range(n):
            nx_graph.add_node(i)
        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i][j] > 0:
                    nx_graph.add_edge(i, j, weight=sim_matrix[i][j])
        scores = nx.pagerank(nx_graph, alpha=damping, max_iter=max_iter, tol=tol)
    except Exception:
        # Pure Python PageRank implementation
        # Initialize uniform scores
        curr_scores = [1.0 / n] * n
        for _ in range(max_iter):
            next_scores = [(1.0 - damping) / n] * n
            for i in range(n):
                for j in range(n):
                    if i != j and sim_matrix[j][i] > 0:
                        out_weight = sum(sim_matrix[j])
                        if out_weight > 0:
                            next_scores[i] += damping * (curr_scores[j] * sim_matrix[j][i] / out_weight)
            # Check convergence
            diff = sum(abs(next_scores[k] - curr_scores[k]) for k in range(n))
            curr_scores = next_scores
            if diff < tol:
                break
        scores = {idx: sc for idx, sc in enumerate(curr_scores)}

    # Rank sentences
    ranked_indices = sorted(scores.keys(), key=lambda idx: scores[idx], reverse=True)
    ranked_records = []
    for rank, idx in enumerate(ranked_indices, start=1):
        ranked_records.append({
            "index": idx,
            "rank": rank,
            "sentence": sentences[idx],
            "score": round(float(scores[idx]), 4)
        })

    # Pick top N sentences, and sort back by chronological order in narrative
    n_selected = min(max(1, top_n), len(sentences))
    chosen_indices = sorted(ranked_indices[:n_selected])
    detailed_summary = " ".join([sentences[i] for i in chosen_indices])

    # Short summary: top 1 or 2 most central sentences
    short_indices = sorted(ranked_indices[:min(2, len(sentences))])
    short_summary = " ".join([sentences[i] for i in short_indices])

    return {
        "short_summary": short_summary,
        "detailed_summary": detailed_summary,
        "sentence_count": len(sentences),
        "selected_count": n_selected,
        "ranked_sentences": ranked_records,
        "method": "TextRank Graph Centrality with PageRank"
    }
