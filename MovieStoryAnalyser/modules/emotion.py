"""
Movie Story Analyser - Emotion Detection Module
Responsible Team Member: Member 3 (Sentiment Analysis & Emotion Detection)
Description:
    Transparent, explainable emotion scoring across 7 core narrative dimensions:
    Happiness, Sadness, Fear, Anger, Surprise, Love, Hope.
    Utilizes an empirical emotion lexicon with contextual intensity modifiers.
    Provides data structures and Plotly visualization builders.
"""

from typing import Dict, List, Any, Tuple
from modules.preprocessing import tokenize_words, segment_sentences
from config import EMOTION_LIST

# Curated, transparent lexicon mapping words and lemmas to narrative emotions
EMOTION_LEXICON: Dict[str, Dict[str, float]] = {
    # Happiness
    "joy": {"Happiness": 1.0}, "celebrate": {"Happiness": 0.8}, "laugh": {"Happiness": 0.9},
    "cheerful": {"Happiness": 0.9}, "delight": {"Happiness": 0.9}, "smile": {"Happiness": 0.7},
    "fun": {"Happiness": 0.8}, "wonderful": {"Happiness": 0.8}, "hilarious": {"Happiness": 1.0},
    "comedy": {"Happiness": 0.8}, "triumph": {"Happiness": 0.8, "Hope": 0.7},
    "glee": {"Happiness": 0.9}, "festive": {"Happiness": 0.8}, "ecstatic": {"Happiness": 1.0},

    # Sadness
    "grief": {"Sadness": 1.0}, "sorrow": {"Sadness": 0.9}, "mourn": {"Sadness": 0.9},
    "loss": {"Sadness": 0.8}, "tear": {"Sadness": 0.8}, "weep": {"Sadness": 0.9},
    "tragedy": {"Sadness": 0.9, "Fear": 0.5}, "heartbroken": {"Sadness": 1.0},
    "depressed": {"Sadness": 0.9}, "lonely": {"Sadness": 0.8}, "despair": {"Sadness": 0.9, "Fear": 0.6},
    "death": {"Sadness": 0.8, "Fear": 0.6}, "died": {"Sadness": 0.8}, "crying": {"Sadness": 0.9},

    # Fear
    "terror": {"Fear": 1.0}, "horror": {"Fear": 1.0}, "panic": {"Fear": 0.9},
    "dread": {"Fear": 0.9}, "nightmare": {"Fear": 0.8}, "danger": {"Fear": 0.7},
    "threat": {"Fear": 0.7, "Anger": 0.4}, "frightened": {"Fear": 0.9}, "scared": {"Fear": 0.9},
    "monster": {"Fear": 0.7}, "creepy": {"Fear": 0.7}, "peril": {"Fear": 0.8},
    "trapped": {"Fear": 0.8}, "stalked": {"Fear": 0.8}, "sinister": {"Fear": 0.7},

    # Anger
    "fury": {"Anger": 1.0}, "rage": {"Anger": 1.0}, "wrath": {"Anger": 1.0},
    "revenge": {"Anger": 0.9}, "hate": {"Anger": 0.9}, "hatred": {"Anger": 1.0},
    "hostile": {"Anger": 0.8}, "destroy": {"Anger": 0.7}, "violent": {"Anger": 0.8},
    "attack": {"Anger": 0.7}, "fight": {"Anger": 0.6}, "betrayal": {"Anger": 0.8, "Sadness": 0.6},
    "cruel": {"Anger": 0.8}, "vicious": {"Anger": 0.8}, "resentment": {"Anger": 0.8},

    # Surprise
    "shock": {"Surprise": 0.9}, "astonish": {"Surprise": 0.9}, "amaze": {"Surprise": 0.8},
    "unexpected": {"Surprise": 0.8}, "twist": {"Surprise": 0.8}, "revelation": {"Surprise": 0.8},
    "stumble": {"Surprise": 0.5}, "mystery": {"Surprise": 0.6}, "sudden": {"Surprise": 0.7},
    "uncover": {"Surprise": 0.6}, "startling": {"Surprise": 0.9}, "curious": {"Surprise": 0.6},
    "secret": {"Surprise": 0.7}, "bizarre": {"Surprise": 0.8},

    # Love
    "romance": {"Love": 1.0}, "romantic": {"Love": 1.0}, "passion": {"Love": 0.8},
    "adore": {"Love": 0.9}, "affection": {"Love": 0.8}, "beloved": {"Love": 0.9},
    "kiss": {"Love": 0.8}, "marry": {"Love": 0.7}, "marriage": {"Love": 0.7},
    "devotion": {"Love": 0.8, "Hope": 0.5}, "cherish": {"Love": 0.8}, "crush": {"Love": 0.7},
    "sweetheart": {"Love": 0.9}, "soulmate": {"Love": 1.0}, "intimacy": {"Love": 0.8},
    "love": {"Love": 1.0},

    # Hope
    "hope": {"Hope": 1.0}, "dream": {"Hope": 0.7}, "aspire": {"Hope": 0.8},
    "faith": {"Hope": 0.8}, "cure": {"Hope": 0.8}, "rescue": {"Hope": 0.7},
    "freedom": {"Hope": 0.8}, "future": {"Hope": 0.6}, "believe": {"Hope": 0.7},
    "redemption": {"Hope": 0.9}, "miracle": {"Hope": 0.9, "Surprise": 0.6},
    "overcome": {"Hope": 0.8}, "survive": {"Hope": 0.8}, "inspire": {"Hope": 0.8},
    "light": {"Hope": 0.6}, "peace": {"Hope": 0.8, "Happiness": 0.5}
}


def detect_emotions(text: str) -> Dict[str, Any]:
    """
    Computes distribution over 7 emotions, tracking matching terms for full transparency.
    """
    words = tokenize_words(text)
    if not words:
        return {
            "distribution": {e: 0.0 for e in EMOTION_LIST},
            "percentages": {e: round(100.0 / len(EMOTION_LIST), 1) for e in EMOTION_LIST},
            "dominant_emotion": "Neutral",
            "secondary_emotion": "None",
            "evidence": {e: [] for e in EMOTION_LIST},
            "method": "Transparent Lexicon Emotion Detection"
        }

    raw_scores = {e: 0.0 for e in EMOTION_LIST}
    evidence = {e: [] for e in EMOTION_LIST}

    for w in words:
        w_lower = w.lower()
        if w_lower in EMOTION_LEXICON:
            mappings = EMOTION_LEXICON[w_lower]
            for emo, weight in mappings.items():
                raw_scores[emo] += weight
                evidence[emo].append((w_lower, weight))

    total_score = sum(raw_scores.values())

    # If no emotional lexicon hit, distribute baseline evenly
    if total_score == 0:
        norm_scores = {e: 0.0 for e in EMOTION_LIST}
        percentages = {e: round(100.0 / len(EMOTION_LIST), 1) for e in EMOTION_LIST}
        dominant = "Objective / Neutral"
        secondary = "None"
    else:
        norm_scores = {e: round(score / total_score, 4) for e, score in raw_scores.items()}
        percentages = {e: round((score / total_score) * 100, 1) for e, score in raw_scores.items()}

        sorted_emotions = sorted(raw_scores.items(), key=lambda x: x[1], reverse=True)
        dominant = sorted_emotions[0][0] if sorted_emotions[0][1] > 0 else "Neutral"
        secondary = sorted_emotions[1][0] if len(sorted_emotions) > 1 and sorted_emotions[1][1] > 0 else "None"

    # Clean up evidence (unique keywords with frequencies)
    clean_evidence = {}
    for emo, items in evidence.items():
        counts = {}
        for term, _ in items:
            counts[term] = counts.get(term, 0) + 1
        clean_evidence[emo] = sorted(counts.items(), key=lambda x: x[1], reverse=True)

    return {
        "raw_scores": raw_scores,
        "distribution": norm_scores,
        "percentages": percentages,
        "dominant_emotion": dominant,
        "secondary_emotion": secondary,
        "evidence": clean_evidence,
        "total_emotion_markers": int(sum(len(v) for v in evidence.values())),
        "method": "Transparent Emotion Lexicon with Lexical Grounding"
    }


def create_emotion_chart(emotion_data: Dict[str, Any], movie_title: str = "Movie"):
    """
    Builds an interactive Plotly bar chart displaying the emotion profile.
    Falls back gracefully if Plotly is not installed.
    """
    try:
        import plotly.graph_objects as go
        from config import COLORS

        percentages = emotion_data["percentages"]
        emotions = list(percentages.keys())
        values = list(percentages.values())

        # Custom distinct aesthetic palette matching the design guidelines
        bar_colors = [
            "#E0A96D",  # Happiness (Warm Gold)
            "#6C7A89",  # Sadness (Slate Muted Blue)
            "#8E44AD",  # Fear (Moody Violet)
            "#C0392B",  # Anger (Brick Crimson)
            "#F39C12",  # Surprise (Amber)
            "#D87093",  # Love (Pale Rose Violet)
            "#27AE60"   # Hope (Emerald Sage)
        ]

        fig = go.Figure(data=[
            go.Bar(
                x=emotions,
                y=values,
                text=[f"{v:.1f}%" for v in values],
                textposition="auto",
                marker=dict(
                    color=bar_colors,
                    line=dict(color="#28333E", width=1.5)
                )
            )
        ])

        fig.update_layout(
            title=f"Emotion Distribution: {movie_title}",
            plot_bgcolor=COLORS["card_background"],
            paper_bgcolor=COLORS["secondary_background"],
            font=dict(color=COLORS["main_text"], family="sans-serif"),
            xaxis=dict(
                title="Emotion Dimension",
                gridcolor="#28333E",
                color=COLORS["main_text"]
            ),
            yaxis=dict(
                title="Relative Percentage (%)",
                gridcolor="#28333E",
                color=COLORS["main_text"],
                range=[0, max(values) + 10] if values else [0, 100]
            ),
            margin=dict(l=40, r=40, t=50, b=40),
            height=380
        )
        return fig
    except Exception as e:
        return None
