"""
Movie Story Analyser - Explainable Sentiment Analysis Module
Responsible Team Member: Member 3 (Sentiment Analysis & Emotion Detection)
Description:
    Rule-based and lexicon-driven sentiment analysis using VADER logic.
    Computes overall compound polarity, positive/negative/neutral proportions,
    sentence-level sentiment distribution, and detailed explainability reports.
    Critically highlights dramatic plot conflict vs audience reception distinction.
"""

import math
import re
from typing import Dict, List, Any, Tuple
from modules.preprocessing import segment_sentences, tokenize_words

# Curated core VADER valence lexicon for deterministic offline execution
VADER_LEXICON = {
    # High negative (conflict, tragedy, horror, crime, loss)
    "kill": -3.0, "killed": -3.2, "murder": -3.5, "death": -3.2, "dead": -3.0, "die": -2.9,
    "terror": -3.4, "terrorist": -3.5, "destroy": -2.8, "destruction": -3.0, "evil": -3.2,
    "war": -2.8, "threat": -2.4, "threaten": -2.6, "danger": -2.4, "dangerous": -2.5,
    "betray": -3.0, "betrayal": -3.2, "revenge": -2.4, "violent": -2.9, "violence": -3.1,
    "disaster": -3.1, "tragedy": -3.2, "tragic": -3.0, "ruin": -2.6, "ruined": -2.8,
    "corrupt": -2.8, "corruption": -2.9, "conspiracy": -2.2, "enemy": -2.3, "enemies": -2.4,
    "hostile": -2.5, "fear": -2.3, "scared": -2.4, "horror": -3.1, "horrific": -3.3,
    "abused": -3.2, "torture": -3.6, "suffer": -2.7, "suffering": -2.9, "pain": -2.3,
    "grief": -2.8, "sorrow": -2.6, "lose": -1.8, "lost": -1.9, "defeat": -2.2,
    "trapped": -2.4, "prison": -2.1, "criminal": -2.6, "crime": -2.5, "dark": -1.4,
    "sinister": -2.7, "nightmare": -2.8, "chaos": -2.5, "devastating": -3.2,
    "monster": -2.6, "demon": -3.0, "catastrophe": -3.4, "doom": -2.8, "fatal": -3.1,

    # High positive (triumph, love, hope, victory, joy)
    "love": 3.2, "loved": 3.1, "loving": 3.2, "hope": 2.5, "hero": 2.7, "heroic": 3.0,
    "victory": 3.1, "win": 2.4, "won": 2.5, "triumph": 3.2, "save": 2.6, "saved": 2.8,
    "savior": 3.0, "protect": 2.2, "protected": 2.3, "peace": 2.8, "peaceful": 2.7,
    "joy": 3.0, "happy": 2.7, "happiness": 3.0, "freedom": 2.9, "free": 2.3,
    "friend": 2.4, "friends": 2.5, "friendship": 2.8, "inspire": 2.7, "inspiration": 2.9,
    "beautiful": 2.8, "wonder": 2.4, "wonderful": 3.0, "miracle": 3.3, "redemption": 2.9,
    "courage": 2.8, "brave": 2.6, "success": 2.7, "successful": 2.8, "master": 2.1,
    "brilliant": 2.9, "glory": 2.8, "champion": 2.9, "reconcile": 2.5, "harmony": 2.6,
    "kind": 2.3, "kindness": 2.7, "cure": 2.5, "rescue": 2.7, "rescued": 2.8,

    # Modifiers
    "not": -0.5, "never": -0.5, "no": -0.4,
    "very": 1.3, "extremely": 1.4, "deeply": 1.2, "incredibly": 1.4
}

BOOSTER_WORDS = {
    "very": 0.293, "extremely": 0.35, "deeply": 0.25, "incredibly": 0.35,
    "totally": 0.25, "absolutely": 0.3, "barely": -0.25, "hardly": -0.25
}

NEGATION_WORDS = {"not", "never", "no", "neither", "hardly", "barely", "scarcely", "without", "doesn't", "don't", "didn't", "won't"}


def analyze_sentence_sentiment(sentence: str) -> Dict[str, Any]:
    """Scores a single sentence, tracking positive/negative valence words."""
    words = tokenize_words(sentence)
    if not words:
        return {
            "sentence": sentence,
            "compound": 0.0,
            "sentiment": "Neutral",
            "pos_words": [],
            "neg_words": []
        }

    valences = []
    pos_words = []
    neg_words = []

    for i, w in enumerate(words):
        w_lower = w.lower()
        if w_lower in VADER_LEXICON:
            val = VADER_LEXICON[w_lower]

            # Check for negation in preceding 3 words
            negated = False
            for j in range(max(0, i - 3), i):
                if words[j].lower() in NEGATION_WORDS:
                    negated = True
                    break

            if negated:
                val = val * -0.74

            # Check for booster word immediately prior
            if i > 0 and words[i - 1].lower() in BOOSTER_WORDS:
                boost = BOOSTER_WORDS[words[i - 1].lower()]
                val = val + (boost if val > 0 else -boost)

            valences.append(val)
            if val > 0.5:
                pos_words.append((w, round(val, 2)))
            elif val < -0.5:
                neg_words.append((w, round(val, 2)))

    if not valences:
        return {
            "sentence": sentence,
            "compound": 0.0,
            "sentiment": "Neutral",
            "pos_words": [],
            "neg_words": []
        }

    # Normalize sum of valences to [-1, 1] using standard VADER formula: sum / sqrt(sum^2 + alpha)
    s = sum(valences)
    alpha = 15.0
    compound = s / math.sqrt((s * s) + alpha)
    compound = round(compound, 4)

    if compound >= 0.15:
        category = "Positive"
    elif compound <= -0.15:
        category = "Negative"
    else:
        category = "Neutral"

    return {
        "sentence": sentence,
        "compound": compound,
        "sentiment": category,
        "pos_words": pos_words,
        "neg_words": neg_words
    }


def analyze_storyline_sentiment(text: str) -> Dict[str, Any]:
    """
    Executes explainable VADER sentiment analysis over full movie storyline.
    Generates detailed breakdown, sentence tallies, and explanatory diagnostic.
    """
    sentences = segment_sentences(text)
    if not sentences:
        return {
            "overall_sentiment": "Neutral",
            "compound_score": 0.0,
            "counts": {"Positive": 0, "Negative": 0, "Neutral": 0, "Total": 0},
            "proportions": {"pos": 0.0, "neg": 0.0, "neu": 1.0},
            "sentences": [],
            "key_neg_sentences": [],
            "key_pos_sentences": [],
            "explanation": "No text provided for analysis."
        }

    # Try using official nltk SentimentIntensityAnalyzer if available
    use_nltk = False
    sia = None
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        use_nltk = True
    except Exception:
        use_nltk = False

    sentence_results = []
    pos_count = 0
    neg_count = 0
    neu_count = 0
    compounds = []
    all_neg_words = []
    all_pos_words = []

    for s in sentences:
        if use_nltk and sia:
            scores = sia.polarity_scores(s)
            c = round(scores["compound"], 4)
            if c >= 0.1:
                cat = "Positive"
                pos_count += 1
            elif c <= -0.1:
                cat = "Negative"
                neg_count += 1
            else:
                cat = "Neutral"
                neu_count += 1

            # Extract words using our fallback lexicon for explainability
            local = analyze_sentence_sentiment(s)
            sentence_results.append({
                "sentence": s,
                "compound": c,
                "sentiment": cat,
                "pos_words": local["pos_words"],
                "neg_words": local["neg_words"]
            })
            all_neg_words.extend(local["neg_words"])
            all_pos_words.extend(local["pos_words"])
            compounds.append(c)
        else:
            res = analyze_sentence_sentiment(s)
            sentence_results.append(res)
            if res["sentiment"] == "Positive":
                pos_count += 1
            elif res["sentiment"] == "Negative":
                neg_count += 1
            else:
                neu_count += 1
            all_neg_words.extend(res["neg_words"])
            all_pos_words.extend(res["pos_words"])
            compounds.append(res["compound"])

    total_sents = len(sentences)
    avg_compound = round(sum(compounds) / total_sents, 4) if total_sents > 0 else 0.0

    # Categorize overall
    if pos_count > 0 and neg_count > 0 and abs(pos_count - neg_count) <= 1 and total_sents >= 3:
        overall = "Mixed"
    elif avg_compound >= 0.12:
        overall = "Positive"
    elif avg_compound <= -0.12:
        overall = "Negative"
    else:
        overall = "Neutral"

    # Identify strongest supporting negative and positive sentences
    sorted_by_compound = sorted(sentence_results, key=lambda x: x["compound"])
    key_neg = [r for r in sorted_by_compound if r["compound"] < -0.1][:3]
    key_pos = [r for r in reversed(sorted_by_compound) if r["compound"] > 0.1][:3]

    # Deduplicate contributing words
    unique_neg_words = list({w: sc for w, sc in all_neg_words}.items())
    unique_pos_words = list({w: sc for w, sc in all_pos_words}.items())

    # Build clear human-readable narrative explanation
    explanation_parts = []
    explanation_parts.append(
        f"The overall storyline sentiment was evaluated as **{overall}** "
        f"with a mean compound score of **{avg_compound:+.2f}** across {total_sents} segmented sentences."
    )

    if overall == "Negative":
        vocab_terms = ", ".join([f"'{w}'" for w, _ in unique_neg_words[:5]]) if unique_neg_words else "conflict and tension markers"
        explanation_parts.append(
            f"The narrative is dominated by adversity, conflict, or threat ({neg_count} negative vs {pos_count} positive sentences). "
            f"Key contributing words include {vocab_terms}. "
            "Note: A negative storyline sentiment indicates intense dramatic stakes, peril, or tragedy within the plot—not a negative audience review or poor film quality."
        )
    elif overall == "Positive":
        vocab_terms = ", ".join([f"'{w}'" for w, _ in unique_pos_words[:5]]) if unique_pos_words else "uplifting vocabulary"
        explanation_parts.append(
            f"The plot emphasizes triumph, discovery, romance, or rescue ({pos_count} positive vs {neg_count} negative sentences). "
            f"Contributing terms include {vocab_terms}."
        )
    elif overall == "Mixed":
        explanation_parts.append(
            f"The storyline features balanced dramatic polarity ({pos_count} positive and {neg_count} negative sentences), "
            "depicting severe conflict followed by redemption, triumph, or ambiguous moral choices."
        )
    else:
        explanation_parts.append(
            f"The synopsis maintains an objective, descriptive narrative tone ({neu_count} neutral sentences), "
            "stating plot premises without heavy emotive or polarized vocabulary."
        )

    full_explanation = " ".join(explanation_parts)

    return {
        "overall_sentiment": overall,
        "compound_score": avg_compound,
        "counts": {
            "Positive": pos_count,
            "Negative": neg_count,
            "Neutral": neu_count,
            "Total": total_sents
        },
        "proportions": {
            "pos": round(pos_count / total_sents, 3) if total_sents > 0 else 0.0,
            "neg": round(neg_count / total_sents, 3) if total_sents > 0 else 0.0,
            "neu": round(neu_count / total_sents, 3) if total_sents > 0 else 0.0,
        },
        "sentences": sentence_results,
        "key_neg_sentences": key_neg,
        "key_pos_sentences": key_pos,
        "contributing_neg_words": unique_neg_words[:8],
        "contributing_pos_words": unique_pos_words[:8],
        "explanation": full_explanation,
        "method": "VADER Lexicon-Rule Model with Sentence Attribution"
    }
