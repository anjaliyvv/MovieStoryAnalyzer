"""
Movie Story Analyser - Narrative Theme Detection Module
Responsible Team Member: Member 4 (Keywords, Themes & Comparative Analytics)
Description:
    Identifies high-level narrative motifs across 10 canonical storytelling themes:
    Friendship, Love, Revenge, Survival, Family, Mystery, Adventure, Betrayal, Sacrifice, Social issues.
    Employs rule-grounded semantic clustering with sentence attribution for transparent evidence.
"""

from typing import Dict, List, Any, Tuple
from modules.preprocessing import segment_sentences, tokenize_words
from config import THEME_LIST

# Semantic vocabulary clusters representing the 10 narrative themes
THEME_LEXICON: Dict[str, Dict[str, Any]] = {
    "Friendship": {
        "keywords": {"friend", "friends", "friendship", "companion", "buddy", "ally", "allies", "bond", "partnership", "loyalty", "brotherhood", "team", "together"},
        "description": "Comradeship, mutual loyalty, and emotional bonds between non-romantic partners."
    },
    "Love": {
        "keywords": {"love", "romance", "romantic", "lover", "passion", "beloved", "marry", "marriage", "crush", "kiss", "affection", "heart", "couple", "soulmate"},
        "description": "Romantic attraction, interpersonal devotion, and courtship."
    },
    "Revenge": {
        "keywords": {"revenge", "avenge", "vengeance", "retaliation", "vendetta", "payback", "grudge", "retribution", "settle", "score", "wrath", "destroy"},
        "description": "Retribution for past wrongs, grievances, or personal vendettas."
    },
    "Survival": {
        "keywords": {"survive", "survival", "survivor", "endure", "alive", "peril", "escape", "stranded", "wilderness", "apocalypse", "starvation", "outlive", "trapped", "danger"},
        "description": "Physical endurance against mortal peril, isolation, hostile forces, or catastrophe."
    },
    "Family": {
        "keywords": {"family", "father", "mother", "son", "daughter", "brother", "sister", "parent", "child", "children", "ancestor", "heritage", "lineage", "home", "bloodline"},
        "description": "Kinship, generational legacy, familial duties, and parental sacrifices."
    },
    "Mystery": {
        "keywords": {"mystery", "secret", "investigate", "detective", "clue", "hidden", "puzzle", "unravel", "truth", "conspiracy", "unknown", "disappearance", "curious", "enigma"},
        "description": "Investigating the unknown, unraveling hidden secrets, and solving enigmas."
    },
    "Adventure": {
        "keywords": {"adventure", "journey", "quest", "travel", "expedition", "explore", "voyage", "uncharted", "realm", "odyssey", "discover", "mission", "epic"},
        "description": "Epic physical journeys, daring exploits, and voyages across uncharted territory."
    },
    "Betrayal": {
        "keywords": {"betray", "betrayal", "treason", "traitor", "deceive", "deception", "double-cross", "backstab", "conspiracy", "fraud", "unfaithful", "infiltrator"},
        "description": "Broken trust, treachery, and sabotage from within allied circles."
    },
    "Sacrifice": {
        "keywords": {"sacrifice", "martyr", "give", "surrender", "protect", "shield", "cost", "forfeit", "selfless", "heroic", "price", "loss"},
        "description": "Willingly forfeiting personal safety, desires, or life for a greater cause."
    },
    "Social issues": {
        "keywords": {"poverty", "inequality", "justice", "society", "class", "corruption", "oppression", "racism", "system", "rebellion", "protest", "rights", "politics", "greed"},
        "description": "Societal friction, systemic injustice, economic disparity, and political struggle."
    }
}


def detect_themes(text: str, threshold: float = 0.5) -> Dict[str, Any]:
    """
    Evaluates presence and strength of 10 narrative themes.
    Identifies specific supporting sentences as transparent empirical proof.
    """
    sentences = segment_sentences(text)
    if not sentences:
        return {
            "themes": [],
            "scores": {t: 0.0 for t in THEME_LIST},
            "dominant_theme": "None",
            "method": "Rule-Grounded Lexical Cluster Attribution"
        }

    theme_scores: Dict[str, float] = {t: 0.0 for t in THEME_LIST}
    theme_evidence_words: Dict[str, List[str]] = {t: [] for t in THEME_LIST}
    theme_supporting_sentences: Dict[str, List[str]] = {t: [] for t in THEME_LIST}

    # Sentence-level detection
    for s in sentences:
        words = [w.lower() for w in tokenize_words(s)]
        s_matched_themes = set()

        for theme_name, theme_data in THEME_LEXICON.items():
            keywords = theme_data["keywords"]
            matches = [w for w in words if w in keywords]
            if matches:
                theme_scores[theme_name] += len(matches) * 1.5
                theme_evidence_words[theme_name].extend(matches)
                s_matched_themes.add(theme_name)

        for tm in s_matched_themes:
            if s not in theme_supporting_sentences[tm]:
                theme_supporting_sentences[tm].append(s)

    # Normalize scores relative to total matches
    total_score = sum(theme_scores.values())

    detected_theme_records = []
    for theme_name in THEME_LIST:
        raw_sc = theme_scores[theme_name]
        if raw_sc > 0:
            percentage = round((raw_sc / total_score) * 100, 1) if total_score > 0 else 0.0
            evidence_words = sorted(list(set(theme_evidence_words[theme_name])))
            sentences_list = theme_supporting_sentences[theme_name]

            detected_theme_records.append({
                "theme": theme_name,
                "score": round(raw_sc, 2),
                "percentage": percentage,
                "description": THEME_LEXICON[theme_name]["description"],
                "keywords_found": evidence_words,
                "supporting_sentences": sentences_list[:2]
            })

    # Sort descending by score
    detected_theme_records = sorted(detected_theme_records, key=lambda x: x["score"], reverse=True)
    dominant = detected_theme_records[0]["theme"] if detected_theme_records else "None"

    return {
        "themes": detected_theme_records,
        "all_scores": theme_scores,
        "dominant_theme": dominant,
        "total_theme_markers": int(sum(len(v) for v in theme_evidence_words.values())),
        "method": "Rule-Grounded Semantic Clustering with Sentence Attribution"
    }
