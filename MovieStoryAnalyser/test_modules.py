"""
Movie Story Analyser - Automated Test Suite
Verifies that all 5 member modules execute cleanly and produce valid outputs.
"""

import os
import sys

# Ensure root is in python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from modules.dataset_loader import DatasetLoader
from modules.preprocessing import preprocess_text
from modules.summarizer import textrank_summarize
from modules.sentiment import analyze_storyline_sentiment
from modules.emotion import detect_emotions
from modules.keywords import extract_keywords_tfidf
from modules.themes import detect_themes
from modules.comparison import compare_two_movies


def test_member_1_dataset_loader():
    print("Testing Member 1: DatasetLoader...")
    loader = DatasetLoader()
    sample_path = os.path.join(CURRENT_DIR, "sample_data", "movies_sample.csv")
    assert os.path.exists(sample_path), f"Sample dataset not found at {sample_path}"

    success, msg, df = loader.load_from_csv(sample_path)
    assert success, f"Failed to load sample dataset: {msg}"
    assert df is not None and len(df) > 0, "DataFrame is empty"
    assert "title" in df.columns and "storyline" in df.columns, "Missing canonical columns"
    print(f"  [PASS] Successfully loaded {len(df)} movies. Stats: {loader.stats['valid_movies']} valid records.")

    # Test Custom Movie Addition
    ok_custom, msg_custom, custom_rec = loader.add_custom_movie(
        title="Chronicles of Quantum Space",
        storyline="A brilliant astrophysicist discovers a hidden tear in spacetime caused by rogue dark matter, risking her life to save earth.",
        genres=["Sci-Fi", "Drama"],
        year=2025
    )
    assert ok_custom, f"Failed to add custom movie: {msg_custom}"
    assert custom_rec is not None, "Custom record is None"
    assert custom_rec["title"] == "Chronicles of Quantum Space"
    print(f"  [PASS] Custom movie successfully added: '{custom_rec['title']}' (ID: {custom_rec['movie_id']}).")


def test_member_2_nlp_and_summarizer():
    print("Testing Member 2: Preprocessing & TextRank Summarizer...")
    text = (
        "Dom Cobb is a skilled thief who steals valuable corporate secrets during the dream state. "
        "His rare ability has made him an international fugitive and cost him everything he loved. "
        "Now Cobb is offered a chance at redemption with one final impossible job: inception. "
        "If he succeeds with his team, it could be the perfect crime, but an enemy anticipates their every move."
    )
    prep = preprocess_text(text)
    assert len(prep["sentences"]) == 4, f"Expected 4 sentences, got {len(prep['sentences'])}"
    assert len(prep["lemmas"]) > 0, "Lemmatization output is empty"

    summary = textrank_summarize(text, top_n=2)
    assert summary["short_summary"], "Short summary is empty"
    assert summary["detailed_summary"], "Detailed summary is empty"
    assert len(summary["ranked_sentences"]) == 4, "Ranking failed"
    print(f"  [PASS] TextRank successfully generated short and detailed summaries.")


def test_member_3_sentiment_and_emotion():
    print("Testing Member 3: VADER Sentiment & Emotion Detection...")
    text = "The hero bravely saves his beloved family from catastrophic destruction and finds peace and joy."
    sent = analyze_storyline_sentiment(text)
    assert sent["overall_sentiment"] in ["Positive", "Negative", "Neutral", "Mixed"]
    assert "compound_score" in sent
    assert len(sent["explanation"]) > 20

    emo = detect_emotions(text)
    assert emo["dominant_emotion"] in emo["percentages"]
    assert emo["total_emotion_markers"] > 0
    print(f"  [PASS] Sentiment: {sent['overall_sentiment']} (compound: {sent['compound_score']}), Dominant Emotion: {emo['dominant_emotion']}")


def test_member_4_keywords_themes_comparison():
    print("Testing Member 4: Keywords, Themes, and Comparison...")
    text1 = "Batman battles the Joker who unleashes terrifying chaos and revenge on Gotham City."
    text2 = "A detective investigates mysterious murders and betrayal within the criminal mob syndicate."

    kw = extract_keywords_tfidf(text1, top_n=5)
    assert len(kw["keywords"]) > 0, "Keyword extraction failed"

    themes = detect_themes(text1)
    assert len(themes["themes"]) > 0, "Theme detection failed"

    movie_a = {"title": "Batman", "storyline": text1, "year": 2008, "genres": ["Action"]}
    movie_b = {"title": "Detective", "storyline": text2, "year": 1995, "genres": ["Mystery"]}
    comp = compare_two_movies(movie_a, movie_b)
    assert "similarity_percentage" in comp
    assert "shared_keywords" in comp
    print(f"  [PASS] Extracted {len(kw['keywords'])} keywords. Comparison similarity: {comp['similarity_percentage']}%")


if __name__ == "__main__":
    print("=" * 60)
    print("Movie Story Analyser - B.Tech CSE AIML Verification Suite")
    print("=" * 60)
    test_member_1_dataset_loader()
    test_member_2_nlp_and_summarizer()
    test_member_3_sentiment_and_emotion()
    test_member_4_keywords_themes_comparison()
    print("=" * 60)
    print("ALL MODULE TESTS PASSED SUCCESSFULLY! Ready for deployment.")
    print("=" * 60)
