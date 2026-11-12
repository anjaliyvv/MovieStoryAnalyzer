"""
Movie Story Analyser - Global Configuration & Constants
B.Tech CSE AIML Group Project
"""

import os

# Application Metadata
APP_NAME = "Movie Story Analyser"
APP_SUBTITLE = "Dataset-Based NLP Application for Storyline Intelligence"
PROJECT_TEAM = "B.Tech CSE AIML Group 5"
VERSION = "1.0.0"

# Color Palette (Strictly adheres to Project Design Requirements)
COLORS = {
    "background": "#11161B",
    "secondary_background": "#1B2229",
    "card_background": "#161D24",
    "card_border": "#28333E",
    "accent": "#C9A47F",
    "secondary_accent": "#8D7058",
    "main_text": "#F1EEE8",
    "muted_text": "#AAB1B9",
    "positive": "#4E9F76",
    "negative": "#D95D5D",
    "neutral": "#8F9DAA",
    "mixed": "#D4A373",
}

# Column aliases for robust automated dataset normalization
COLUMN_MAPPINGS = {
    "title": [
        "title", "movie_title", "original_title", "film_title", "movie_name", "name"
    ],
    "storyline": [
        "overview", "plot", "description", "storyline", "synopsis", "summary", "story"
    ],
    "genres": [
        "genres", "genre", "genre_names", "genres_list", "category"
    ],
    "year": [
        "year", "release_year", "release_date", "released_year", "date"
    ],
    "director": [
        "director", "directors", "directed_by"
    ],
    "cast": [
        "cast", "actors", "starring", "stars", "lead_actors"
    ],
    "poster_url": [
        "poster_url", "poster_path", "poster", "image_url", "image", "img_url"
    ]
}

REQUIRED_CANONICAL_COLUMNS = ["title", "storyline"]

# Target Emotions for NLP Classification
EMOTION_LIST = [
    "Happiness", "Sadness", "Fear", "Anger", "Surprise", "Love", "Hope"
]

# Target Storyline Themes for Narrative Extraction
THEME_LIST = [
    "Friendship", "Love", "Revenge", "Survival", "Family",
    "Mystery", "Adventure", "Betrayal", "Sacrifice", "Social issues"
]

# Team Roles (Five-Member B.Tech CSE AIML Project Structure)
TEAM_ROLES = {
    "Member 1": {
        "name": "Member 1",
        "role": "Dataset Engineering & Integration",
        "tasks": "CSV ingestion, automatic schema recognition, data validation, deduplication, missing-value imputation, filtering."
    },
    "Member 2": {
        "name": "Member 2",
        "role": "NLP Text Processing & Summarization",
        "tasks": "Sentence segmentation, tokenization, lemmatization, stopword removal, TextRank graph summarizer using NetworkX."
    },
    "Member 3": {
        "name": "Member 3",
        "role": "Sentiment Analysis & Emotion Detection",
        "tasks": "VADER rule-based sentiment model, sentence attribution, explainable negative polarity diagnosis, 7-emotion scoring."
    },
    "Member 4": {
        "name": "Member 4",
        "role": "Keywords, Themes & Comparative Analytics",
        "tasks": "TF-IDF unigram/bigram keyword extraction, rule-grounded theme identification, cosine similarity matrix calculation."
    },
    "Member 5": {
        "name": "Member 5",
        "role": "UI/UX Architecture & Streamlit Orchestration",
        "tasks": "Custom Streamlit CSS layout, multi-page routing, session state caching, Plotly visualizations, end-to-end testing."
    }
}

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
MODULES_DIR = os.path.join(BASE_DIR, "modules")
SAMPLE_DATA_DIR = os.path.join(BASE_DIR, "sample_data")
