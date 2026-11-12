"""
Movie Story Analyser - Main Streamlit Application
B.Tech CSE AIML Group Project (5 Members)
Framework: Streamlit exclusively for UI
"""

import os
import sys
import pandas as pd
import streamlit as st

# Configure module import paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from config import (
    APP_NAME, APP_SUBTITLE, PROJECT_TEAM, VERSION, COLORS,
    THEME_LIST, EMOTION_LIST, TEAM_ROLES, ASSETS_DIR, SAMPLE_DATA_DIR
)
from modules.dataset_loader import DatasetLoader
from modules.preprocessing import preprocess_text
from modules.summarizer import textrank_summarize
from modules.sentiment import analyze_storyline_sentiment
from modules.emotion import detect_emotions, create_emotion_chart
from modules.keywords import extract_keywords_tfidf, create_keywords_chart
from modules.themes import detect_themes
from modules.comparison import compare_two_movies

# ---------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title=f"{APP_NAME} | NLP Analytics",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Inject Custom CSS Styling
# ---------------------------------------------------------
css_path = os.path.join(ASSETS_DIR, "style.css")
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "loader" not in st.session_state:
    st.session_state.loader = DatasetLoader()

if "selected_movie_id" not in st.session_state:
    st.session_state.selected_movie_id = None

if "dataset_loaded" not in st.session_state:
    st.session_state.dataset_loaded = False

if "nav_page" not in st.session_state:
    st.session_state.nav_page = "Dashboard"

if "compare_a_id" not in st.session_state:
    st.session_state.compare_a_id = None

if "compare_b_id" not in st.session_state:
    st.session_state.compare_b_id = None


def load_dataset(file_or_path, is_sample=False):
    """Loads and validates dataset into session state."""
    success, msg, df = st.session_state.loader.load_from_csv(file_or_path)
    if success and df is not None and not df.empty:
        st.session_state.dataset_loaded = True
        st.session_state.selected_movie_id = df.iloc[0]["movie_id"]
        if len(df) > 1:
            st.session_state.compare_a_id = df.iloc[0]["movie_id"]
            st.session_state.compare_b_id = df.iloc[1]["movie_id"]
        return True, msg
    return False, msg


# Automatically load sample dataset on initial cold start if no user file yet
sample_csv_path = os.path.join(SAMPLE_DATA_DIR, "movies_sample.csv")
if not st.session_state.dataset_loaded and os.path.exists(sample_csv_path):
    load_dataset(sample_csv_path, is_sample=True)

loader: DatasetLoader = st.session_state.loader
movies_df = loader.cleaned_df

# ---------------------------------------------------------
# Sidebar Navigation & Dataset Controls
# ---------------------------------------------------------
with st.sidebar:
    st.markdown(f"""
        <div style="padding: 10px 0 20px 0; border-bottom: 1px solid #28333E; margin-bottom: 20px;">
            <div style="font-size: 1.4rem; font-weight: 700; color: #C9A47F; letter-spacing: -0.02em;">
                🎬 {APP_NAME}
            </div>
            <div style="font-size: 0.8rem; color: #AAB1B9; margin-top: 4px;">
                {APP_SUBTITLE}
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Navigation Menu
    pages = [
        "Dashboard",
        "Movie Explorer",
        "Story Analysis",
        "Sentiment and Emotions",
        "Keywords and Themes",
        "Compare Movies"
    ]

    selected_page = st.radio(
        "Navigation",
        pages,
        index=pages.index(st.session_state.nav_page) if st.session_state.nav_page in pages else 0,
        label_visibility="collapsed"
    )
    st.session_state.nav_page = selected_page

    st.markdown("<hr style='border: none; border-top: 1px solid #28333E; margin: 20px 0;'>", unsafe_allow_html=True)

    # Custom Movie Entry Section
    st.markdown("### ✍️ Add Custom Movie")
    with st.expander("Enter Movie Title & Plot", expanded=True):
        custom_title = st.text_input("Movie Title", placeholder="e.g. Inception 2 / My Movie", key="sb_cust_title")
        custom_plot = st.text_area("Storyline / Plot", placeholder="Enter the complete movie storyline or synopsis here (at least 15 characters)...", height=120, key="sb_cust_plot")
        custom_genre = st.selectbox("Genre", ["Sci-Fi", "Drama", "Action", "Thriller", "Romance", "Crime", "Comedy", "Animation", "Horror", "Adventure", "Fantasy"], key="sb_cust_genre")
        custom_year = st.number_input("Year (Optional)", min_value=1900, max_value=2099, value=2024, step=1, key="sb_cust_year")

        if st.button("🚀 Analyze Custom Movie", type="primary", use_container_width=True, key="sb_cust_submit"):
            if not custom_title.strip():
                st.error("Please enter a movie title.")
            elif len(custom_plot.strip()) < 15:
                st.error("Please enter at least 15 characters of storyline.")
            else:
                ok, msg, rec = loader.add_custom_movie(
                    title=custom_title,
                    storyline=custom_plot,
                    genres=[custom_genre],
                    year=int(custom_year) if custom_year else None
                )
                if ok and rec:
                    st.session_state.dataset_loaded = True
                    st.session_state.selected_movie_id = rec["movie_id"]
                    st.session_state.nav_page = "Story Analysis"
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

    st.markdown("<hr style='border: none; border-top: 1px solid #28333E; margin: 15px 0;'>", unsafe_allow_html=True)

    # Optional CSV Upload Section
    with st.expander("📁 Or Upload Dataset CSV", expanded=False):
        uploaded_file = st.file_uploader(
            "Upload Movie Dataset (CSV)",
            type=["csv"],
            help="Upload any CSV file with movie titles and storylines/overviews."
        )

        if uploaded_file is not None:
            if st.button("Ingest Uploaded CSV", type="secondary", use_container_width=True):
                with st.spinner("Parsing, reconciling columns and validating records..."):
                    ok, msg = load_dataset(uploaded_file)
                    if ok:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)

    # Button to reload curated sample data
    if os.path.exists(sample_csv_path):
        if st.button("Reload Built-in Sample Dataset", use_container_width=True):
            ok, msg = load_dataset(sample_csv_path, is_sample=True)
            if ok:
                st.success("Sample dataset reloaded!")
                st.rerun()

    # Dataset Live Metrics in Sidebar
    if st.session_state.dataset_loaded and loader.stats:
        st.markdown("<hr style='border: none; border-top: 1px solid #28333E; margin: 15px 0;'>", unsafe_allow_html=True)
        st.markdown("#### 📊 Dataset Status")
        st.markdown(f"**Valid Movies:** `{loader.stats.get('valid_movies', 0)}`")
        st.markdown(f"**Genres Count:** `{loader.stats.get('unique_genres_count', 0)}`")
        yr_min, yr_max = loader.stats.get('year_range', ('N/A', 'N/A'))
        st.markdown(f"**Release Span:** `{yr_min} - {yr_max}`")

    # Team Footer
    st.markdown("<hr style='border: none; border-top: 1px solid #28333E; margin: 20px 0;'>", unsafe_allow_html=True)
    st.markdown(f"""
        <div style="font-size: 0.75rem; color: #AAB1B9; text-align: center;">
            {PROJECT_TEAM} &bull; v{VERSION}<br>
            NLP Group Project
        </div>
    """, unsafe_allow_html=True)


# Helper to get current selected movie record
def get_selected_movie():
    if movies_df is None or movies_df.empty:
        return None
    if st.session_state.selected_movie_id is not None:
        match = movies_df[movies_df["movie_id"] == st.session_state.selected_movie_id]
        if not match.empty:
            return match.iloc[0].to_dict()
    return movies_df.iloc[0].to_dict()


# ---------------------------------------------------------
# PAGE 1: DASHBOARD
# ---------------------------------------------------------
if st.session_state.nav_page == "Dashboard":
    st.markdown(f"""
        <div style="margin-bottom: 25px;">
            <h1 style="color: #F1EEE8; font-size: 2.2rem; font-weight: 700; margin-bottom: 6px;">
                {APP_NAME}
            </h1>
            <p style="color: #AAB1B9; font-size: 1.05rem; margin-top: 0;">
                Comprehensive NLP intelligence platform for movie storyline analysis, extractive summarization,
                explainable sentiment attribution, emotion profiling, thematic detection, and dual-story comparison.
            </p>
        </div>
    """, unsafe_allow_html=True)

    if not st.session_state.dataset_loaded or movies_df is None or movies_df.empty:
        st.warning("Please upload a CSV dataset or load the sample dataset from the sidebar to begin.")
    else:
        stats = loader.stats

        # Metric Cards Row
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total Movies", stats.get("valid_movies", 0), help="Deduplicated and validated movies")
        with c2:
            st.metric("Unique Genres", stats.get("unique_genres_count", 0), help="Number of distinct genres in dataset")
        with c3:
            y_min, y_max = stats.get("year_range", ("N/A", "N/A"))
            st.metric("Release Range", f"{y_min}–{y_max}", help="Earliest to latest release year")
        with c4:
            st.metric("Avg Plot Length", f"{stats.get('avg_storyline_words', 0)} words", help="Average words per storyline")

        st.markdown("<br>", unsafe_allow_html=True)

        # Custom Movie Direct Entry Card on Dashboard
        st.markdown("""
            <div style="background: #161D24; border: 1px solid #C9A47F; border-radius: 12px; padding: 20px; margin-bottom: 25px; box-shadow: 0 4px 16px rgba(201, 164, 127, 0.1);">
                <div style="font-size: 0.8rem; font-weight: 700; color: #C9A47F; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px;">
                    ✍️ Custom Movie Entry
                </div>
                <h3 style="color: #F1EEE8; margin-top: 0; margin-bottom: 8px; font-size: 1.25rem;">
                    Analyze Any Custom Movie Storyline
                </h3>
                <p style="color: #AAB1B9; font-size: 0.88rem; margin-top: 0; margin-bottom: 15px;">
                    Enter your custom movie title and full plot below to instantly generate extractive summaries, explainable sentiment, emotion profiling, and thematic extraction.
                </p>
            </div>
        """, unsafe_allow_html=True)

        with st.container():
            d_c1, d_c2 = st.columns([1, 1])
            with d_c1:
                dash_custom_title = st.text_input("Custom Movie Title", placeholder="e.g., Inception 2: Deep Dreams", key="dash_c_title")
                dash_custom_genre = st.selectbox("Genre", ["Sci-Fi", "Drama", "Action", "Thriller", "Romance", "Crime", "Comedy", "Animation", "Horror", "Adventure", "Fantasy"], key="dash_c_genre")
            with d_c2:
                dash_custom_year = st.number_input("Release Year (Optional)", min_value=1900, max_value=2099, value=2024, step=1, key="dash_c_year")
                dash_custom_director = st.text_input("Director (Optional)", placeholder="e.g., Christopher Nolan / Independent", key="dash_c_dir")

            dash_custom_plot = st.text_area(
                "Movie Storyline / Plot Synopsis",
                placeholder="Paste or write the movie storyline, narrative arc, conflict, and climax here (at least 15 characters)...",
                height=130,
                key="dash_c_plot"
            )

            if st.button("🚀 Add & Analyze Custom Movie Now", type="primary", use_container_width=True, key="dash_c_submit"):
                if not dash_custom_title.strip():
                    st.error("Please provide a movie title.")
                elif len(dash_custom_plot.strip()) < 15:
                    st.error("Please provide at least 15 characters of storyline plot.")
                else:
                    ok, msg, rec = loader.add_custom_movie(
                        title=dash_custom_title,
                        storyline=dash_custom_plot,
                        genres=[dash_custom_genre],
                        year=int(dash_custom_year) if dash_custom_year else None,
                        director=dash_custom_director if dash_custom_director else "Custom / Independent"
                    )
                    if ok and rec:
                        st.session_state.dataset_loaded = True
                        st.session_state.selected_movie_id = rec["movie_id"]
                        st.session_state.nav_page = "Story Analysis"
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)

        st.markdown("<hr style='border: none; border-top: 1px solid #28333E; margin: 25px 0;'>", unsafe_allow_html=True)

        # Recently Selected Movie Banner / Card
        curr_movie = get_selected_movie()
        if curr_movie:
            st.markdown("### 🎯 Currently Selected Movie")
            with st.container():
                cols = st.columns([1, 4])
                with cols[0]:
                    if curr_movie.get("poster_url"):
                        st.image(curr_movie["poster_url"], use_container_width=True)
                    else:
                        st.markdown("""
                            <div style="background: #1B2229; height: 180px; display: flex; align-items: center; justify-content: center; border-radius: 8px; border: 1px dashed #28333E; color: #AAB1B9;">
                                🎬 No Poster
                            </div>
                        """, unsafe_allow_html=True)
                with cols[1]:
                    st.markdown(f"## {curr_movie['title']} ({curr_movie.get('year', 'N/A')})")
                    genres_html = " ".join([f"<span class='badge badge-accent'>{g}</span>" for g in curr_movie.get("genres", [])])
                    st.markdown(genres_html, unsafe_allow_html=True)
                    st.markdown(f"**Director:** {curr_movie.get('director', 'Unknown')} &nbsp;|&nbsp; **Cast:** {curr_movie.get('cast', 'Unknown')}")
                    st.markdown(f"<p style='color: #AAB1B9; font-size: 0.95rem; line-height: 1.5;'>{curr_movie['storyline'][:250]}...</p>", unsafe_allow_html=True)

                    btn_c1, btn_c2 = st.columns([1, 1])
                    with btn_c1:
                        if st.button("🚀 Analyze This Storyline", key="dash_analyze_btn", type="primary"):
                            st.session_state.nav_page = "Story Analysis"
                            st.rerun()
                    with btn_c2:
                        if st.button("🔍 Explore More Movies", key="dash_explore_btn"):
                            st.session_state.nav_page = "Movie Explorer"
                            st.rerun()

        st.markdown("<hr style='border: none; border-top: 1px solid #28333E; margin: 30px 0;'>", unsafe_allow_html=True)

        # Dataset Preview & Column Mapping Diagnostic
        st.markdown("### 📋 Active Dataset Preview")
        preview_cols = ["title", "year", "primary_genre", "director", "word_count", "storyline"]
        display_df = movies_df[[c for c in preview_cols if c in movies_df.columns]].head(10)
        st.dataframe(display_df, use_container_width=True, height=260)

        with st.expander("🛠️ Column Mapping & Schema Reconciliation Details"):
            st.markdown(
                "The application automatically mapped the following columns from your uploaded CSV file into standard canonical fields:"
            )
            mapping_data = [{"Canonical Field": k, "Matched CSV Column": v} for k, v in stats.get("detected_columns", {}).items()]
            st.table(pd.DataFrame(mapping_data))


# ---------------------------------------------------------
# PAGE 2: MOVIE EXPLORER
# ---------------------------------------------------------
elif st.session_state.nav_page == "Movie Explorer":
    st.markdown("## 🔍 Movie Explorer")
    st.markdown("Search, filter, and choose any movie from the dataset for in-depth NLP storyline decomposition.")

    if movies_df is None or movies_df.empty:
        st.warning("No dataset loaded.")
    else:
        # Search & Filter Controls
        f1, f2, f3 = st.columns([2, 1, 1])
        with f1:
            search_query = st.text_input("Search by title, director, or keyword", placeholder="e.g. Inception, Nolan, dreams...")
        with f2:
            all_genres = ["All"] + loader.stats.get("unique_genres", [])
            selected_genre = st.selectbox("Genre Filter", all_genres)
        with f3:
            years_present = movies_df["year"].dropna()
            if not years_present.empty and int(years_present.min()) < int(years_present.max()):
                min_y, max_y = int(years_present.min()), int(years_present.max())
                year_range = st.slider("Year Range", min_value=min_y, max_value=max_y, value=(min_y, max_y))
            else:
                year_range = None

        # Filter movies
        filtered_movies = loader.search_and_filter(
            query=search_query,
            selected_genre=selected_genre,
            year_range=year_range
        )

        st.markdown(f"**Found {len(filtered_movies)} movies matching criteria**")
        st.markdown("<br>", unsafe_allow_html=True)

        # Grid of Movie Cards
        for idx, row in filtered_movies.iterrows():
            m_id = row["movie_id"]
            is_active = (m_id == st.session_state.selected_movie_id)

            card_border_color = COLORS["accent"] if is_active else "#28333E"

            st.markdown(f"""
                <div style="background: #161D24; border: 1px solid {card_border_color}; border-radius: 12px; padding: 18px; margin-bottom: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.3);">
            """, unsafe_allow_html=True)

            c_img, c_info, c_action = st.columns([1, 4, 1.2])

            with c_img:
                if row.get("poster_url"):
                    st.image(row["poster_url"], use_container_width=True)
                else:
                    st.markdown("""
                        <div style="background: #1B2229; height: 140px; display: flex; align-items: center; justify-content: center; border-radius: 6px; color: #AAB1B9; font-size: 0.8rem;">
                            🎬 No Image
                        </div>
                    """, unsafe_allow_html=True)

            with c_info:
                st.markdown(f"<h3 style='margin: 0; color: #F1EEE8;'>{row['title']} <span style='color: #AAB1B9; font-size: 0.9rem;'>({row.get('year', 'N/A')})</span></h3>", unsafe_allow_html=True)
                badges = " ".join([f"<span class='badge'>{g}</span>" for g in row.get("genres", [])])
                st.markdown(f"<div style='margin: 6px 0;'>{badges}</div>", unsafe_allow_html=True)
                st.markdown(f"<p style='color: #AAB1B9; font-size: 0.9rem; line-height: 1.4; margin: 6px 0;'>{row['storyline'][:240]}...</p>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-size: 0.8rem; color: #8D7058;'>Director: {row.get('director', 'Unknown')} | Words: {row.get('word_count', 0)}</div>", unsafe_allow_html=True)

            with c_action:
                st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
                if st.button("Select & Analyze", key=f"sel_btn_{m_id}", type="primary" if is_active else "secondary", use_container_width=True):
                    st.session_state.selected_movie_id = m_id
                    st.session_state.nav_page = "Story Analysis"
                    st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------
# PAGE 3: STORY ANALYSIS (PREPROCESSING & SUMMARIZATION)
# ---------------------------------------------------------
elif st.session_state.nav_page == "Story Analysis":
    curr_movie = get_selected_movie()
    if not curr_movie:
        st.warning("Please select a movie first.")
    else:
        st.markdown(f"## 📖 Story Analysis: {curr_movie['title']} ({curr_movie.get('year', 'N/A')})")

        # Header Info Banner
        st.markdown(f"""
            <div style="background: #161D24; border: 1px solid #28333E; border-radius: 10px; padding: 14px 20px; margin-bottom: 20px;">
                <span class="badge badge-accent">{curr_movie.get('primary_genre', 'General')}</span>
                <span style="color: #AAB1B9; font-size: 0.9rem; margin-left: 10px;">Director: <strong>{curr_movie.get('director', 'Unknown')}</strong></span>
                <span style="color: #AAB1B9; font-size: 0.9rem; margin-left: 15px;">Cast: <strong>{curr_movie.get('cast', 'Unknown')}</strong></span>
                <span style="color: #AAB1B9; font-size: 0.9rem; margin-left: 15px;">Story Length: <strong>{curr_movie.get('word_count', 0)} words</strong></span>
            </div>
        """, unsafe_allow_html=True)

        story_text = curr_movie["storyline"]

        # Top Section: Full Raw Storyline
        with st.expander("📄 View Complete Unprocessed Storyline", expanded=True):
            st.markdown(f"<div style='line-height: 1.6; font-size: 1.05rem;'>{story_text}</div>", unsafe_allow_html=True)

        st.markdown("<hr style='border: none; border-top: 1px solid #28333E; margin: 25px 0;'>", unsafe_allow_html=True)

        # Section 1: Story Summarization (TextRank + NetworkX)
        st.markdown("### ⚡ Extractive Story Summarization (TextRank Graph Centrality)")
        st.markdown(
            "Extractive summarization constructs a weighted sentence graph based on semantic overlap "
            "and uses PageRank graph centrality to identify the most salient storyline sentences."
        )

        sum_cols = st.columns([1, 3])
        with sum_cols[0]:
            top_n_sents = st.slider("Detailed Summary Sentences", min_value=1, max_value=5, value=2)

        summary_result = textrank_summarize(story_text, top_n=top_n_sents)

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown("#### 📌 Short Executive Summary (Top 1–2 Sentences)")
            st.markdown(f"""
                <div style="background: #1B2229; border-left: 4px solid #C9A47F; padding: 16px; border-radius: 0 8px 8px 0; line-height: 1.6;">
                    {summary_result['short_summary']}
                </div>
            """, unsafe_allow_html=True)

        with col_s2:
            st.markdown(f"#### 📜 Detailed Extractive Summary ({summary_result['selected_count']} Sentences)")
            st.markdown(f"""
                <div style="background: #1B2229; border-left: 4px solid #8D7058; padding: 16px; border-radius: 0 8px 8px 0; line-height: 1.6;">
                    {summary_result['detailed_summary']}
                </div>
            """, unsafe_allow_html=True)

        # Ranked Sentences with Graph Centrality Scores Table
        with st.expander("🔬 View Sentence Graph Centrality & PageRank Scores"):
            ranked_df = pd.DataFrame(summary_result["ranked_sentences"])
            if not ranked_df.empty:
                st.dataframe(ranked_df[["rank", "score", "sentence"]], use_container_width=True)

        st.markdown("<hr style='border: none; border-top: 1px solid #28333E; margin: 25px 0;'>", unsafe_allow_html=True)

        # Section 2: NLP Text Preprocessing Pipeline
        st.markdown("### ⚙️ Multi-Stage Text Preprocessing Pipeline")
        st.markdown("Inspect how the raw narrative text transforms across each standard NLP stage.")

        prep = preprocess_text(story_text)

        tab1, tab2, tab3, tab4 = st.tabs([
            "1. Sentence Segmentation",
            "2. Tokenization & Cleaning",
            "3. Stopword Removal",
            "4. Lemmatization"
        ])

        with tab1:
            st.markdown(f"**Segmented into {len(prep['sentences'])} discrete sentences:**")
            for i, s in enumerate(prep["sentences"], 1):
                st.markdown(f"**[{i}]** {s}")

        with tab2:
            st.markdown(f"**Extracted {len(prep['tokens'])} tokens (lowercase & punctuation stripped):**")
            st.write(", ".join(prep["cleaned_tokens"][:80]) + ("..." if len(prep["cleaned_tokens"]) > 80 else ""))

        with tab3:
            st.markdown(f"**Filtered {len(prep['filtered_tokens'])} non-stopword tokens (stopwords removed):**")
            st.write(", ".join(prep["filtered_tokens"][:80]) + ("..." if len(prep["filtered_tokens"]) > 80 else ""))

        with tab4:
            st.markdown(f"**Lemmatized Canonical Vocabulary ({prep['stats']['vocab_size']} unique lemmas):**")
            st.write(prep["processed_text_string"][:400] + "...")


# ---------------------------------------------------------
# PAGE 4: SENTIMENT AND EMOTIONS
# ---------------------------------------------------------
elif st.session_state.nav_page == "Sentiment and Emotions":
    curr_movie = get_selected_movie()
    if not curr_movie:
        st.warning("Please select a movie first.")
    else:
        st.markdown(f"## 🎭 Sentiment & Emotion Analysis: {curr_movie['title']}")

        story_text = curr_movie["storyline"]

        # Run Analysis
        sent_data = analyze_storyline_sentiment(story_text)
        emo_data = detect_emotions(story_text)

        # Top Sentiment Metrics
        st.markdown("### 1. Explainable Storyline Sentiment (VADER)")

        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            sent_label = sent_data["overall_sentiment"]
            st.metric("Overall Polarity", sent_label)
        with m2:
            st.metric("Mean Compound", f"{sent_data['compound_score']:+.2f}", help="VADER compound score bounded between -1.0 and +1.0")
        with m3:
            st.metric("Positive Sents", sent_data["counts"]["Positive"])
        with m4:
            st.metric("Negative Sents", sent_data["counts"]["Negative"])
        with m5:
            st.metric("Neutral Sents", sent_data["counts"]["Neutral"])

        # Explanation Box
        st.markdown(f"""
            <div style="background: #161D24; border-left: 4px solid #C9A47F; padding: 16px 20px; border-radius: 0 10px 10px 0; margin: 15px 0 25px 0; line-height: 1.6;">
                <div style="font-weight: 600; color: #C9A47F; margin-bottom: 6px;">💡 Explainability Diagnostic: Why this classification was assigned</div>
                {sent_data['explanation']}
            </div>
        """, unsafe_allow_html=True)

        # Sentence-Level Breakdown
        st.markdown("#### Sentence-by-Sentence Sentiment Breakdown")
        for s_res in sent_data["sentences"]:
            cat = s_res["sentiment"]
            compound = s_res["compound"]
            if cat == "Positive":
                box_class = "sentence-pos"
                badge_class = "badge-positive"
            elif cat == "Negative":
                box_class = "sentence-neg"
                badge_class = "badge-negative"
            else:
                box_class = "sentence-neu"
                badge_class = "badge-neutral"

            st.markdown(f"""
                <div class="sentence-box {box_class}">
                    <span class="badge {badge_class}">{cat} ({compound:+.2f})</span>
                    <span>{s_res['sentence']}</span>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<hr style='border: none; border-top: 1px solid #28333E; margin: 30px 0;'>", unsafe_allow_html=True)

        # Section 2: Emotion Detection
        st.markdown("### 2. Narrative Emotion Profiling (7 Dimensions)")
        st.markdown(
            "Emotions are scored across 7 psychological narrative dimensions using an empirical emotion lexicon "
            "with contextual intensity and lexical ground-truth tracking."
        )

        e_col1, e_col2 = st.columns([1, 2])
        with e_col1:
            st.markdown("#### Dominant Emotions")
            st.markdown(f"""
                <div style="background: #161D24; border: 1px solid #28333E; border-radius: 10px; padding: 18px; margin-bottom: 15px;">
                    <div style="font-size: 0.85rem; color: #AAB1B9; text-transform: uppercase;">Dominant Emotion</div>
                    <div style="font-size: 1.6rem; font-weight: 700; color: #C9A47F; margin: 4px 0 12px 0;">
                        {emo_data['dominant_emotion']}
                    </div>
                    <div style="font-size: 0.85rem; color: #AAB1B9; text-transform: uppercase;">Secondary Emotion</div>
                    <div style="font-size: 1.3rem; font-weight: 600; color: #8D7058; margin-top: 4px;">
                        {emo_data['secondary_emotion']}
                    </div>
                    <div style="font-size: 0.8rem; color: #AAB1B9; margin-top: 14px;">
                        Emotion Lexical Markers Found: <strong>{emo_data['total_emotion_markers']}</strong>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            with st.expander("🔎 View Detected Emotion Keywords"):
                for emo, kw_list in emo_data["evidence"].items():
                    if kw_list:
                        terms = ", ".join([f"{term} ({cnt})" for term, cnt in kw_list[:4]])
                        st.markdown(f"**{emo}:** {terms}")

        with e_col2:
            fig = create_emotion_chart(emo_data, curr_movie["title"])
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write(emo_data["percentages"])


# ---------------------------------------------------------
# PAGE 5: KEYWORDS AND THEMES
# ---------------------------------------------------------
elif st.session_state.nav_page == "Keywords and Themes":
    curr_movie = get_selected_movie()
    if not curr_movie:
        st.warning("Please select a movie first.")
    else:
        st.markdown(f"## 🏷️ Keywords & Thematic Analysis: {curr_movie['title']}")

        story_text = curr_movie["storyline"]

        # Run extraction
        corpus = list(movies_df["storyline"].dropna()) if movies_df is not None else None
        kw_data = extract_keywords_tfidf(story_text, corpus=corpus, top_n=10)
        theme_data = detect_themes(story_text)

        # Section 1: Keywords
        st.markdown("### 1. TF-IDF Salient Keywords & Phrases")
        st.markdown(
            "Extracts distinctive unigrams and bigram phrases using sublinear Term Frequency - Inverse Document Frequency (TF-IDF)."
        )

        k1, k2 = st.columns([1.2, 1])
        with k1:
            fig_kw = create_keywords_chart(kw_data["keywords"], curr_movie["title"])
            if fig_kw:
                st.plotly_chart(fig_kw, use_container_width=True)
            else:
                st.dataframe(pd.DataFrame(kw_data["keywords"]))

        with k2:
            st.markdown("#### Ranked TF-IDF Table")
            if kw_data["keywords"]:
                df_kw = pd.DataFrame(kw_data["keywords"])
                st.dataframe(df_kw[["keyword", "type", "score", "frequency"]], use_container_width=True, height=330)

        st.markdown("<hr style='border: none; border-top: 1px solid #28333E; margin: 30px 0;'>", unsafe_allow_html=True)

        # Section 2: Narrative Themes
        st.markdown("### 2. Storyline Thematic Detection")
        st.markdown(
            "Detects high-level narrative themes across 10 storytelling archetypes, "
            "substantiated by keyword frequency and supporting storyline sentences."
        )

        st.markdown(f"**Dominant Theme:** <span class='badge badge-accent' style='font-size: 1rem;'>{theme_data['dominant_theme']}</span>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        if not theme_data["themes"]:
            st.info("No prominent thematic motifs exceeded the detection threshold for this storyline.")
        else:
            for t_item in theme_data["themes"]:
                st.markdown(f"""
                    <div style="background: #161D24; border: 1px solid #28333E; border-radius: 10px; padding: 16px 20px; margin-bottom: 14px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <h4 style="margin: 0; color: #F1EEE8;">{t_item['theme']}</h4>
                            <span class="badge badge-accent">{t_item['percentage']}% Relevance (Score: {t_item['score']})</span>
                        </div>
                        <p style="color: #AAB1B9; font-size: 0.88rem; margin: 6px 0;">{t_item['description']}</p>
                        <div style="font-size: 0.85rem; color: #C9A47F; margin: 4px 0;">
                            <strong>Lexical Evidence:</strong> {', '.join(t_item['keywords_found'])}
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                if t_item["supporting_sentences"]:
                    with st.expander(f"Supporting sentence for '{t_item['theme']}'"):
                        for sent in t_item["supporting_sentences"]:
                            st.markdown(f"&bull; *\"{sent}\"*")


# ---------------------------------------------------------
# PAGE 6: COMPARE MOVIES
# ---------------------------------------------------------
elif st.session_state.nav_page == "Compare Movies":
    st.markdown("## ⚖️ Dual Movie Comparison")
    st.markdown("Select any two movies from the uploaded dataset to perform dynamic side-by-side NLP comparative intelligence.")

    if movies_df is None or len(movies_df) < 2:
        st.warning("Please upload a dataset with at least two movies to enable dual comparison.")
    else:
        movie_titles = {row["movie_id"]: f"{row['title']} ({row.get('year', 'N/A')})" for _, row in movies_df.iterrows()}
        movie_ids = list(movie_titles.keys())

        # Selectboxes for Movie A and Movie B
        col_sel1, col_sel2 = st.columns(2)
        with col_sel1:
            default_a_idx = movie_ids.index(st.session_state.compare_a_id) if st.session_state.compare_a_id in movie_ids else 0
            sel_a_id = st.selectbox(
                "Select Movie A",
                movie_ids,
                index=default_a_idx,
                format_func=lambda m_id: movie_titles[m_id]
            )
            st.session_state.compare_a_id = sel_a_id

        with col_sel2:
            default_b_idx = movie_ids.index(st.session_state.compare_b_id) if st.session_state.compare_b_id in movie_ids and len(movie_ids) > 1 else (1 if len(movie_ids) > 1 else 0)
            sel_b_id = st.selectbox(
                "Select Movie B",
                movie_ids,
                index=default_b_idx,
                format_func=lambda m_id: movie_titles[m_id]
            )
            st.session_state.compare_b_id = sel_b_id

        # Fetch records
        rec_a = movies_df[movies_df["movie_id"] == sel_a_id].iloc[0].to_dict()
        rec_b = movies_df[movies_df["movie_id"] == sel_b_id].iloc[0].to_dict()

        # Compute dynamic comparison
        with st.spinner("Calculating TF-IDF vector space similarity, sentiment delta, and emotion overlap..."):
            comp = compare_two_movies(rec_a, rec_b)

        # Similarity Meter Banner
        sim_pct = comp["similarity_percentage"]
        st.markdown(f"""
            <div style="background: #161D24; border: 1px solid #28333E; border-radius: 12px; padding: 20px; text-align: center; margin: 20px 0;">
                <div style="font-size: 0.9rem; color: #AAB1B9; text-transform: uppercase; letter-spacing: 0.05em;">
                    TF-IDF Vector Space Storyline Similarity
                </div>
                <div style="font-size: 3rem; font-weight: 800; color: #C9A47F; margin: 8px 0;">
                    {sim_pct}%
                </div>
                <div style="font-size: 0.95rem; color: #F1EEE8;">
                    Cosine Similarity Score: <code>{comp['similarity_score']:.4f}</code>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Side-by-Side Summary & Core Metrics
        c_a, c_b = st.columns(2)

        with c_a:
            st.markdown(f"### 🎬 {comp['movie_a']['title']}")
            if rec_a.get("poster_url"):
                st.image(rec_a["poster_url"], width=200)

            genres_a = " ".join([f"<span class='badge badge-accent'>{g}</span>" for g in comp['movie_a']['genres']])
            st.markdown(genres_a, unsafe_allow_html=True)

            st.markdown(f"""
                <div style="background: #1B2229; border: 1px solid #28333E; border-radius: 8px; padding: 14px; margin: 12px 0;">
                    <div><strong>Word Count:</strong> {comp['movie_a']['word_count']} words</div>
                    <div><strong>Sentiment:</strong> {comp['movie_a']['sentiment']} (Compound: {comp['movie_a']['compound']:+.2f})</div>
                    <div><strong>Dominant Emotion:</strong> {comp['movie_a']['dominant_emotion']}</div>
                    <div><strong>Dominant Theme:</strong> {comp['movie_a']['dominant_theme']}</div>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("#### Summary")
            st.markdown(f"*{comp['movie_a']['summary']}*")

        with c_b:
            st.markdown(f"### 🎬 {comp['movie_b']['title']}")
            if rec_b.get("poster_url"):
                st.image(rec_b["poster_url"], width=200)

            genres_b = " ".join([f"<span class='badge badge-accent'>{g}</span>" for g in comp['movie_b']['genres']])
            st.markdown(genres_b, unsafe_allow_html=True)

            st.markdown(f"""
                <div style="background: #1B2229; border: 1px solid #28333E; border-radius: 8px; padding: 14px; margin: 12px 0;">
                    <div><strong>Word Count:</strong> {comp['movie_b']['word_count']} words</div>
                    <div><strong>Sentiment:</strong> {comp['movie_b']['sentiment']} (Compound: {comp['movie_b']['compound']:+.2f})</div>
                    <div><strong>Dominant Emotion:</strong> {comp['movie_b']['dominant_emotion']}</div>
                    <div><strong>Dominant Theme:</strong> {comp['movie_b']['dominant_theme']}</div>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("#### Summary")
            st.markdown(f"*{comp['movie_b']['summary']}*")

        st.markdown("<hr style='border: none; border-top: 1px solid #28333E; margin: 30px 0;'>", unsafe_allow_html=True)

        # Cross Comparison: Shared & Unique Keywords
        st.markdown("### 🔤 Keyword Convergence & Divergence")
        k_c1, k_c2, k_c3 = st.columns(3)

        with k_c1:
            st.markdown("#### Shared Keywords")
            if comp["shared_keywords"]:
                for kw in comp["shared_keywords"]:
                    st.markdown(f"<span class='badge badge-accent'>{kw}</span>", unsafe_allow_html=True)
            else:
                st.markdown("<span style='color: #AAB1B9;'>No direct keyword overlap</span>", unsafe_allow_html=True)

        with k_c2:
            st.markdown(f"#### Unique to {comp['movie_a']['title']}")
            for kw in comp["unique_keywords_a"][:8]:
                st.markdown(f"<span class='badge'>{kw}</span>", unsafe_allow_html=True)

        with k_c3:
            st.markdown(f"#### Unique to {comp['movie_b']['title']}")
            for kw in comp["unique_keywords_b"][:8]:
                st.markdown(f"<span class='badge'>{kw}</span>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Cross Comparison: Themes
        st.markdown("### 🎭 Thematic Convergence")
        t_c1, t_c2 = st.columns(2)
        with t_c1:
            st.markdown("#### Shared Storyline Themes")
            if comp["shared_themes"]:
                for thm in comp["shared_themes"]:
                    st.markdown(f"<span class='badge badge-accent'>{thm}</span>", unsafe_allow_html=True)
            else:
                st.markdown("<span style='color: #AAB1B9;'>Distinct narrative themes (no shared active themes)</span>", unsafe_allow_html=True)

        with t_c2:
            st.markdown("#### Distinct Themes")
            st.markdown(f"**{comp['movie_a']['title']}:** {', '.join(comp['unique_themes_a']) if comp['unique_themes_a'] else 'None'}")
            st.markdown(f"**{comp['movie_b']['title']}:** {', '.join(comp['unique_themes_b']) if comp['unique_themes_b'] else 'None'}")
