# Movie Story Analyser
## Dataset-Based NLP Application Using Python and Streamlit

**B.Tech CSE AIML Group Project**

An interactive, dataset-driven Natural Language Processing (NLP) web application for analysing movie storylines. The application supports dynamic CSV uploads, automatic schema reconciliation, extractive storyline summarization using TextRank, explainable VADER sentiment analysis with sentence-level attribution, 7-dimension narrative emotion profiling, TF-IDF unigram and bigram extraction, rule-grounded thematic clustering, and dual-movie comparative analytics.

---

## 1. Five-Member Team Roles & Responsibilities

| Member | Role | Key Contributions |
| :--- | :--- | :--- |
| **Member 1** | **Dataset & Custom Movie Ingestion** | Implemented `modules/dataset_loader.py`. Built direct custom movie entry (title & plot input), automated CSV schema detection, column normalization, deduplication, missing-value handling, and dataset summary statistics. |
| **Member 2** | **NLP Preprocessing & Summarization** | Implemented `modules/preprocessing.py` and `modules/summarizer.py`. Developed multi-stage text processing (tokenization, stopword removal, lemmatization) and graph-based extractive TextRank summarization via NetworkX and PageRank. |
| **Member 3** | **Sentiment Analysis & Emotion Detection** | Implemented `modules/sentiment.py` and `modules/emotion.py`. Developed explainable VADER sentiment analysis, sentence-level polarity scoring, dramatic conflict diagnostics, and 7-dimension narrative emotion scoring. |
| **Member 4** | **Keywords, Themes & Comparison** | Implemented `modules/keywords.py`, `modules/themes.py`, and `modules/comparison.py`. Engineered TF-IDF unigram/bigram extraction, 10-archetype storyline theme detection with sentence proof, and vector-space cosine similarity dual comparison. |
| **Member 5** | **UI/UX Architecture & Streamlit Integration** | Implemented `app.py`, `config.py`, and `assets/style.css`. Created the dark cinematic UI theme (`#11161B`, `#C9A47F`), multi-page navigation, session-state management, Plotly visualization charts, and end-to-end integration testing. |

---

## 2. Project Architecture

```
MovieStoryAnalyser/
├── app.py                     # Main Streamlit web application & multi-page router
├── config.py                  # Global configurations, theme colors, column mappings
├── requirements.txt           # Python dependencies
├── README.md                  # Complete documentation and execution guide
├── test_modules.py            # Automated unit testing suite
├── assets/
│   └── style.css              # Custom dark cinematic Streamlit styling
├── modules/
│   ├── dataset_loader.py      # Member 1: Dataset ingestion, validation, and filtering
│   ├── preprocessing.py       # Member 2: Tokenization, stopword removal, lemmatization
│   ├── summarizer.py          # Member 2: TextRank graph summarizer with PageRank
│   ├── sentiment.py           # Member 3: Explainable VADER sentiment analysis
│   ├── emotion.py             # Member 3: 7-dimension emotion profiling & Plotly charts
│   ├── keywords.py            # Member 4: TF-IDF unigram & bigram extraction
│   ├── themes.py              # Member 4: 10 narrative theme detection with evidence
│   └── comparison.py          # Member 4: Dual movie cosine similarity & cross-metrics
├── sample_data/
│   └── movies_sample.csv      # Curated multi-genre sample dataset (25+ films)
└── uploads/                   # Runtime directory for user-uploaded CSV files
```

---

## 3. Explanation of NLP Algorithms

### A. Extractive Summarization (TextRank + PageRank)
- **Algorithm**: Graph-based ranking inspired by Google PageRank.
- **Formulation**: Each sentence forms a vertex $V_i$ in graph $G = (V, E)$. The edge weight $w_{ij}$ corresponds to the normalized cosine similarity between sentence vectors $s_i$ and $s_j$.
- **PageRank Formula**:
  $$S(V_i) = (1 - d) + d \sum_{V_j \in In(V_i)} \frac{w_{ji}}{\sum_{V_k \in Out(V_j)} w_{jk}} S(V_j)$$
  where damping factor $d = 0.85$.
- **Outcome**: Sentences that share significant vocabulary with other central plot events receive higher centrality scores. The top sentences are extracted and arranged chronologically.

### B. Explainable Sentiment Analysis (VADER)
- **Algorithm**: Valence Aware Dictionary and sEntiment Reasoner (VADER).
- **Sentence Attribution**: Evaluates punctuation intensity, booster words (e.g., *very*, *extremely*), and 3-token negation windows (*not*, *never*).
- **Compound Polarity Formula**:
  $$\text{Compound} = \frac{\sum x}{\sqrt{(\sum x)^2 + \alpha}}$$
  where $x$ represents individual valence scores and $\alpha = 15$.
- **Narrative Explainability**: Storyline sentiment reflects plot conflict, tragedy, and peril, NOT audience reception. A film like *The Dark Knight* has negative storyline polarity due to crime, chaos, and death, while being an acclaimed cinematic masterpiece.

### C. Narrative Emotion Detection (7 Dimensions)
- Evaluates **Happiness, Sadness, Fear, Anger, Surprise, Love, Hope**.
- Employs an empirical emotion lexicon mapping keywords to intensity scores.
- Avoids opaque black-box hallucination by tracking and displaying the exact keywords responsible for triggering each emotion dimension.

### D. Keyword & Keyphrase Extraction (TF-IDF)
- Extracts both single words (unigrams) and 2-word phrases (bigrams) to preserve cinematic collocations (e.g., *"dream state"*, *"criminal underworld"*).
- Uses sublinear term frequency: $\text{TF}(t, d) = 1 + \log(\text{tf}(t, d))$ for $\text{tf} > 0$.
- Uses smoothed inverse document frequency: $\text{IDF}(t) = \log\left(\frac{N + 1}{\text{df}(t) + 1}\right) + 1$.

### E. Thematic Motif Detection
- Classifies storylines across 10 archetypes: **Friendship, Love, Revenge, Survival, Family, Mystery, Adventure, Betrayal, Sacrifice, Social issues**.
- Computes theme strength based on semantic cluster hits and presents transparent sentence-level quotations demonstrating where the theme manifests.

### F. Vector Space Dual Comparison (Cosine Similarity)
- Represents two movie storylines as TF-IDF vectors $\vec{A}$ and $\vec{B}$ across the union of their vocabulary.
- Computes cosine similarity:
  $$\text{Similarity}(\vec{A}, \vec{B}) = \frac{\vec{A} \cdot \vec{B}}{\|\vec{A}\| \|\vec{B}\|}$$
- Dynamically derives shared keywords, unique keywords, and comparative emotion/sentiment deltas.

---

## 4. Google Colab Execution Instructions

Follow these exact steps to run the complete application inside a free Google Colab instance:

### Step 1: Open Google Colab
Go to [colab.research.google.com](https://colab.research.google.com) and create a **New Notebook**.

### Step 2: Install Required Dependencies
In the first code cell, execute:
```python
!pip install streamlit pandas numpy nltk scikit-learn networkx plotly spacy pyngrok -q
!pip install localtunnel -q
```

### Step 3: Clone / Download Application Files
In the second code cell, clone your project repository or create the files:
```python
# Create project directories
!mkdir -p MovieStoryAnalyser/modules MovieStoryAnalyser/assets MovieStoryAnalyser/sample_data MovieStoryAnalyser/uploads

# Run verification test
!python MovieStoryAnalyser/test_modules.py
```

### Step 4: Launch Streamlit with LocalTunnel
In the third code cell, run Streamlit in the background and expose it via a public tunnel:
```python
# Get public IP for LocalTunnel password
!curl ipv4.icanhazip.com

# Start Streamlit via localtunnel
!streamlit run MovieStoryAnalyser/app.py & npx localtunnel --port 8501
```
Click the generated `localtunnel.me` URL, paste your IP from `icanhazip.com` as the tunnel password, and the web application will open immediately in your browser!

---

## 5. Local Execution (Windows & VS Code)

### Prerequisites
- Python 3.9, 3.10, or 3.11 installed on your system.
- VS Code or any modern code editor.

### Installation & Run Steps
1. **Open Terminal in the project root directory:**
   ```bash
   cd MovieStoryAnalyser
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run automated module test suite:**
   ```bash
   python test_modules.py
   ```

5. **Start the Streamlit application:**
   ```bash
   streamlit run app.py
   ```
   The application will automatically launch in your default browser at `http://localhost:8501`.

---

## 6. Dataset Format & Guidelines

The application accepts any standard CSV file. The columns are automatically mapped without requiring manual schema edits:

| Field | Accepted CSV Header Names (Case-Insensitive) | Required? |
| :--- | :--- | :--- |
| **Movie Title** | `title`, `movie_title`, `original_title`, `film_title`, `name` | **Yes** |
| **Storyline** | `overview`, `plot`, `description`, `storyline`, `synopsis`, `summary` | **Yes** |
| **Genres** | `genres`, `genre`, `category`, `genres_list` | Optional |
| **Release Year**| `year`, `release_year`, `release_date`, `date` | Optional |
| **Director** | `director`, `directors`, `directed_by` | Optional |
| **Cast** | `cast`, `actors`, `starring`, `lead_actors` | Optional |
| **Poster URL** | `poster_url`, `poster_path`, `poster`, `image_url`, `image` | Optional |

---

## 7. Verification & Testing

To run the automated validation suite on all 5 member components:
```bash
python test_modules.py
```
Output:
```
============================================================
Movie Story Analyser - B.Tech CSE AIML Verification Suite
============================================================
Testing Member 1: DatasetLoader...
  [PASS] Successfully loaded 15 movies. Stats: 15 valid records.
Testing Member 2: Preprocessing & TextRank Summarizer...
  [PASS] TextRank successfully generated short and detailed summaries.
Testing Member 3: VADER Sentiment & Emotion Detection...
  [PASS] Sentiment: Positive (compound: +0.84), Dominant Emotion: Hope
Testing Member 4: Keywords, Themes, and Comparison...
  [PASS] Extracted 5 keywords. Comparison similarity: 24.8%
============================================================
ALL MODULE TESTS PASSED SUCCESSFULLY! Ready for deployment.
============================================================
```
