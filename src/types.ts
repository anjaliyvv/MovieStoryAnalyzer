/**
 * Movie Story Analyser - Type Definitions
 */

export interface Movie {
  movie_id: number;
  title: string;
  storyline: string;
  storyline_length: number;
  word_count: number;
  genres: string[];
  primary_genre: string;
  year?: number | null;
  director: string;
  cast: string;
  poster_url?: string;
}

export interface DatasetStats {
  total_raw_rows: number;
  valid_movies: number;
  duplicates_removed: number;
  unique_genres_count: number;
  unique_genres: string[];
  year_range: [number | string, number | string];
  avg_storyline_words: number;
  detected_columns: Record<string, string>;
}

export interface PreprocessingResult {
  raw_text: string;
  sentences: string[];
  tokens: string[];
  cleaned_tokens: string[];
  filtered_tokens: string[];
  lemmas: string[];
  processed_text_string: string;
  stats: {
    raw_chars: number;
    sentence_count: number;
    token_count: number;
    filtered_token_count: number;
    vocab_size: number;
  };
}

export interface RankedSentence {
  index: number;
  rank: number;
  sentence: string;
  score: number;
}

export interface SummaryResult {
  short_summary: string;
  detailed_summary: string;
  sentence_count: number;
  selected_count: number;
  ranked_sentences: RankedSentence[];
  method: string;
}

export interface SentenceSentiment {
  sentence: string;
  compound: number;
  sentiment: 'Positive' | 'Negative' | 'Neutral';
  pos_words: [string, number][];
  neg_words: [string, number][];
}

export interface SentimentResult {
  overall_sentiment: 'Positive' | 'Negative' | 'Neutral' | 'Mixed';
  compound_score: number;
  counts: {
    Positive: number;
    Negative: number;
    Neutral: number;
    Total: number;
  };
  proportions: {
    pos: number;
    neg: number;
    neu: number;
  };
  sentences: SentenceSentiment[];
  key_neg_sentences: SentenceSentiment[];
  key_pos_sentences: SentenceSentiment[];
  contributing_neg_words: [string, number][];
  contributing_pos_words: [string, number][];
  explanation: string;
  method: string;
}

export interface EmotionResult {
  raw_scores: Record<string, number>;
  distribution: Record<string, number>;
  percentages: Record<string, number>;
  dominant_emotion: string;
  secondary_emotion: string;
  evidence: Record<string, [string, number][]>;
  total_emotion_markers: number;
  method: string;
}

export interface KeywordItem {
  keyword: string;
  type: 'Unigram' | 'Bigram';
  score: number;
  frequency: number;
}

export interface KeywordsResult {
  keywords: KeywordItem[];
  unigrams: KeywordItem[];
  bigrams: KeywordItem[];
  total_extracted: number;
  method: string;
}

export interface DetectedTheme {
  theme: string;
  score: number;
  percentage: number;
  description: string;
  keywords_found: string[];
  supporting_sentences: string[];
}

export interface ThemesResult {
  themes: DetectedTheme[];
  all_scores: Record<string, number>;
  dominant_theme: string;
  total_theme_markers: number;
  method: string;
}

export interface ComparisonResult {
  similarity_score: number;
  similarity_percentage: number;
  movie_a: {
    title: string;
    year: number | string;
    genres: string[];
    word_count: number;
    summary: string;
    sentiment: string;
    compound: number;
    dominant_emotion: string;
    dominant_theme: string;
    top_keywords: string[];
    emotions: Record<string, number>;
    theme_names: string[];
  };
  movie_b: {
    title: string;
    year: number | string;
    genres: string[];
    word_count: number;
    summary: string;
    sentiment: string;
    compound: number;
    dominant_emotion: string;
    dominant_theme: string;
    top_keywords: string[];
    emotions: Record<string, number>;
    theme_names: string[];
  };
  shared_keywords: string[];
  unique_keywords_a: string[];
  unique_keywords_b: string[];
  shared_themes: string[];
  unique_themes_a: string[];
  unique_themes_b: string[];
}
