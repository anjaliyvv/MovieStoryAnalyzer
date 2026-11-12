"""
Movie Story Analyser - Dataset Management Module
Responsible Team Member: Member 1 (Dataset Engineering & Integration)
Description:
    Handles dynamic dataset ingestion from user-uploaded CSV files.
    Recognizes common column name variations, validates required fields,
    removes duplicates, imputes missing metadata, and computes dataset stats.
"""

import io
import re
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from config import COLUMN_MAPPINGS, REQUIRED_CANONICAL_COLUMNS


class DatasetLoader:
    """Manages loading, schema reconciliation, and validation of movie datasets."""

    def __init__(self):
        self.raw_df: Optional[pd.DataFrame] = None
        self.cleaned_df: Optional[pd.DataFrame] = None
        self.column_mapping: Dict[str, str] = {}
        self.stats: Dict[str, Any] = {}

    def load_from_csv(self, file_or_path: Any) -> Tuple[bool, str, Optional[pd.DataFrame]]:
        """
        Loads CSV from file buffer or file path with encoding fallbacks.
        Returns (success_flag, message, dataframe).
        """
        encodings = ["utf-8", "latin1", "iso-8859-1", "cp1252"]
        df = None
        last_error = ""

        # If file_or_path is BytesIO or file uploader, remember position
        is_stream = hasattr(file_or_path, "seek")

        for enc in encodings:
            try:
                if is_stream:
                    file_or_path.seek(0)
                df = pd.read_csv(file_or_path, encoding=enc)
                break
            except Exception as e:
                last_error = str(e)
                continue

        if df is None:
            return False, f"Failed to read CSV. Error: {last_error}", None

        if df.empty:
            return False, "Uploaded CSV file is completely empty.", None

        self.raw_df = df
        return self._process_and_validate(df)

    def _detect_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """Maps user's arbitrary column names to canonical schema keys."""
        detected = {}
        lower_cols = {str(col).strip().lower(): col for col in df.columns}

        for canonical_key, aliases in COLUMN_MAPPINGS.items():
            for alias in aliases:
                if alias.lower() in lower_cols:
                    detected[canonical_key] = lower_cols[alias.lower()]
                    break

        return detected

    def _extract_year(self, val: Any) -> Optional[int]:
        """Safely parses a 4-digit release year from date strings or numbers."""
        if pd.isna(val):
            return None
        val_str = str(val).strip()
        match = re.search(r"\b(19\d{2}|20\d{2})\b", val_str)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
        return None

    def _clean_genres(self, val: Any) -> List[str]:
        """Normalizes comma, pipe, or bracket-separated genre strings."""
        if pd.isna(val):
            return ["Unknown"]
        val_str = str(val).strip()
        # Handle stringified lists like "['Action', 'Drama']"
        val_str = re.sub(r"[\[\]'\"\{\}]", "", val_str)
        # Split on comma, slash, or pipe
        genres = [g.strip().title() for g in re.split(r"[,/|;]", val_str) if g.strip()]
        return genres if genres else ["Unknown"]

    def _process_and_validate(self, df: pd.DataFrame) -> Tuple[bool, str, Optional[pd.DataFrame]]:
        """Validates required columns, cleans text, deduplicates, and extracts metadata."""
        mapping = self._detect_columns(df)
        self.column_mapping = mapping

        # Check required fields
        missing_required = [
            req for req in REQUIRED_CANONICAL_COLUMNS if req not in mapping
        ]
        if missing_required:
            available_cols = ", ".join([f"'{c}'" for c in df.columns])
            return (
                False,
                f"Missing required column(s): {', '.join(missing_required)}. "
                f"Your file contains: [{available_cols}]. "
                f"Please ensure your CSV has a movie title and a plot/storyline/overview column.",
                None
            )

        # Build standardized working DataFrame
        proc_data = []
        title_col = mapping["title"]
        story_col = mapping["storyline"]

        genre_col = mapping.get("genres")
        year_col = mapping.get("year")
        director_col = mapping.get("director")
        cast_col = mapping.get("cast")
        poster_col = mapping.get("poster_url")

        for idx, row in df.iterrows():
            title_val = row[title_col]
            story_val = row[story_col]

            if pd.isna(title_val) or pd.isna(story_val):
                continue

            title_str = str(title_val).strip()
            story_str = str(story_val).strip()

            # Must have non-trivial storyline length (at least 15 chars)
            if len(title_str) == 0 or len(story_str) < 15:
                continue

            # Year extraction
            year_val = self._extract_year(row[year_col]) if year_col else None

            # Genres extraction
            genre_list = self._clean_genres(row[genre_col]) if genre_col else ["General"]

            # Optional metadata
            director_val = str(row[director_col]).strip() if director_col and not pd.isna(row[director_col]) else "Unknown"
            cast_val = str(row[cast_col]).strip() if cast_col and not pd.isna(row[cast_col]) else "Unknown"

            poster_val = ""
            if poster_col and not pd.isna(row[poster_col]):
                p_str = str(row[poster_col]).strip()
                if p_str.startswith("http://") or p_str.startswith("https://"):
                    poster_val = p_str

            proc_data.append({
                "movie_id": idx,
                "title": title_str,
                "storyline": story_str,
                "storyline_length": len(story_str),
                "word_count": len(story_str.split()),
                "genres": genre_list,
                "primary_genre": genre_list[0] if genre_list else "General",
                "year": year_val,
                "director": director_val,
                "cast": cast_val,
                "poster_url": poster_val
            })

        if not proc_data:
            return (
                False,
                "No valid movie records found. Make sure movie titles and storylines are non-empty text.",
                None
            )

        clean_df = pd.DataFrame(proc_data)

        # Deduplicate on title (case-insensitive)
        clean_df["_lower_title"] = clean_df["title"].str.lower()
        initial_count = len(clean_df)
        clean_df = clean_df.drop_duplicates(subset=["_lower_title"]).drop(columns=["_lower_title"])
        clean_df = clean_df.reset_index(drop=True)
        clean_df["movie_id"] = clean_df.index

        self.cleaned_df = clean_df

        # Compile comprehensive dataset statistics
        all_genres = set()
        for g_list in clean_df["genres"]:
            all_genres.update(g_list)

        years_present = clean_df["year"].dropna()
        min_year = int(years_present.min()) if not years_present.empty else None
        max_year = int(years_present.max()) if not years_present.empty else None

        self.stats = {
            "total_raw_rows": len(df),
            "valid_movies": len(clean_df),
            "duplicates_removed": initial_count - len(clean_df),
            "unique_genres_count": len(all_genres),
            "unique_genres": sorted(list(all_genres)),
            "year_range": (min_year, max_year) if min_year and max_year else ("N/A", "N/A"),
            "avg_storyline_words": int(clean_df["word_count"].mean()) if not clean_df.empty else 0,
            "detected_columns": mapping
        }

        return (
            True,
            f"Successfully processed {len(clean_df)} valid movie records "
            f"from {len(df)} total rows ({self.stats['duplicates_removed']} duplicates removed).",
            clean_df
        )

    def search_and_filter(
        self,
        query: str = "",
        selected_genre: str = "All",
        year_range: Optional[Tuple[int, int]] = None
    ) -> pd.DataFrame:
        """Filters the clean dataset dynamically by search term, genre, and release year."""
        if self.cleaned_df is None or self.cleaned_df.empty:
            return pd.DataFrame()

        filtered = self.cleaned_df.copy()

        # Title / Storyline text search
        if query and query.strip():
            q = query.strip().lower()
            mask = (
                filtered["title"].str.lower().str.contains(q, regex=False) |
                filtered["storyline"].str.lower().str.contains(q, regex=False) |
                filtered["director"].str.lower().str.contains(q, regex=False)
            )
            filtered = filtered[mask]

        # Genre filter
        if selected_genre and selected_genre != "All":
            filtered = filtered[filtered["genres"].apply(lambda g_list: selected_genre in g_list)]

        # Year range filter
        if year_range and year_range[0] is not None and year_range[1] is not None:
            min_y, max_y = year_range
            # Include records with valid years in range, or keep None if no year column exists
            has_year = filtered["year"].notna()
            filtered = filtered[has_year & (filtered["year"] >= min_y) & (filtered["year"] <= max_y)]

        return filtered

    def add_custom_movie(
        self,
        title: str,
        storyline: str,
        genres: Optional[Any] = None,
        year: Optional[int] = None,
        director: str = "Custom / Independent",
        cast: str = "Custom Cast",
        poster_url: str = ""
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """Adds a user-entered custom movie title and plot to the active dataset."""
        title_clean = str(title).strip()
        story_clean = str(storyline).strip()

        if not title_clean:
            return False, "Movie title cannot be empty.", None

        if len(story_clean) < 15:
            return False, "Storyline plot must be at least 15 characters long for meaningful NLP analysis.", None

        # Clean genres
        if genres is None:
            genre_list = ["Drama"]
        elif isinstance(genres, list):
            genre_list = genres if genres else ["Drama"]
        else:
            genre_list = self._clean_genres(genres)

        # Initialize cleaned_df if None
        if self.cleaned_df is None or self.cleaned_df.empty:
            new_id = 0
            new_rows = []
        else:
            new_id = len(self.cleaned_df)
            new_rows = self.cleaned_df.to_dict("records")

        # Check if movie with same title already exists
        for idx, row in enumerate(new_rows):
            if row["title"].strip().lower() == title_clean.lower():
                # Update existing movie record
                new_rows[idx]["storyline"] = story_clean
                new_rows[idx]["storyline_length"] = len(story_clean)
                new_rows[idx]["word_count"] = len(story_clean.split())
                new_rows[idx]["genres"] = genre_list
                new_rows[idx]["primary_genre"] = genre_list[0] if genre_list else "General"
                if year:
                    new_rows[idx]["year"] = year
                if director:
                    new_rows[idx]["director"] = director
                if cast:
                    new_rows[idx]["cast"] = cast
                if poster_url:
                    new_rows[idx]["poster_url"] = poster_url

                self.cleaned_df = pd.DataFrame(new_rows)
                self._update_stats()
                return True, f"Updated existing movie record for '{title_clean}'.", new_rows[idx]

        # Add new movie
        new_record = {
            "movie_id": new_id,
            "title": title_clean,
            "storyline": story_clean,
            "storyline_length": len(story_clean),
            "word_count": len(story_clean.split()),
            "genres": genre_list,
            "primary_genre": genre_list[0] if genre_list else "General",
            "year": int(year) if year else None,
            "director": str(director).strip() if director else "Custom / Independent",
            "cast": str(cast).strip() if cast else "Custom Cast",
            "poster_url": str(poster_url).strip() if poster_url else ""
        }

        new_rows.append(new_record)
        self.cleaned_df = pd.DataFrame(new_rows)
        self._update_stats()

        return True, f"Custom movie '{title_clean}' successfully added to dataset!", new_record

    def _update_stats(self):
        """Recalculates summary stats after adding or updating movies."""
        if self.cleaned_df is None or self.cleaned_df.empty:
            return

        all_genres = set()
        for g_list in self.cleaned_df["genres"]:
            all_genres.update(g_list)

        years_present = self.cleaned_df["year"].dropna()
        min_year = int(years_present.min()) if not years_present.empty else None
        max_year = int(years_present.max()) if not years_present.empty else None

        self.stats = {
            "total_raw_rows": len(self.cleaned_df),
            "valid_movies": len(self.cleaned_df),
            "duplicates_removed": self.stats.get("duplicates_removed", 0) if self.stats else 0,
            "unique_genres_count": len(all_genres),
            "unique_genres": sorted(list(all_genres)),
            "year_range": (min_year, max_year) if min_year and max_year else ("N/A", "N/A"),
            "avg_storyline_words": int(self.cleaned_df["word_count"].mean()) if not self.cleaned_df.empty else 0,
            "detected_columns": self.stats.get("detected_columns", {}) if self.stats else {}
        }
