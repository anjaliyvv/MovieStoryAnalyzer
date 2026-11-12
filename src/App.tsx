/**
 * Movie Story Analyser - Streamlit Dark Cinematic Web Application
 * B.Tech CSE AIML Group Project (5 Members)
 */

import React, { useState, useEffect, useMemo, useRef } from 'react';
import {
  Film,
  Compass,
  FileText,
  Heart,
  Tag,
  Scale,
  Upload,
  RefreshCw,
  Search,
  Sliders,
  ChevronRight,
  CheckCircle2,
  AlertCircle,
  BarChart3,
  Layers,
  Sparkles,
  BookOpen,
  Users,
  PenTool,
  PlusCircle,
  Wand2
} from 'lucide-react';
import {
  Movie,
  DatasetStats,
  SummaryResult,
  SentimentResult,
  EmotionResult,
  KeywordsResult,
  ThemesResult,
  ComparisonResult
} from './types';
import {
  parseCSV,
  SAMPLE_MOVIES_CSV,
  preprocessText,
  textrankSummarize,
  analyzeStorylineSentiment,
  detectEmotions,
  extractKeywordsTfidf,
  detectThemes,
  compareTwoMovies,
  addCustomMovieToCollection
} from './nlpEngine';

export default function App() {
  // Navigation
  const [navPage, setNavPage] = useState<string>('Custom Movie');

  // Dataset State
  const [movies, setMovies] = useState<Movie[]>([]);
  const [stats, setStats] = useState<DatasetStats | null>(null);
  const [selectedMovieId, setSelectedMovieId] = useState<number | null>(null);
  const [compareAId, setCompareAId] = useState<number | null>(null);
  const [compareBId, setCompareBId] = useState<number | null>(null);
  const [uploadMessage, setUploadMessage] = useState<string>('');
  const [uploadError, setUploadError] = useState<boolean>(false);

  // Custom Movie Form State
  const [customTitle, setCustomTitle] = useState<string>('');
  const [customPlot, setCustomPlot] = useState<string>('');
  const [customGenre, setCustomGenre] = useState<string>('Sci-Fi');
  const [customYear, setCustomYear] = useState<string>('2025');
  const [customDirector, setCustomDirector] = useState<string>('');
  const [customCast, setCustomCast] = useState<string>('');
  const [customPoster, setCustomPoster] = useState<string>('');
  const [customNotice, setCustomNotice] = useState<string>('');
  const [customError, setCustomError] = useState<string>('');

  // Search & Filter State
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [selectedGenre, setSelectedGenre] = useState<string>('All');
  const [summarySentencesCount, setSummarySentencesCount] = useState<number>(3);
  const [activePrepTab, setActivePrepTab] = useState<number>(0);

  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // Load initial sample dataset
  useEffect(() => {
    const res = parseCSV(SAMPLE_MOVIES_CSV);
    if (res.success && res.movies.length > 0) {
      setMovies(res.movies);
      setStats(res.stats || null);
      setSelectedMovieId(res.movies[0].movie_id);
      if (res.movies.length > 1) {
        setCompareAId(res.movies[0].movie_id);
        setCompareBId(res.movies[1].movie_id);
      }
    }
  }, []);

  // Handle adding custom movie
  const handleAddCustomMovie = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    setCustomError('');
    setCustomNotice('');

    if (!customTitle.trim()) {
      setCustomError('Please enter a movie title.');
      return;
    }

    if (customPlot.trim().length < 15) {
      setCustomError('Please enter at least 15 characters of storyline or plot synopsis.');
      return;
    }

    const yr = customYear ? parseInt(customYear, 10) : null;
    const res = addCustomMovieToCollection(
      movies,
      customTitle,
      customPlot,
      [customGenre],
      isNaN(yr as number) ? null : yr,
      customDirector || 'Independent Director',
      customCast || 'Custom Cast',
      customPoster
    );

    if (res.success) {
      setMovies(res.movies);
      setStats(res.stats);
      setSelectedMovieId(res.newMovie.movie_id);
      setCustomNotice(`Custom movie '${res.newMovie.title}' created and selected for NLP analysis!`);
      // Navigate to Story Analysis to show results immediately
      setNavPage('Story Analysis');
    }
  };

  // Preset Template loader for quick testing
  const loadPresetTemplate = (type: 'scifi' | 'crime' | 'drama') => {
    if (type === 'scifi') {
      setCustomTitle('Aethelgard: The Last Frequency');
      setCustomGenre('Sci-Fi');
      setCustomYear('2026');
      setCustomDirector('Elena Vance');
      setCustomPlot(
        'In deep orbit beyond Neptune, an isolated communications outpost intercepts an anomalous synthetic signal echoing from an uncharted void. Chief engineer Maya Lin soon discovers that the transmission carries the memories and voices of astronauts lost thirty years ago. As gravitational disturbances warp reality within the station, Maya realizes the entity is a sentient cosmic intelligence seeking refuge from an encroaching stellar collapse. With oxygen failing and her crew descending into hallucinations, Maya must decide whether to transmit the coordinates back to Earth or sacrifice herself to preserve the alien entity.'
      );
    } else if (type === 'crime') {
      setCustomTitle('Shadows of Manhattan');
      setCustomGenre('Crime');
      setCustomYear('2024');
      setCustomDirector('Marcus Bell');
      setCustomPlot(
        'Veteran detective David Cole is forced to investigate a series of calculated bank heists orchestrating the systematic collapse of corrupt financial oligarchs. As Cole unravels the tactical brilliance behind each robbery, he discovers the mastermind is his former police partner who was betrayed and framed by the department five years ago. Torn between personal loyalty and upholding the law, Cole is thrust into a high-stakes psychological chess game across the neon streets of New York, where vengeance and systemic corruption collide in a deadly showdown.'
      );
    } else {
      setCustomTitle('The Horizon Between Us');
      setCustomGenre('Drama');
      setCustomYear('2025');
      setCustomDirector('Sophia Laurent');
      setCustomPlot(
        'Separated by war and geopolitical borders, two classical musicians, Julian and Clara, communicate exclusively through secret handwritten sheet music smuggled across checkpoints. Over twelve tumultuous years, their encoded musical letters chronicle love, tragedy, resilience, and the relentless desire for freedom. When an unexpected political thaw finally opens the border, Julian embarks on a perilous journey across the mountains to find Clara before her fading memory surrenders to dementia, proving that the language of love transcends physical boundaries.'
      );
    }
  };

  // Handle CSV file upload
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = event => {
      const content = event.target?.result as string;
      const res = parseCSV(content);
      if (res.success && res.movies.length > 0) {
        setMovies(res.movies);
        setStats(res.stats || null);
        setSelectedMovieId(res.movies[0].movie_id);
        if (res.movies.length > 1) {
          setCompareAId(res.movies[0].movie_id);
          setCompareBId(res.movies[1].movie_id);
        }
        setUploadMessage(res.message);
        setUploadError(false);
      } else {
        setUploadMessage(res.message);
        setUploadError(true);
      }
    };
    reader.readAsText(file);
  };

  // Reload built-in sample data
  const handleReloadSample = () => {
    const res = parseCSV(SAMPLE_MOVIES_CSV);
    if (res.success) {
      setMovies(res.movies);
      setStats(res.stats || null);
      setSelectedMovieId(res.movies[0].movie_id);
      if (res.movies.length > 1) {
        setCompareAId(res.movies[0].movie_id);
        setCompareBId(res.movies[1].movie_id);
      }
      setUploadMessage('Curated sample dataset loaded successfully!');
      setUploadError(false);
    }
  };

  // Current selected movie object
  const selectedMovie = useMemo(() => {
    if (!movies.length) return null;
    return movies.find(m => m.movie_id === selectedMovieId) || movies[0];
  }, [movies, selectedMovieId]);

  // NLP Analytics computed for selected movie
  const analysis = useMemo(() => {
    if (!selectedMovie) return null;
    const text = selectedMovie.storyline;
    return {
      preprocessing: preprocessText(text),
      summary: textrankSummarize(text, summarySentencesCount),
      sentiment: analyzeStorylineSentiment(text),
      emotion: detectEmotions(text),
      keywords: extractKeywordsTfidf(text, 10),
      themes: detectThemes(text)
    };
  }, [selectedMovie, summarySentencesCount]);

  // Dual Comparison Data
  const comparison = useMemo(() => {
    if (compareAId === null || compareBId === null || movies.length < 2) return null;
    const movieA = movies.find(m => m.movie_id === compareAId);
    const movieB = movies.find(m => m.movie_id === compareBId);
    if (!movieA || !movieB) return null;
    return compareTwoMovies(movieA, movieB);
  }, [movies, compareAId, compareBId]);

  // Filtered movies in Movie Explorer
  const filteredMovies = useMemo(() => {
    return movies.filter(m => {
      const q = searchQuery.toLowerCase().trim();
      const matchQuery =
        !q ||
        m.title.toLowerCase().includes(q) ||
        m.storyline.toLowerCase().includes(q) ||
        m.director.toLowerCase().includes(q);

      const matchGenre = selectedGenre === 'All' || m.genres.includes(selectedGenre);
      return matchQuery && matchGenre;
    });
  }, [movies, searchQuery, selectedGenre]);

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[#11161B] text-[#F1EEE8] font-sans antialiased selection:bg-[#C9A47F] selection:text-[#11161B]">
      {/* ----------------------------------------------------------------- */}
      {/* SIDEBAR NAVIGATION                                                */}
      {/* ----------------------------------------------------------------- */}
      <aside className="w-80 flex-shrink-0 bg-[#1B2229] border-r border-[#28333E] flex flex-col justify-between overflow-y-auto">
        <div className="p-5">
          {/* Brand Header */}
          <div className="pb-5 border-b border-[#28333E] mb-5">
            <div className="flex items-center gap-2.5">
              <div className="w-9 h-9 rounded-lg bg-[#C9A47F] text-[#11161B] flex items-center justify-center font-bold text-lg shadow-md">
                🎬
              </div>
              <div>
                <h1 className="text-lg font-bold tracking-tight text-[#C9A47F] leading-tight">
                  Movie Story Analyser
                </h1>
                <p className="text-xs text-[#AAB1B9]">Dataset-Based NLP Intelligence</p>
              </div>
            </div>
          </div>

          {/* Navigation Menu */}
          <div className="space-y-1.5 mb-6">
            {[
              { id: 'Custom Movie', label: '✍️ My Custom Movie', icon: PenTool, highlight: true },
              { id: 'Dashboard', label: 'Dashboard', icon: BarChart3 },
              { id: 'Movie Explorer', label: 'Movie Explorer', icon: Compass },
              { id: 'Story Analysis', label: 'Story Analysis', icon: BookOpen },
              { id: 'Sentiment and Emotions', label: 'Sentiment & Emotions', icon: Heart },
              { id: 'Keywords and Themes', label: 'Keywords & Themes', icon: Tag },
              { id: 'Compare Movies', label: 'Compare Movies', icon: Scale }
            ].map(item => {
              const Icon = item.icon;
              const active = navPage === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => setNavPage(item.id)}
                  className={`w-full flex items-center gap-3 px-3.5 py-2.5 rounded-lg text-sm font-medium transition-all text-left ${
                    active
                      ? 'bg-[#161D24] text-[#C9A47F] border border-[#C9A47F]/40 shadow-sm'
                      : item.highlight
                      ? 'bg-[#C9A47F]/10 text-[#C9A47F] hover:bg-[#C9A47F]/20 border border-[#C9A47F]/30'
                      : 'text-[#AAB1B9] hover:bg-[#161D24]/60 hover:text-[#F1EEE8]'
                  }`}
                >
                  <Icon className={`w-4 h-4 ${active || item.highlight ? 'text-[#C9A47F]' : 'text-[#AAB1B9]'}`} />
                  <span>{item.label}</span>
                </button>
              );
            })}
          </div>

          <hr className="border-[#28333E] mb-5" />

          {/* Quick Custom Movie Form in Sidebar */}
          <div className="bg-[#161D24] border border-[#28333E] rounded-xl p-4 space-y-3 mb-5">
            <div className="flex items-center gap-1.5 text-xs font-bold text-[#C9A47F] uppercase tracking-wider">
              <Sparkles className="w-3.5 h-3.5" />
              Quick Custom Movie
            </div>
            <input
              type="text"
              placeholder="Movie title..."
              value={customTitle}
              onChange={e => setCustomTitle(e.target.value)}
              className="w-full bg-[#11161B] border border-[#28333E] rounded-md px-3 py-1.5 text-xs text-[#F1EEE8] placeholder-[#AAB1B9]/60 focus:outline-none focus:border-[#C9A47F]"
            />
            <textarea
              placeholder="Paste plot or storyline synopsis..."
              value={customPlot}
              onChange={e => setCustomPlot(e.target.value)}
              rows={3}
              className="w-full bg-[#11161B] border border-[#28333E] rounded-md px-3 py-1.5 text-xs text-[#F1EEE8] placeholder-[#AAB1B9]/60 focus:outline-none focus:border-[#C9A47F] resize-none"
            />
            <button
              onClick={() => handleAddCustomMovie()}
              className="w-full bg-[#C9A47F] hover:bg-[#8D7058] text-[#11161B] hover:text-[#F1EEE8] text-xs font-bold py-2 rounded-md transition-all shadow-sm flex items-center justify-center gap-1.5"
            >
              <Sparkles className="w-3 h-3" />
              <span>Analyze Storyline</span>
            </button>
          </div>

          {/* Optional CSV Ingestion Section */}
          <details className="text-xs text-[#AAB1B9] space-y-2 mb-4">
            <summary className="cursor-pointer font-medium text-[#AAB1B9] hover:text-[#C9A47F] flex items-center justify-between py-1">
              <span>Optional: CSV Dataset Ingestion</span>
              <Upload className="w-3 h-3" />
            </summary>
            <div className="pt-2 space-y-2">
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileUpload}
                accept=".csv"
                className="hidden"
              />
              <button
                onClick={() => fileInputRef.current?.click()}
                className="w-full flex items-center justify-center gap-2 bg-[#161D24] hover:bg-[#202933] border border-[#28333E] text-[#F1EEE8] text-xs py-2 px-3 rounded-lg transition-all"
              >
                <Upload className="w-3 h-3 text-[#C9A47F]" />
                <span>Upload CSV Dataset</span>
              </button>
              <button
                onClick={handleReloadSample}
                className="w-full flex items-center justify-center gap-1.5 bg-[#161D24]/50 hover:bg-[#161D24] border border-[#28333E] text-[#AAB1B9] hover:text-[#F1EEE8] text-xs py-1.5 px-3 rounded-lg transition-all"
              >
                <RefreshCw className="w-2.5 h-2.5 text-[#8D7058]" />
                <span>Reset Built-in Movies</span>
              </button>
              {uploadMessage && (
                <div
                  className={`text-[11px] p-2 rounded border ${
                    uploadError
                      ? 'bg-[#D95D5D]/10 border-[#D95D5D]/30 text-[#D95D5D]'
                      : 'bg-[#4E9F76]/10 border-[#4E9F76]/30 text-[#4E9F76]'
                  }`}
                >
                  {uploadMessage}
                </div>
              )}
            </div>
          </details>

          {/* Quick Stats */}
          {stats && (
            <div className="bg-[#161D24] border border-[#28333E] rounded-lg p-3 text-xs space-y-1.5">
              <div className="text-[11px] font-semibold text-[#C9A47F] uppercase tracking-wider mb-1">
                Active Dataset
              </div>
              <div className="flex justify-between text-[#AAB1B9]">
                <span>Total Movies:</span>
                <span className="text-[#F1EEE8] font-mono font-medium">{stats.valid_movies}</span>
              </div>
              <div className="flex justify-between text-[#AAB1B9]">
                <span>Selected:</span>
                <span className="text-[#C9A47F] font-semibold truncate max-w-[120px]">
                  {selectedMovie?.title || 'None'}
                </span>
              </div>
            </div>
          )}
        </div>

        {/* Sidebar Footer */}
        <div className="p-4 border-t border-[#28333E] bg-[#161D24]/40 text-center">
          <div className="text-[11px] text-[#AAB1B9] font-medium">B.Tech CSE AIML Group 5</div>
          <div className="text-[10px] text-[#8D7058] mt-0.5">Streamlit & NLP Architecture</div>
        </div>
      </aside>

      {/* ----------------------------------------------------------------- */}
      {/* MAIN CONTENT AREA                                                 */}
      {/* ----------------------------------------------------------------- */}
      <main className="flex-1 h-screen overflow-y-auto bg-[#11161B] p-6 lg:p-8">
        {/* =============================================================== */}
        {/* PAGE: MY CUSTOM MOVIE (FIRST-CLASS FEATURE)                     */}
        {/* =============================================================== */}
        {navPage === 'Custom Movie' && (
          <div className="max-w-4xl mx-auto space-y-6">
            {/* Header */}
            <div className="space-y-1">
              <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-[#C9A47F]">
                <PenTool className="w-4 h-4" />
                Custom Movie Storyline Analyzer
              </div>
              <h2 className="text-3xl font-extrabold text-[#F1EEE8] tracking-tight">
                Enter Your Custom Movie
              </h2>
              <p className="text-sm text-[#AAB1B9]">
                Enter any movie title and storyline plot below to immediately run TextRank extractive
                summarization, explainable VADER sentiment analysis, 7-dimension emotion profiling,
                and thematic motif detection.
              </p>
            </div>

            {/* Error / Notice Banners */}
            {customError && (
              <div className="bg-[#D95D5D]/10 border border-[#D95D5D]/40 text-[#D95D5D] p-3.5 rounded-xl text-xs flex items-center gap-2">
                <AlertCircle className="w-4 h-4 flex-shrink-0" />
                <span>{customError}</span>
              </div>
            )}
            {customNotice && (
              <div className="bg-[#4E9F76]/10 border border-[#4E9F76]/40 text-[#4E9F76] p-3.5 rounded-xl text-xs flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 flex-shrink-0" />
                <span>{customNotice}</span>
              </div>
            )}

            {/* Quick Preset Buttons */}
            <div className="bg-[#161D24] border border-[#28333E] rounded-xl p-4 flex flex-wrap items-center justify-between gap-3">
              <div className="flex items-center gap-2 text-xs text-[#AAB1B9]">
                <Wand2 className="w-3.5 h-3.5 text-[#C9A47F]" />
                <span>Load Sample Custom Plot:</span>
              </div>
              <div className="flex flex-wrap gap-2">
                <button
                  type="button"
                  onClick={() => loadPresetTemplate('scifi')}
                  className="px-3 py-1.5 rounded-lg text-xs bg-[#1B2229] border border-[#28333E] hover:border-[#C9A47F] text-[#F1EEE8] hover:text-[#C9A47F] transition-all"
                >
                  🛸 Sci-Fi Mystery
                </button>
                <button
                  type="button"
                  onClick={() => loadPresetTemplate('crime')}
                  className="px-3 py-1.5 rounded-lg text-xs bg-[#1B2229] border border-[#28333E] hover:border-[#C9A47F] text-[#F1EEE8] hover:text-[#C9A47F] transition-all"
                >
                  🕵️ Crime Thriller
                </button>
                <button
                  type="button"
                  onClick={() => loadPresetTemplate('drama')}
                  className="px-3 py-1.5 rounded-lg text-xs bg-[#1B2229] border border-[#28333E] hover:border-[#C9A47F] text-[#F1EEE8] hover:text-[#C9A47F] transition-all"
                >
                  🎻 Emotional Drama
                </button>
              </div>
            </div>

            {/* Custom Movie Input Form */}
            <form
              onSubmit={handleAddCustomMovie}
              className="bg-[#161D24] border border-[#28333E] rounded-2xl p-6 sm:p-8 space-y-6 shadow-xl"
            >
              {/* Row 1: Title and Genre */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <div className="sm:col-span-2 space-y-1.5">
                  <label className="text-xs font-bold text-[#F1EEE8] uppercase tracking-wider flex items-center gap-1.5">
                    <span>Movie Title</span>
                    <span className="text-[#C9A47F]">*</span>
                  </label>
                  <input
                    type="text"
                    required
                    placeholder="e.g. Inception: Dream World / My Story Project"
                    value={customTitle}
                    onChange={e => setCustomTitle(e.target.value)}
                    className="w-full bg-[#11161B] border border-[#28333E] focus:border-[#C9A47F] rounded-xl px-4 py-2.5 text-sm text-[#F1EEE8] placeholder-[#AAB1B9]/50 focus:outline-none transition-all shadow-inner"
                  />
                </div>

                <div className="space-y-1.5">
                  <label className="text-xs font-bold text-[#F1EEE8] uppercase tracking-wider">
                    Genre
                  </label>
                  <select
                    value={customGenre}
                    onChange={e => setCustomGenre(e.target.value)}
                    className="w-full bg-[#11161B] border border-[#28333E] focus:border-[#C9A47F] rounded-xl px-3 py-2.5 text-sm text-[#F1EEE8] focus:outline-none transition-all cursor-pointer"
                  >
                    {[
                      'Sci-Fi',
                      'Drama',
                      'Action',
                      'Thriller',
                      'Romance',
                      'Crime',
                      'Comedy',
                      'Animation',
                      'Horror',
                      'Adventure',
                      'Fantasy',
                      'Mystery'
                    ].map(g => (
                      <option key={g} value={g}>
                        {g}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              {/* Row 2: Storyline / Plot Textarea */}
              <div className="space-y-1.5">
                <div className="flex items-center justify-between">
                  <label className="text-xs font-bold text-[#F1EEE8] uppercase tracking-wider flex items-center gap-1.5">
                    <span>Storyline Plot / Synopsis</span>
                    <span className="text-[#C9A47F]">*</span>
                  </label>
                  <span className="text-xs font-mono text-[#AAB1B9]">
                    {customPlot.length} chars &bull; {customPlot.split(/\s+/).filter(Boolean).length}{' '}
                    words
                  </span>
                </div>
                <textarea
                  required
                  rows={7}
                  placeholder="Paste or write your detailed movie storyline here. Include character motivations, dramatic conflict, perils, and narrative climax for comprehensive NLP analysis..."
                  value={customPlot}
                  onChange={e => setCustomPlot(e.target.value)}
                  className="w-full bg-[#11161B] border border-[#28333E] focus:border-[#C9A47F] rounded-xl p-4 text-sm text-[#F1EEE8] placeholder-[#AAB1B9]/50 focus:outline-none transition-all leading-relaxed shadow-inner"
                />
              </div>

              {/* Row 3: Optional Metadata */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-2 border-t border-[#28333E]">
                <div className="space-y-1.5">
                  <label className="text-xs text-[#AAB1B9]">Release Year (Optional)</label>
                  <input
                    type="number"
                    min="1900"
                    max="2099"
                    placeholder="2025"
                    value={customYear}
                    onChange={e => setCustomYear(e.target.value)}
                    className="w-full bg-[#11161B] border border-[#28333E] focus:border-[#C9A47F] rounded-lg px-3 py-2 text-xs text-[#F1EEE8] focus:outline-none"
                  />
                </div>

                <div className="space-y-1.5">
                  <label className="text-xs text-[#AAB1B9]">Director (Optional)</label>
                  <input
                    type="text"
                    placeholder="e.g. Christopher Nolan / Independent"
                    value={customDirector}
                    onChange={e => setCustomDirector(e.target.value)}
                    className="w-full bg-[#11161B] border border-[#28333E] focus:border-[#C9A47F] rounded-lg px-3 py-2 text-xs text-[#F1EEE8] focus:outline-none"
                  />
                </div>

                <div className="space-y-1.5">
                  <label className="text-xs text-[#AAB1B9]">Poster Image URL (Optional)</label>
                  <input
                    type="url"
                    placeholder="https://images.unsplash.com/..."
                    value={customPoster}
                    onChange={e => setCustomPoster(e.target.value)}
                    className="w-full bg-[#11161B] border border-[#28333E] focus:border-[#C9A47F] rounded-lg px-3 py-2 text-xs text-[#F1EEE8] focus:outline-none"
                  />
                </div>
              </div>

              {/* Action Buttons */}
              <div className="flex flex-col sm:flex-row items-center justify-between gap-4 pt-4 border-t border-[#28333E]">
                <button
                  type="button"
                  onClick={() => {
                    setCustomTitle('');
                    setCustomPlot('');
                    setCustomDirector('');
                    setCustomCast('');
                    setCustomPoster('');
                    setCustomError('');
                    setCustomNotice('');
                  }}
                  className="w-full sm:w-auto px-4 py-2.5 rounded-xl border border-[#28333E] text-xs text-[#AAB1B9] hover:text-[#F1EEE8] hover:bg-[#1B2229] transition-all"
                >
                  Clear Fields
                </button>

                <button
                  type="submit"
                  className="w-full sm:w-auto bg-[#C9A47F] hover:bg-[#8D7058] text-[#11161B] hover:text-[#F1EEE8] font-bold text-sm py-3 px-8 rounded-xl transition-all flex items-center justify-center gap-2 shadow-lg"
                >
                  <Sparkles className="w-4 h-4" />
                  <span>Analyze Storyline Now</span>
                </button>
              </div>
            </form>
          </div>
        )}

        {/* =============================================================== */}
        {/* PAGE 1: DASHBOARD                                               */}
        {/* =============================================================== */}
        {navPage === 'Dashboard' && (
          <div className="max-w-6xl mx-auto space-y-6">
            {/* Header */}
            <div className="space-y-1">
              <h2 className="text-2xl font-bold text-[#F1EEE8] tracking-tight">
                Movie Story Analyser
              </h2>
              <p className="text-sm text-[#AAB1B9]">
                Interactive Natural Language Processing system for storyline intelligence, narrative
                summarization, explainable sentiment, emotion profiling, and multi-movie comparison.
              </p>
            </div>

            {/* Custom Movie Direct Callout */}
            <div className="bg-gradient-to-r from-[#161D24] to-[#1B2229] border border-[#C9A47F]/60 rounded-2xl p-6 flex flex-col md:flex-row items-start md:items-center justify-between gap-4 shadow-lg">
              <div className="space-y-1">
                <div className="flex items-center gap-2 text-xs font-bold text-[#C9A47F] uppercase tracking-wider">
                  <PenTool className="w-4 h-4" />
                  Analyze Your Own Storyline
                </div>
                <h3 className="text-xl font-bold text-[#F1EEE8]">
                  Have your own movie title and plot?
                </h3>
                <p className="text-xs text-[#AAB1B9] max-w-xl">
                  You don't need a CSV file! Simply enter your movie title and plot to decompose its
                  narrative structure, extract sentiment, and identify overarching themes.
                </p>
              </div>
              <button
                onClick={() => setNavPage('Custom Movie')}
                className="bg-[#C9A47F] hover:bg-[#8D7058] text-[#11161B] hover:text-[#F1EEE8] font-bold text-xs py-2.5 px-5 rounded-xl transition-all shadow-md flex items-center gap-2 flex-shrink-0"
              >
                <span>Enter Custom Movie</span>
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>

            {/* 4 Metric Cards */}
            {stats && (
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                <div className="bg-[#161D24] border border-[#28333E] rounded-xl p-5 hover:border-[#C9A47F]/40 transition-all shadow-md">
                  <div className="text-xs font-semibold text-[#AAB1B9] uppercase tracking-wider">
                    Total Valid Movies
                  </div>
                  <div className="text-3xl font-extrabold text-[#C9A47F] mt-2 font-mono">
                    {stats.valid_movies}
                  </div>
                  <div className="text-xs text-[#AAB1B9] mt-1">Available in active catalog</div>
                </div>

                <div className="bg-[#161D24] border border-[#28333E] rounded-xl p-5 hover:border-[#C9A47F]/40 transition-all shadow-md">
                  <div className="text-xs font-semibold text-[#AAB1B9] uppercase tracking-wider">
                    Unique Genres
                  </div>
                  <div className="text-3xl font-extrabold text-[#C9A47F] mt-2 font-mono">
                    {stats.unique_genres_count}
                  </div>
                  <div className="text-xs text-[#AAB1B9] mt-1">Categorized across dataset</div>
                </div>

                <div className="bg-[#161D24] border border-[#28333E] rounded-xl p-5 hover:border-[#C9A47F]/40 transition-all shadow-md">
                  <div className="text-xs font-semibold text-[#AAB1B9] uppercase tracking-wider">
                    Release Year Span
                  </div>
                  <div className="text-2xl font-extrabold text-[#C9A47F] mt-2 font-mono">
                    {stats.year_range[0]} – {stats.year_range[1]}
                  </div>
                  <div className="text-xs text-[#AAB1B9] mt-1">Earliest to latest release</div>
                </div>

                <div className="bg-[#161D24] border border-[#28333E] rounded-xl p-5 hover:border-[#C9A47F]/40 transition-all shadow-md">
                  <div className="text-xs font-semibold text-[#AAB1B9] uppercase tracking-wider">
                    Avg Plot Length
                  </div>
                  <div className="text-3xl font-extrabold text-[#C9A47F] mt-2 font-mono">
                    {stats.avg_storyline_words}{' '}
                    <span className="text-base font-normal text-[#AAB1B9]">words</span>
                  </div>
                  <div className="text-xs text-[#AAB1B9] mt-1">Narrative density score</div>
                </div>
              </div>
            )}

            {/* Currently Selected Movie Banner */}
            {selectedMovie && (
              <div className="bg-[#161D24] border border-[#28333E] rounded-xl p-6 shadow-lg">
                <div className="text-xs font-semibold text-[#C9A47F] uppercase tracking-wider mb-4 flex items-center gap-2">
                  <Sparkles className="w-3.5 h-3.5" />
                  Currently Selected Movie for NLP Decomposition
                </div>

                <div className="flex flex-col md:flex-row gap-6 items-start">
                  {selectedMovie.poster_url ? (
                    <img
                      src={selectedMovie.poster_url}
                      alt={selectedMovie.title}
                      className="w-36 h-52 object-cover rounded-lg border border-[#28333E] shadow-md flex-shrink-0"
                    />
                  ) : (
                    <div className="w-36 h-52 rounded-lg bg-[#1B2229] border border-[#28333E] flex flex-col items-center justify-center text-xs text-[#AAB1B9] flex-shrink-0">
                      <Film className="w-8 h-8 mb-2 text-[#8D7058]" />
                      No Poster
                    </div>
                  )}

                  <div className="flex-1 space-y-3">
                    <div className="flex flex-wrap items-baseline gap-3">
                      <h3 className="text-2xl font-bold text-[#F1EEE8] tracking-tight">
                        {selectedMovie.title}
                      </h3>
                      {selectedMovie.year && (
                        <span className="text-sm font-mono text-[#AAB1B9]">
                          ({selectedMovie.year})
                        </span>
                      )}
                    </div>

                    <div className="flex flex-wrap gap-1.5">
                      {selectedMovie.genres.map(g => (
                        <span
                          key={g}
                          className="px-2.5 py-0.5 rounded text-xs font-medium bg-[#1B2229] border border-[#28333E] text-[#C9A47F]"
                        >
                          {g}
                        </span>
                      ))}
                    </div>

                    <div className="text-xs text-[#AAB1B9] space-x-4">
                      <span>
                        Director:{' '}
                        <strong className="text-[#F1EEE8]">{selectedMovie.director}</strong>
                      </span>
                      <span>
                        Cast: <strong className="text-[#F1EEE8]">{selectedMovie.cast}</strong>
                      </span>
                      <span>
                        Storyline Length:{' '}
                        <strong className="text-[#F1EEE8]">{selectedMovie.word_count} words</strong>
                      </span>
                    </div>

                    <p className="text-sm text-[#AAB1B9] line-clamp-3 leading-relaxed">
                      {selectedMovie.storyline}
                    </p>

                    <div className="flex gap-3 pt-2">
                      <button
                        onClick={() => setNavPage('Story Analysis')}
                        className="bg-[#C9A47F] hover:bg-[#8D7058] text-[#11161B] hover:text-[#F1EEE8] text-xs font-semibold py-2 px-4 rounded-lg transition-all flex items-center gap-1.5 shadow-sm"
                      >
                        <span>Analyze Storyline</span>
                        <ChevronRight className="w-3.5 h-3.5" />
                      </button>
                      <button
                        onClick={() => setNavPage('Movie Explorer')}
                        className="bg-[#1B2229] hover:bg-[#202933] border border-[#28333E] text-[#F1EEE8] text-xs font-medium py-2 px-4 rounded-lg transition-all"
                      >
                        Explore More Movies
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Dataset Preview Table */}
            <div className="bg-[#161D24] border border-[#28333E] rounded-xl p-5 space-y-3">
              <div className="flex items-center justify-between">
                <h4 className="text-sm font-semibold text-[#F1EEE8]">
                  Dataset Records Table Preview
                </h4>
                <span className="text-xs text-[#AAB1B9]">
                  Showing first {Math.min(10, movies.length)} of {movies.length} records
                </span>
              </div>

              <div className="overflow-x-auto rounded-lg border border-[#28333E]">
                <table className="w-full text-xs text-left">
                  <thead className="bg-[#1B2229] text-[#C9A47F] uppercase tracking-wider font-semibold border-b border-[#28333E]">
                    <tr>
                      <th className="py-2.5 px-3">Title</th>
                      <th className="py-2.5 px-3">Year</th>
                      <th className="py-2.5 px-3">Primary Genre</th>
                      <th className="py-2.5 px-3">Director</th>
                      <th className="py-2.5 px-3">Word Count</th>
                      <th className="py-2.5 px-3">Overview Excerpt</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#28333E] text-[#AAB1B9]">
                    {movies.slice(0, 8).map(m => (
                      <tr
                        key={m.movie_id}
                        onClick={() => setSelectedMovieId(m.movie_id)}
                        className={`hover:bg-[#1B2229]/60 cursor-pointer transition-colors ${
                          m.movie_id === selectedMovieId ? 'bg-[#1B2229] text-[#F1EEE8]' : ''
                        }`}
                      >
                        <td className="py-2.5 px-3 font-semibold text-[#F1EEE8]">{m.title}</td>
                        <td className="py-2.5 px-3 font-mono">{m.year || 'N/A'}</td>
                        <td className="py-2.5 px-3">
                          <span className="px-2 py-0.5 rounded bg-[#11161B] text-[#C9A47F] border border-[#28333E]">
                            {m.primary_genre}
                          </span>
                        </td>
                        <td className="py-2.5 px-3">{m.director}</td>
                        <td className="py-2.5 px-3 font-mono">{m.word_count}</td>
                        <td className="py-2.5 px-3 truncate max-w-xs">{m.storyline}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* =============================================================== */}
        {/* PAGE 2: MOVIE EXPLORER                                          */}
        {/* =============================================================== */}
        {navPage === 'Movie Explorer' && (
          <div className="max-w-6xl mx-auto space-y-6">
            <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
              <div>
                <h2 className="text-2xl font-bold text-[#F1EEE8] tracking-tight">Movie Explorer</h2>
                <p className="text-sm text-[#AAB1B9]">
                  Search, filter, and inspect movies dynamically from the ingested dataset.
                </p>
              </div>
              <div className="flex items-center gap-3">
                <button
                  onClick={() => setNavPage('Custom Movie')}
                  className="bg-[#C9A47F] hover:bg-[#8D7058] text-[#11161B] hover:text-[#F1EEE8] text-xs font-bold py-2 px-3 rounded-lg transition-all flex items-center gap-1.5 shadow-sm"
                >
                  <PlusCircle className="w-3.5 h-3.5" />
                  <span>Add Custom Movie</span>
                </button>
                <div className="text-xs text-[#C9A47F] font-mono">
                  Showing {filteredMovies.length} of {movies.length} movies
                </div>
              </div>
            </div>

            {/* Filter Controls Bar */}
            <div className="bg-[#161D24] border border-[#28333E] rounded-xl p-4 grid grid-cols-1 md:grid-cols-3 gap-3">
              <div className="relative">
                <Search className="w-4 h-4 text-[#AAB1B9] absolute left-3 top-3" />
                <input
                  type="text"
                  placeholder="Search title, director, storyline..."
                  value={searchQuery}
                  onChange={e => setSearchQuery(e.target.value)}
                  className="w-full bg-[#11161B] border border-[#28333E] rounded-lg pl-9 pr-3 py-2 text-xs text-[#F1EEE8] placeholder-[#AAB1B9]/60 focus:outline-none focus:border-[#C9A47F]"
                />
              </div>

              <div>
                <select
                  value={selectedGenre}
                  onChange={e => setSelectedGenre(e.target.value)}
                  className="w-full bg-[#11161B] border border-[#28333E] rounded-lg px-3 py-2 text-xs text-[#F1EEE8] focus:outline-none focus:border-[#C9A47F]"
                >
                  <option value="All">All Genres</option>
                  {stats?.unique_genres.map(g => (
                    <option key={g} value={g}>
                      {g}
                    </option>
                  ))}
                </select>
              </div>

              <div className="flex items-center gap-2">
                <button
                  onClick={() => {
                    setSearchQuery('');
                    setSelectedGenre('All');
                  }}
                  className="w-full bg-[#1B2229] hover:bg-[#202933] border border-[#28333E] text-[#AAB1B9] hover:text-[#F1EEE8] text-xs py-2 px-3 rounded-lg transition-all"
                >
                  Clear Filters
                </button>
              </div>
            </div>

            {/* Movies Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {filteredMovies.map(m => {
                const isSelected = m.movie_id === selectedMovieId;
                return (
                  <div
                    key={m.movie_id}
                    className={`bg-[#161D24] border rounded-xl p-5 flex gap-4 transition-all hover:border-[#C9A47F]/60 ${
                      isSelected
                        ? 'border-[#C9A47F] shadow-lg ring-1 ring-[#C9A47F]/30'
                        : 'border-[#28333E]'
                    }`}
                  >
                    {m.poster_url ? (
                      <img
                        src={m.poster_url}
                        alt={m.title}
                        className="w-24 h-36 object-cover rounded-md border border-[#28333E] flex-shrink-0"
                      />
                    ) : (
                      <div className="w-24 h-36 rounded-md bg-[#1B2229] border border-[#28333E] flex flex-col items-center justify-center text-xs text-[#AAB1B9] flex-shrink-0">
                        <Film className="w-6 h-6 mb-1 text-[#8D7058]" />
                        No Poster
                      </div>
                    )}

                    <div className="flex-1 flex flex-col justify-between">
                      <div className="space-y-1.5">
                        <div className="flex items-baseline justify-between gap-2">
                          <h4 className="text-base font-bold text-[#F1EEE8] line-clamp-1">
                            {m.title}
                          </h4>
                          {m.year && (
                            <span className="text-xs font-mono text-[#AAB1B9]">({m.year})</span>
                          )}
                        </div>

                        <div className="flex flex-wrap gap-1">
                          {m.genres.slice(0, 3).map(g => (
                            <span
                              key={g}
                              className="text-[10px] px-2 py-0.5 rounded bg-[#1B2229] text-[#C9A47F] border border-[#28333E]"
                            >
                              {g}
                            </span>
                          ))}
                        </div>

                        <p className="text-xs text-[#AAB1B9] line-clamp-3 leading-relaxed">
                          {m.storyline}
                        </p>
                      </div>

                      <div className="pt-3 border-t border-[#28333E]/60 flex items-center justify-between mt-2">
                        <span className="text-[11px] text-[#8D7058]">
                          Director: {m.director || 'Unknown'}
                        </span>
                        <button
                          onClick={() => {
                            setSelectedMovieId(m.movie_id);
                            setNavPage('Story Analysis');
                          }}
                          className={`text-xs px-3 py-1.5 rounded font-medium transition-all ${
                            isSelected
                              ? 'bg-[#C9A47F] text-[#11161B] font-semibold'
                              : 'bg-[#1B2229] text-[#F1EEE8] hover:bg-[#C9A47F] hover:text-[#11161B]'
                          }`}
                        >
                          Select & Analyze
                        </button>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* =============================================================== */}
        {/* PAGE 3: STORY ANALYSIS (PREPROCESSING & SUMMARIZATION)           */}
        {/* =============================================================== */}
        {navPage === 'Story Analysis' && selectedMovie && analysis && (
          <div className="max-w-6xl mx-auto space-y-6">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div className="space-y-1">
                <div className="flex items-center gap-2 text-xs text-[#C9A47F] font-semibold uppercase tracking-wider">
                  <BookOpen className="w-3.5 h-3.5" />
                  NLP Storyline Decomposition &bull; {selectedMovie.title}
                </div>
                <h2 className="text-2xl font-bold text-[#F1EEE8]">
                  {selectedMovie.title} {selectedMovie.year && `(${selectedMovie.year})`}
                </h2>
              </div>
              <button
                onClick={() => setNavPage('Custom Movie')}
                className="bg-[#161D24] hover:bg-[#202933] border border-[#C9A47F]/40 text-[#C9A47F] text-xs font-semibold py-2 px-3.5 rounded-lg transition-all flex items-center gap-1.5 self-start"
              >
                <PlusCircle className="w-3.5 h-3.5" />
                <span>Analyze Another Custom Movie</span>
              </button>
            </div>

            {/* Storyline Full Box */}
            <div className="bg-[#161D24] border border-[#28333E] rounded-xl p-5 space-y-3">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-semibold text-[#F1EEE8]">Complete Raw Storyline</h3>
                <span className="text-xs text-[#C9A47F] font-mono">
                  {selectedMovie.word_count} words &bull; {analysis.preprocessing.stats.sentence_count} sentences
                </span>
              </div>
              <p className="text-sm text-[#F1EEE8]/90 leading-relaxed bg-[#11161B] p-4 rounded-lg border border-[#28333E]">
                {selectedMovie.storyline}
              </p>
            </div>

            {/* Extractive Summarization Section */}
            <div className="bg-[#161D24] border border-[#28333E] rounded-xl p-6 space-y-5">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-[#28333E]">
                <div>
                  <h3 className="text-base font-bold text-[#C9A47F]">
                    Extractive Story Summarization (TextRank Graph Centrality)
                  </h3>
                  <p className="text-xs text-[#AAB1B9]">
                    Cosine similarity sentence graph ranked via PageRank power iteration
                  </p>
                </div>
                <div className="flex items-center gap-3">
                  <span className="text-xs text-[#AAB1B9]">Summary Sentences:</span>
                  <input
                    type="range"
                    min="1"
                    max="5"
                    value={summarySentencesCount}
                    onChange={e => setSummarySentencesCount(parseInt(e.target.value, 10))}
                    className="accent-[#C9A47F] cursor-pointer"
                  />
                  <span className="text-xs font-mono font-bold text-[#C9A47F]">
                    {summarySentencesCount}
                  </span>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Short Executive Summary */}
                <div className="bg-[#1B2229] border-l-4 border-[#C9A47F] p-4 rounded-r-lg space-y-2">
                  <div className="text-xs font-semibold uppercase text-[#C9A47F] tracking-wider">
                    Executive Summary (Top 1–2 Central Sentences)
                  </div>
                  <p className="text-sm text-[#F1EEE8] leading-relaxed">
                    {analysis.summary.short_summary}
                  </p>
                </div>

                {/* Detailed Summary */}
                <div className="bg-[#1B2229] border-l-4 border-[#8D7058] p-4 rounded-r-lg space-y-2">
                  <div className="text-xs font-semibold uppercase text-[#8D7058] tracking-wider">
                    Detailed Summary ({analysis.summary.selected_count} Chronological Sentences)
                  </div>
                  <p className="text-sm text-[#F1EEE8] leading-relaxed">
                    {analysis.summary.detailed_summary}
                  </p>
                </div>
              </div>

              {/* Ranked Sentences Breakdown */}
              <details className="text-xs pt-2">
                <summary className="cursor-pointer font-semibold text-[#AAB1B9] hover:text-[#C9A47F]">
                  View TextRank Sentence Centrality & PageRank Scores
                </summary>
                <div className="mt-3 overflow-x-auto border border-[#28333E] rounded-lg">
                  <table className="w-full text-left text-xs">
                    <thead className="bg-[#1B2229] text-[#C9A47F] border-b border-[#28333E]">
                      <tr>
                        <th className="py-2 px-3 w-16">Rank</th>
                        <th className="py-2 px-3 w-24">PageRank</th>
                        <th className="py-2 px-3">Sentence Text</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-[#28333E] text-[#AAB1B9]">
                      {analysis.summary.ranked_sentences.map(r => (
                        <tr key={r.index} className="hover:bg-[#1B2229]/40">
                          <td className="py-2 px-3 font-mono font-bold text-[#F1EEE8]">
                            #{r.rank}
                          </td>
                          <td className="py-2 px-3 font-mono text-[#C9A47F]">
                            {r.score.toFixed(4)}
                          </td>
                          <td className="py-2 px-3 text-[#F1EEE8]">{r.sentence}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </details>
            </div>

            {/* NLP Preprocessing Multi-Stage Tabs */}
            <div className="bg-[#161D24] border border-[#28333E] rounded-xl p-6 space-y-4">
              <h3 className="text-base font-bold text-[#F1EEE8]">
                Multi-Stage NLP Text Preprocessing Pipeline
              </h3>

              <div className="flex border-b border-[#28333E] gap-2 overflow-x-auto">
                {[
                  '1. Sentence Segmentation',
                  '2. Tokenization & Cleaning',
                  '3. Stopword Removal',
                  '4. Lemmatization'
                ].map((tab, idx) => (
                  <button
                    key={tab}
                    onClick={() => setActivePrepTab(idx)}
                    className={`py-2 px-3 text-xs font-medium whitespace-nowrap transition-colors border-b-2 ${
                      activePrepTab === idx
                        ? 'border-[#C9A47F] text-[#C9A47F]'
                        : 'border-transparent text-[#AAB1B9] hover:text-[#F1EEE8]'
                    }`}
                  >
                    {tab}
                  </button>
                ))}
              </div>

              {activePrepTab === 0 && (
                <div className="space-y-2 text-xs">
                  <div className="text-[#AAB1B9]">
                    Segmented into {analysis.preprocessing.sentences.length} grammatical sentences:
                  </div>
                  <div className="space-y-1.5">
                    {analysis.preprocessing.sentences.map((s, i) => (
                      <div
                        key={i}
                        className="p-2.5 rounded bg-[#1B2229] border border-[#28333E] text-[#F1EEE8]"
                      >
                        <span className="font-mono text-[#C9A47F] mr-2">[{i + 1}]</span>
                        {s}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {activePrepTab === 1 && (
                <div className="space-y-2 text-xs">
                  <div className="text-[#AAB1B9]">
                    Extracted {analysis.preprocessing.tokens.length} word tokens (alphanumeric
                    filtering):
                  </div>
                  <div className="p-3 bg-[#1B2229] rounded border border-[#28333E] font-mono text-[#F1EEE8]/80 leading-relaxed max-h-48 overflow-y-auto">
                    {analysis.preprocessing.cleaned_tokens.join('  •  ')}
                  </div>
                </div>
              )}

              {activePrepTab === 2 && (
                <div className="space-y-2 text-xs">
                  <div className="text-[#AAB1B9]">
                    Filtered {analysis.preprocessing.filtered_tokens.length} non-stopword tokens
                    (standard NLTK stopwords stripped):
                  </div>
                  <div className="p-3 bg-[#1B2229] rounded border border-[#28333E] font-mono text-[#F1EEE8]/80 leading-relaxed max-h-48 overflow-y-auto">
                    {analysis.preprocessing.filtered_tokens.join('  •  ')}
                  </div>
                </div>
              )}

              {activePrepTab === 3 && (
                <div className="space-y-2 text-xs">
                  <div className="text-[#AAB1B9]">
                    Lemmatized Vocabulary ({analysis.preprocessing.stats.vocab_size} unique
                    dictionary lemmas):
                  </div>
                  <div className="p-3 bg-[#1B2229] rounded border border-[#28333E] text-[#C9A47F] leading-relaxed">
                    {analysis.preprocessing.processed_text_string}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* =============================================================== */}
        {/* PAGE 4: SENTIMENT AND EMOTIONS                                  */}
        {/* =============================================================== */}
        {navPage === 'Sentiment and Emotions' && selectedMovie && analysis && (
          <div className="max-w-6xl mx-auto space-y-6">
            <div className="space-y-1">
              <div className="flex items-center gap-2 text-xs text-[#C9A47F] font-semibold uppercase tracking-wider">
                <Heart className="w-3.5 h-3.5" />
                Member 3 Module &bull; Explainable Sentiment & Emotion Profiling
              </div>
              <h2 className="text-2xl font-bold text-[#F1EEE8]">
                Sentiment & Emotions: {selectedMovie.title}
              </h2>
            </div>

            {/* Explainable Sentiment Section */}
            <div className="bg-[#161D24] border border-[#28333E] rounded-xl p-6 space-y-5">
              <h3 className="text-base font-bold text-[#C9A47F]">
                1. Explainable Storyline Sentiment (VADER)
              </h3>

              {/* Metrics Row */}
              <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                <div className="bg-[#1B2229] border border-[#28333E] p-4 rounded-lg">
                  <div className="text-[11px] uppercase tracking-wider text-[#AAB1B9]">
                    Overall Polarity
                  </div>
                  <div
                    className={`text-xl font-bold mt-1 ${
                      analysis.sentiment.overall_sentiment === 'Positive'
                        ? 'text-[#4E9F76]'
                        : analysis.sentiment.overall_sentiment === 'Negative'
                        ? 'text-[#D95D5D]'
                        : 'text-[#C9A47F]'
                    }`}
                  >
                    {analysis.sentiment.overall_sentiment}
                  </div>
                </div>

                <div className="bg-[#1B2229] border border-[#28333E] p-4 rounded-lg">
                  <div className="text-[11px] uppercase tracking-wider text-[#AAB1B9]">
                    Compound Score
                  </div>
                  <div className="text-xl font-mono font-bold text-[#F1EEE8] mt-1">
                    {analysis.sentiment.compound_score >= 0 ? '+' : ''}
                    {analysis.sentiment.compound_score.toFixed(2)}
                  </div>
                </div>

                <div className="bg-[#1B2229] border border-[#28333E] p-4 rounded-lg">
                  <div className="text-[11px] uppercase tracking-wider text-[#AAB1B9]">
                    Positive Sents
                  </div>
                  <div className="text-xl font-mono font-bold text-[#4E9F76] mt-1">
                    {analysis.sentiment.counts.Positive}
                  </div>
                </div>

                <div className="bg-[#1B2229] border border-[#28333E] p-4 rounded-lg">
                  <div className="text-[11px] uppercase tracking-wider text-[#AAB1B9]">
                    Negative Sents
                  </div>
                  <div className="text-xl font-mono font-bold text-[#D95D5D] mt-1">
                    {analysis.sentiment.counts.Negative}
                  </div>
                </div>

                <div className="bg-[#1B2229] border border-[#28333E] p-4 rounded-lg">
                  <div className="text-[11px] uppercase tracking-wider text-[#AAB1B9]">
                    Neutral Sents
                  </div>
                  <div className="text-xl font-mono font-bold text-[#8F9DAA] mt-1">
                    {analysis.sentiment.counts.Neutral}
                  </div>
                </div>
              </div>

              {/* Explainability Diagnostic Box */}
              <div className="bg-[#1B2229] border-l-4 border-[#C9A47F] p-4 rounded-r-lg space-y-1.5">
                <div className="text-xs font-semibold text-[#C9A47F] uppercase tracking-wider">
                  💡 Explainability Diagnostic: Why this classification was assigned
                </div>
                <p className="text-xs text-[#F1EEE8]/90 leading-relaxed">
                  {analysis.sentiment.explanation}
                </p>
              </div>

              {/* Sentence-by-sentence attribution */}
              <div className="space-y-2">
                <div className="text-xs font-semibold text-[#AAB1B9] uppercase tracking-wider">
                  Sentence-Level Attribution & Valence Markers
                </div>
                <div className="space-y-2">
                  {analysis.sentiment.sentences.map((s, idx) => {
                    const isPos = s.sentiment === 'Positive';
                    const isNeg = s.sentiment === 'Negative';
                    return (
                      <div
                        key={idx}
                        className={`p-3 rounded-lg border text-xs space-y-1.5 ${
                          isPos
                            ? 'bg-[#4E9F76]/10 border-[#4E9F76]/40'
                            : isNeg
                            ? 'bg-[#D95D5D]/10 border-[#D95D5D]/40'
                            : 'bg-[#1B2229] border-[#28333E]'
                        }`}
                      >
                        <div className="flex items-center justify-between">
                          <span
                            className={`font-semibold px-2 py-0.5 rounded text-[10px] ${
                              isPos
                                ? 'bg-[#4E9F76]/20 text-[#4E9F76]'
                                : isNeg
                                ? 'bg-[#D95D5D]/20 text-[#D95D5D]'
                                : 'bg-[#8F9DAA]/20 text-[#8F9DAA]'
                            }`}
                          >
                            {s.sentiment} ({s.compound >= 0 ? '+' : ''}
                            {s.compound.toFixed(2)})
                          </span>
                        </div>
                        <p className="text-[#F1EEE8] text-sm leading-relaxed">{s.sentence}</p>
                        {(s.pos_words.length > 0 || s.neg_words.length > 0) && (
                          <div className="text-[11px] text-[#AAB1B9] space-x-3 pt-1">
                            {s.pos_words.length > 0 && (
                              <span>
                                Positive cues:{' '}
                                <strong className="text-[#4E9F76]">
                                  {s.pos_words.map(([w]) => w).join(', ')}
                                </strong>
                              </span>
                            )}
                            {s.neg_words.length > 0 && (
                              <span>
                                Negative cues:{' '}
                                <strong className="text-[#D95D5D]">
                                  {s.neg_words.map(([w]) => w).join(', ')}
                                </strong>
                              </span>
                            )}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>

            {/* 7-Dimension Emotion Profiling */}
            <div className="bg-[#161D24] border border-[#28333E] rounded-xl p-6 space-y-5">
              <h3 className="text-base font-bold text-[#C9A47F]">
                2. Narrative Emotion Profiling (7 Dimensions)
              </h3>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
                {/* Dominant Emotion Summary */}
                <div className="space-y-4">
                  <div className="bg-[#1B2229] border border-[#28333E] rounded-lg p-5">
                    <div className="text-xs uppercase text-[#AAB1B9] tracking-wider">
                      Dominant Emotion
                    </div>
                    <div className="text-2xl font-bold text-[#C9A47F] mt-1">
                      {analysis.emotion.dominant_emotion}
                    </div>
                    <div className="text-xs uppercase text-[#AAB1B9] tracking-wider mt-4">
                      Secondary Emotion
                    </div>
                    <div className="text-lg font-semibold text-[#8D7058] mt-1">
                      {analysis.emotion.secondary_emotion}
                    </div>
                    <div className="text-xs text-[#AAB1B9] mt-4 pt-3 border-t border-[#28333E]">
                      Emotion Lexical Markers Found:{' '}
                      <strong className="text-[#F1EEE8]">
                        {analysis.emotion.total_emotion_markers}
                      </strong>
                    </div>
                  </div>

                  {/* Lexical Evidence */}
                  <details className="text-xs">
                    <summary className="cursor-pointer text-[#C9A47F] font-semibold">
                      Inspect Grounded Emotion Terms
                    </summary>
                    <div className="p-3 mt-2 bg-[#1B2229] rounded border border-[#28333E] space-y-2">
                      {Object.entries(analysis.emotion.evidence).map(([emo, list]) => {
                        if (!list.length) return null;
                        return (
                          <div key={emo}>
                            <strong className="text-[#F1EEE8]">{emo}:</strong>{' '}
                            <span className="text-[#AAB1B9]">
                              {list.map(([w, wt]) => `${w} (${wt})`).join(', ')}
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  </details>
                </div>

                {/* Emotion Distribution Bars */}
                <div className="md:col-span-2 bg-[#1B2229] border border-[#28333E] rounded-lg p-5 space-y-3">
                  <div className="text-xs font-semibold text-[#AAB1B9] uppercase tracking-wider mb-2">
                    7-Emotion Percentage Distribution
                  </div>

                  {Object.entries(analysis.emotion.percentages).map(([emo, pct]) => {
                    const colors: Record<string, string> = {
                      Happiness: '#E0A96D',
                      Sadness: '#6C7A89',
                      Fear: '#8E44AD',
                      Anger: '#C0392B',
                      Surprise: '#F39C12',
                      Love: '#D87093',
                      Hope: '#27AE60'
                    };
                    const color = colors[emo] || '#C9A47F';

                    return (
                      <div key={emo} className="space-y-1">
                        <div className="flex justify-between text-xs">
                          <span className="font-medium text-[#F1EEE8]">{emo}</span>
                          <span className="font-mono text-[#AAB1B9]">{pct.toFixed(1)}%</span>
                        </div>
                        <div className="h-3 w-full bg-[#11161B] rounded-full overflow-hidden border border-[#28333E]">
                          <div
                            className="h-full rounded-full transition-all duration-500"
                            style={{
                              width: `${Math.min(100, Math.max(3, pct))}%`,
                              backgroundColor: color
                            }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* =============================================================== */}
        {/* PAGE 5: KEYWORDS AND THEMES                                     */}
        {/* =============================================================== */}
        {navPage === 'Keywords and Themes' && selectedMovie && analysis && (
          <div className="max-w-6xl mx-auto space-y-6">
            <div className="space-y-1">
              <div className="flex items-center gap-2 text-xs text-[#C9A47F] font-semibold uppercase tracking-wider">
                <Tag className="w-3.5 h-3.5" />
                Member 4 Module &bull; TF-IDF Keywords & Thematic Clusters
              </div>
              <h2 className="text-2xl font-bold text-[#F1EEE8]">
                Keywords & Themes: {selectedMovie.title}
              </h2>
            </div>

            {/* TF-IDF Keywords Section */}
            <div className="bg-[#161D24] border border-[#28333E] rounded-xl p-6 space-y-5">
              <h3 className="text-base font-bold text-[#C9A47F]">
                1. TF-IDF Salient Keywords & N-Gram Keyphrases
              </h3>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                {/* Horizontal Bar Chart */}
                <div className="bg-[#1B2229] border border-[#28333E] rounded-lg p-5 space-y-3">
                  <div className="text-xs font-semibold text-[#AAB1B9] uppercase tracking-wider mb-2">
                    Top Salient Keywords (TF-IDF Weight)
                  </div>
                  <div className="space-y-2.5">
                    {analysis.keywords.keywords.slice(0, 8).map(k => (
                      <div key={k.keyword} className="space-y-1">
                        <div className="flex justify-between text-xs">
                          <span className="font-semibold text-[#F1EEE8]">{k.keyword}</span>
                          <span className="text-[11px] font-mono text-[#C9A47F]">
                            {k.score.toFixed(2)} ({k.type})
                          </span>
                        </div>
                        <div className="h-2.5 w-full bg-[#11161B] rounded-full overflow-hidden">
                          <div
                            className={`h-full rounded-full ${
                              k.type === 'Bigram' ? 'bg-[#C9A47F]' : 'bg-[#8D7058]'
                            }`}
                            style={{
                              width: `${Math.min(100, Math.max(10, (k.score / 6) * 100))}%`
                            }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Ranked TF-IDF Table */}
                <div className="bg-[#1B2229] border border-[#28333E] rounded-lg p-4 space-y-2">
                  <div className="text-xs font-semibold text-[#AAB1B9] uppercase tracking-wider mb-2">
                    TF-IDF Ranked Terms
                  </div>
                  <div className="overflow-x-auto">
                    <table className="w-full text-left text-xs">
                      <thead className="text-[#C9A47F] border-b border-[#28333E]">
                        <tr>
                          <th className="py-1.5 px-2">Term</th>
                          <th className="py-1.5 px-2">Type</th>
                          <th className="py-1.5 px-2">Score</th>
                          <th className="py-1.5 px-2">Freq</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-[#28333E] text-[#AAB1B9]">
                        {analysis.keywords.keywords.map(k => (
                          <tr key={k.keyword}>
                            <td className="py-1.5 px-2 font-medium text-[#F1EEE8]">{k.keyword}</td>
                            <td className="py-1.5 px-2 text-[10px]">
                              <span
                                className={`px-1.5 py-0.5 rounded ${
                                  k.type === 'Bigram'
                                    ? 'bg-[#C9A47F]/20 text-[#C9A47F]'
                                    : 'bg-[#8D7058]/20 text-[#8D7058]'
                                }`}
                              >
                                {k.type}
                              </span>
                            </td>
                            <td className="py-1.5 px-2 font-mono text-[#F1EEE8]">
                              {k.score.toFixed(2)}
                            </td>
                            <td className="py-1.5 px-2 font-mono">{k.frequency}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </div>

            {/* 10 Thematic Clusters */}
            <div className="bg-[#161D24] border border-[#28333E] rounded-xl p-6 space-y-5">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-base font-bold text-[#C9A47F]">
                    2. Storyline Narrative Thematic Motifs
                  </h3>
                  <p className="text-xs text-[#AAB1B9]">
                    Rule-grounded semantic clustering across 10 canonical storytelling archetypes
                  </p>
                </div>
                <div className="text-xs font-semibold px-3 py-1 rounded bg-[#1B2229] border border-[#C9A47F]/40 text-[#C9A47F]">
                  Dominant: {analysis.themes.dominant_theme}
                </div>
              </div>

              <div className="space-y-3">
                {analysis.themes.themes.length === 0 ? (
                  <div className="text-xs text-[#AAB1B9] p-4 bg-[#1B2229] rounded">
                    No major thematic motifs surpassed the threshold for this storyline.
                  </div>
                ) : (
                  analysis.themes.themes.map(t => (
                    <div
                      key={t.theme}
                      className="bg-[#1B2229] border border-[#28333E] rounded-lg p-4 space-y-2 hover:border-[#C9A47F]/40 transition-all"
                    >
                      <div className="flex items-center justify-between">
                        <div className="font-bold text-sm text-[#F1EEE8]">{t.theme}</div>
                        <span className="text-xs px-2.5 py-0.5 rounded bg-[#C9A47F]/15 text-[#C9A47F] border border-[#C9A47F]/30 font-mono">
                          {t.percentage}% Relevance (Score: {t.score})
                        </span>
                      </div>
                      <p className="text-xs text-[#AAB1B9]">{t.description}</p>
                      <div className="text-xs text-[#C9A47F]">
                        <strong>Lexical Markers:</strong> {t.keywords_found.join(', ')}
                      </div>
                      {t.supporting_sentences.length > 0 && (
                        <div className="mt-2 pt-2 border-t border-[#28333E]/60 text-xs italic text-[#F1EEE8]/80">
                          "{t.supporting_sentences[0]}"
                        </div>
                      )}
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        )}

        {/* =============================================================== */}
        {/* PAGE 6: COMPARE MOVIES                                          */}
        {/* =============================================================== */}
        {navPage === 'Compare Movies' && movies.length >= 2 && (
          <div className="max-w-6xl mx-auto space-y-6">
            <div className="space-y-1">
              <div className="flex items-center gap-2 text-xs text-[#C9A47F] font-semibold uppercase tracking-wider">
                <Scale className="w-3.5 h-3.5" />
                Member 4 Module &bull; Dual Movie Comparative Analytics
              </div>
              <h2 className="text-2xl font-bold text-[#F1EEE8]">Dual Movie Comparison</h2>
              <p className="text-sm text-[#AAB1B9]">
                Select any two films from the dataset (including your custom movies) to compute TF-IDF
                vector space cosine similarity, keyword divergence, and thematic alignment.
              </p>
            </div>

            {/* Dropdown Selectors */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-[#161D24] border border-[#28333E] rounded-xl p-4 space-y-2">
                <label className="text-xs font-semibold text-[#C9A47F] uppercase tracking-wider">
                  Movie A
                </label>
                <select
                  value={compareAId ?? ''}
                  onChange={e => setCompareAId(parseInt(e.target.value, 10))}
                  className="w-full bg-[#11161B] border border-[#28333E] rounded-lg px-3 py-2 text-sm text-[#F1EEE8] focus:outline-none focus:border-[#C9A47F]"
                >
                  {movies.map(m => (
                    <option key={m.movie_id} value={m.movie_id}>
                      {m.title} {m.year && `(${m.year})`}
                    </option>
                  ))}
                </select>
              </div>

              <div className="bg-[#161D24] border border-[#28333E] rounded-xl p-4 space-y-2">
                <label className="text-xs font-semibold text-[#8D7058] uppercase tracking-wider">
                  Movie B
                </label>
                <select
                  value={compareBId ?? ''}
                  onChange={e => setCompareBId(parseInt(e.target.value, 10))}
                  className="w-full bg-[#11161B] border border-[#28333E] rounded-lg px-3 py-2 text-sm text-[#F1EEE8] focus:outline-none focus:border-[#C9A47F]"
                >
                  {movies.map(m => (
                    <option key={m.movie_id} value={m.movie_id}>
                      {m.title} {m.year && `(${m.year})`}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {/* Dynamic Comparison Dashboard */}
            {comparison && (
              <div className="space-y-6">
                {/* Similarity Meter Banner */}
                <div className="bg-[#161D24] border border-[#28333E] rounded-xl p-6 text-center shadow-lg space-y-2">
                  <div className="text-xs font-semibold uppercase tracking-widest text-[#AAB1B9]">
                    TF-IDF Vector Space Storyline Similarity
                  </div>
                  <div className="text-5xl font-extrabold text-[#C9A47F] font-mono">
                    {comparison.similarity_percentage}%
                  </div>
                  <div className="text-xs text-[#AAB1B9]">
                    Cosine Similarity Metric:{' '}
                    <code className="text-[#F1EEE8] bg-[#11161B] px-2 py-0.5 rounded border border-[#28333E]">
                      {comparison.similarity_score.toFixed(4)}
                    </code>
                  </div>
                </div>

                {/* Side-by-Side Cards */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* Movie A Card */}
                  <div className="bg-[#161D24] border border-[#28333E] rounded-xl p-5 space-y-4">
                    <div className="flex items-baseline justify-between border-b border-[#28333E] pb-3">
                      <h3 className="text-lg font-bold text-[#F1EEE8]">
                        {comparison.movie_a.title}
                      </h3>
                      <span className="text-xs font-mono text-[#AAB1B9]">
                        ({comparison.movie_a.year})
                      </span>
                    </div>

                    <div className="flex flex-wrap gap-1">
                      {comparison.movie_a.genres.map(g => (
                        <span
                          key={g}
                          className="text-[10px] px-2 py-0.5 rounded bg-[#1B2229] text-[#C9A47F] border border-[#28333E]"
                        >
                          {g}
                        </span>
                      ))}
                    </div>

                    <div className="bg-[#1B2229] border border-[#28333E] rounded-lg p-3 text-xs space-y-1.5 text-[#AAB1B9]">
                      <div>
                        Word Count:{' '}
                        <strong className="text-[#F1EEE8]">
                          {comparison.movie_a.word_count} words
                        </strong>
                      </div>
                      <div>
                        Storyline Sentiment:{' '}
                        <strong className="text-[#F1EEE8]">{comparison.movie_a.sentiment}</strong>{' '}
                        ({comparison.movie_a.compound.toFixed(2)})
                      </div>
                      <div>
                        Dominant Emotion:{' '}
                        <strong className="text-[#C9A47F]">
                          {comparison.movie_a.dominant_emotion}
                        </strong>
                      </div>
                      <div>
                        Dominant Theme:{' '}
                        <strong className="text-[#8D7058]">
                          {comparison.movie_a.dominant_theme}
                        </strong>
                      </div>
                    </div>

                    <div className="space-y-1 text-xs">
                      <div className="font-semibold text-[#C9A47F]">Extractive Summary:</div>
                      <p className="text-[#F1EEE8]/90 italic leading-relaxed bg-[#11161B] p-3 rounded border border-[#28333E]">
                        "{comparison.movie_a.summary}"
                      </p>
                    </div>
                  </div>

                  {/* Movie B Card */}
                  <div className="bg-[#161D24] border border-[#28333E] rounded-xl p-5 space-y-4">
                    <div className="flex items-baseline justify-between border-b border-[#28333E] pb-3">
                      <h3 className="text-lg font-bold text-[#F1EEE8]">
                        {comparison.movie_b.title}
                      </h3>
                      <span className="text-xs font-mono text-[#AAB1B9]">
                        ({comparison.movie_b.year})
                      </span>
                    </div>

                    <div className="flex flex-wrap gap-1">
                      {comparison.movie_b.genres.map(g => (
                        <span
                          key={g}
                          className="text-[10px] px-2 py-0.5 rounded bg-[#1B2229] text-[#C9A47F] border border-[#28333E]"
                        >
                          {g}
                        </span>
                      ))}
                    </div>

                    <div className="bg-[#1B2229] border border-[#28333E] rounded-lg p-3 text-xs space-y-1.5 text-[#AAB1B9]">
                      <div>
                        Word Count:{' '}
                        <strong className="text-[#F1EEE8]">
                          {comparison.movie_b.word_count} words
                        </strong>
                      </div>
                      <div>
                        Storyline Sentiment:{' '}
                        <strong className="text-[#F1EEE8]">{comparison.movie_b.sentiment}</strong>{' '}
                        ({comparison.movie_b.compound.toFixed(2)})
                      </div>
                      <div>
                        Dominant Emotion:{' '}
                        <strong className="text-[#C9A47F]">
                          {comparison.movie_b.dominant_emotion}
                        </strong>
                      </div>
                      <div>
                        Dominant Theme:{' '}
                        <strong className="text-[#8D7058]">
                          {comparison.movie_b.dominant_theme}
                        </strong>
                      </div>
                    </div>

                    <div className="space-y-1 text-xs">
                      <div className="font-semibold text-[#C9A47F]">Extractive Summary:</div>
                      <p className="text-[#F1EEE8]/90 italic leading-relaxed bg-[#11161B] p-3 rounded border border-[#28333E]">
                        "{comparison.movie_b.summary}"
                      </p>
                    </div>
                  </div>
                </div>

                {/* Keyword Overlap Analysis */}
                <div className="bg-[#161D24] border border-[#28333E] rounded-xl p-6 space-y-4">
                  <h3 className="text-base font-bold text-[#F1EEE8]">
                    Keyword Convergence & Divergence
                  </h3>

                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
                    <div className="p-4 rounded-lg bg-[#1B2229] border border-[#28333E] space-y-2">
                      <div className="font-semibold text-[#C9A47F] uppercase tracking-wider">
                        Shared Keywords
                      </div>
                      <div className="flex flex-wrap gap-1">
                        {comparison.shared_keywords.length > 0 ? (
                          comparison.shared_keywords.map(k => (
                            <span
                              key={k}
                              className="px-2 py-0.5 rounded bg-[#C9A47F]/20 text-[#C9A47F] border border-[#C9A47F]/30"
                            >
                              {k}
                            </span>
                          ))
                        ) : (
                          <span className="text-[#AAB1B9]">No direct keyword overlap</span>
                        )}
                      </div>
                    </div>

                    <div className="p-4 rounded-lg bg-[#1B2229] border border-[#28333E] space-y-2">
                      <div className="font-semibold text-[#AAB1B9] uppercase tracking-wider">
                        Unique to {comparison.movie_a.title}
                      </div>
                      <div className="flex flex-wrap gap-1">
                        {comparison.unique_keywords_a.slice(0, 8).map(k => (
                          <span
                            key={k}
                            className="px-2 py-0.5 rounded bg-[#11161B] text-[#F1EEE8] border border-[#28333E]"
                          >
                            {k}
                          </span>
                        ))}
                      </div>
                    </div>

                    <div className="p-4 rounded-lg bg-[#1B2229] border border-[#28333E] space-y-2">
                      <div className="font-semibold text-[#AAB1B9] uppercase tracking-wider">
                        Unique to {comparison.movie_b.title}
                      </div>
                      <div className="flex flex-wrap gap-1">
                        {comparison.unique_keywords_b.slice(0, 8).map(k => (
                          <span
                            key={k}
                            className="px-2 py-0.5 rounded bg-[#11161B] text-[#F1EEE8] border border-[#28333E]"
                          >
                            {k}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Thematic Overlap */}
                <div className="bg-[#161D24] border border-[#28333E] rounded-xl p-6 space-y-4">
                  <h3 className="text-base font-bold text-[#F1EEE8]">Thematic Overlap</h3>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
                    <div className="p-4 rounded-lg bg-[#1B2229] border border-[#28333E] space-y-2">
                      <div className="font-semibold text-[#C9A47F] uppercase tracking-wider">
                        Shared Active Themes
                      </div>
                      <div className="flex flex-wrap gap-1">
                        {comparison.shared_themes.length > 0 ? (
                          comparison.shared_themes.map(t => (
                            <span
                              key={t}
                              className="px-2.5 py-1 rounded bg-[#C9A47F]/20 text-[#C9A47F] border border-[#C9A47F]/40 font-medium"
                            >
                              {t}
                            </span>
                          ))
                        ) : (
                          <span className="text-[#AAB1B9]">No shared themes</span>
                        )}
                      </div>
                    </div>

                    <div className="p-4 rounded-lg bg-[#1B2229] border border-[#28333E] space-y-2">
                      <div className="font-semibold text-[#AAB1B9] uppercase tracking-wider">
                        Distinct Thematic Focus
                      </div>
                      <div className="space-y-1 text-[#AAB1B9]">
                        <div>
                          <strong>{comparison.movie_a.title}:</strong>{' '}
                          {comparison.unique_themes_a.join(', ') || 'None'}
                        </div>
                        <div>
                          <strong>{comparison.movie_b.title}:</strong>{' '}
                          {comparison.unique_themes_b.join(', ') || 'None'}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
