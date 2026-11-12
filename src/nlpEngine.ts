/**
 * Movie Story Analyser - Client-Side NLP Analytics Engine
 * Implements TextRank, VADER, TF-IDF n-grams, 7-emotion scoring, 10-theme extraction, and dual comparison.
 */

import {
  Movie,
  DatasetStats,
  PreprocessingResult,
  SummaryResult,
  SentimentResult,
  EmotionResult,
  KeywordsResult,
  ThemesResult,
  ComparisonResult,
  SentenceSentiment
} from './types';

// Curated Stopwords
export const STOPWORDS = new Set([
  'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are',
  'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but',
  'by', 'could', 'did', 'do', 'does', 'doing', 'down', 'during', 'each', 'few', 'for', 'from',
  'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here', 'hers', 'herself', 'him',
  'himself', 'his', 'how', 'i', 'if', 'in', 'into', 'is', 'it', 'its', 'itself', 'just', 'me',
  'more', 'most', 'my', 'myself', 'no', 'nor', 'not', 'now', 'of', 'off', 'on', 'once', 'only',
  'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', 'she', 'should',
  'so', 'some', 'such', 'than', 'that', 'the', 'their', 'theirs', 'them', 'themselves', 'then',
  'there', 'these', 'they', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up',
  'very', 'was', 'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why',
  'with', 'would', 'you', 'your', 'yours', 'yourself', 'yourselves'
]);

// VADER Lexicon
const VADER_LEXICON: Record<string, number> = {
  kill: -3.0, killed: -3.2, murder: -3.5, death: -3.2, dead: -3.0, die: -2.9,
  terror: -3.4, terrorist: -3.5, destroy: -2.8, destruction: -3.0, evil: -3.2,
  war: -2.8, threat: -2.4, threaten: -2.6, danger: -2.4, dangerous: -2.5,
  betray: -3.0, betrayal: -3.2, revenge: -2.4, violent: -2.9, violence: -3.1,
  disaster: -3.1, tragedy: -3.2, tragic: -3.0, ruin: -2.6, ruined: -2.8,
  corrupt: -2.8, corruption: -2.9, conspiracy: -2.2, enemy: -2.3, enemies: -2.4,
  hostile: -2.5, fear: -2.3, scared: -2.4, horror: -3.1, horrific: -3.3,
  abused: -3.2, torture: -3.6, suffer: -2.7, suffering: -2.9, pain: -2.3,
  grief: -2.8, sorrow: -2.6, lose: -1.8, lost: -1.9, defeat: -2.2,
  trapped: -2.4, prison: -2.1, criminal: -2.6, crime: -2.5, dark: -1.4,
  sinister: -2.7, nightmare: -2.8, chaos: -2.5, devastating: -3.2,
  monster: -2.6, demon: -3.0, catastrophe: -3.4, doom: -2.8, fatal: -3.1,

  love: 3.2, loved: 3.1, loving: 3.2, hope: 2.5, hero: 2.7, heroic: 3.0,
  victory: 3.1, win: 2.4, won: 2.5, triumph: 3.2, save: 2.6, saved: 2.8,
  savior: 3.0, protect: 2.2, protected: 2.3, peace: 2.8, peaceful: 2.7,
  joy: 3.0, happy: 2.7, happiness: 3.0, freedom: 2.9, free: 2.3,
  friend: 2.4, friends: 2.5, friendship: 2.8, inspire: 2.7, inspiration: 2.9,
  beautiful: 2.8, wonder: 2.4, wonderful: 3.0, miracle: 3.3, redemption: 2.9,
  courage: 2.8, brave: 2.6, success: 2.7, successful: 2.8, master: 2.1,
  brilliant: 2.9, glory: 2.8, champion: 2.9, reconcile: 2.5, harmony: 2.6,
  kind: 2.3, kindness: 2.7, cure: 2.5, rescue: 2.7, rescued: 2.8
};

const NEGATIONS = new Set(['not', 'never', 'no', 'neither', 'hardly', 'barely', 'scarcely', 'without', "doesn't", "don't", "didn't", "won't"]);
const BOOSTERS: Record<string, number> = {
  very: 0.3, extremely: 0.35, deeply: 0.25, incredibly: 0.35, totally: 0.25, absolutely: 0.3
};

// Emotion Lexicon (7 Dimensions)
const EMOTION_LEXICON: Record<string, Record<string, number>> = {
  joy: { Happiness: 1.0 }, celebrate: { Happiness: 0.8 }, laugh: { Happiness: 0.9 },
  cheerful: { Happiness: 0.9 }, delight: { Happiness: 0.9 }, smile: { Happiness: 0.7 },
  fun: { Happiness: 0.8 }, wonderful: { Happiness: 0.8 }, hilarious: { Happiness: 1.0 },
  comedy: { Happiness: 0.8 }, triumph: { Happiness: 0.8, Hope: 0.7 },

  grief: { Sadness: 1.0 }, sorrow: { Sadness: 0.9 }, mourn: { Sadness: 0.9 },
  loss: { Sadness: 0.8 }, tear: { Sadness: 0.8 }, weep: { Sadness: 0.9 },
  tragedy: { Sadness: 0.9, Fear: 0.5 }, heartbroken: { Sadness: 1.0 },
  death: { Sadness: 0.8, Fear: 0.6 }, died: { Sadness: 0.8 }, crying: { Sadness: 0.9 },

  terror: { Fear: 1.0 }, horror: { Fear: 1.0 }, panic: { Fear: 0.9 },
  dread: { Fear: 0.9 }, nightmare: { Fear: 0.8 }, danger: { Fear: 0.7 },
  threat: { Fear: 0.7, Anger: 0.4 }, frightened: { Fear: 0.9 }, scared: { Fear: 0.9 },
  monster: { Fear: 0.7 }, trapped: { Fear: 0.8 }, sinister: { Fear: 0.7 },

  fury: { Anger: 1.0 }, rage: { Anger: 1.0 }, wrath: { Anger: 1.0 },
  revenge: { Anger: 0.9 }, hate: { Anger: 0.9 }, hatred: { Anger: 1.0 },
  hostile: { Anger: 0.8 }, destroy: { Anger: 0.7 }, violent: { Anger: 0.8 },
  attack: { Anger: 0.7 }, fight: { Anger: 0.6 }, betrayal: { Anger: 0.8, Sadness: 0.6 },

  shock: { Surprise: 0.9 }, astonish: { Surprise: 0.9 }, amaze: { Surprise: 0.8 },
  unexpected: { Surprise: 0.8 }, twist: { Surprise: 0.8 }, revelation: { Surprise: 0.8 },
  mystery: { Surprise: 0.6 }, sudden: { Surprise: 0.7 }, secret: { Surprise: 0.7 },

  romance: { Love: 1.0 }, romantic: { Love: 1.0 }, passion: { Love: 0.8 },
  adore: { Love: 0.9 }, affection: { Love: 0.8 }, beloved: { Love: 0.9 },
  kiss: { Love: 0.8 }, marry: { Love: 0.7 }, marriage: { Love: 0.7 },
  love: { Love: 1.0 }, sweetheart: { Love: 0.9 }, soulmate: { Love: 1.0 },

  hope: { Hope: 1.0 }, dream: { Hope: 0.7 }, faith: { Hope: 0.8 },
  rescue: { Hope: 0.7 }, freedom: { Hope: 0.8 }, redemption: { Hope: 0.9 },
  miracle: { Hope: 0.9, Surprise: 0.6 }, survive: { Hope: 0.8 }, inspire: { Hope: 0.8 }
};

// 10 Storyline Themes
const THEMES_CONFIG: Record<string, { keywords: Set<string>; description: string }> = {
  Friendship: {
    keywords: new Set(['friend', 'friends', 'friendship', 'companion', 'buddy', 'ally', 'allies', 'bond', 'partnership', 'loyalty', 'brotherhood', 'team', 'together']),
    description: 'Comradeship, mutual loyalty, and emotional bonds between non-romantic partners.'
  },
  Love: {
    keywords: new Set(['love', 'romance', 'romantic', 'lover', 'passion', 'beloved', 'marry', 'marriage', 'crush', 'kiss', 'affection', 'heart', 'couple', 'soulmate']),
    description: 'Romantic attraction, interpersonal devotion, and courtship.'
  },
  Revenge: {
    keywords: new Set(['revenge', 'avenge', 'vengeance', 'retaliation', 'vendetta', 'payback', 'grudge', 'retribution', 'settle', 'score', 'wrath', 'destroy']),
    description: 'Retribution for past wrongs, grievances, or personal vendettas.'
  },
  Survival: {
    keywords: new Set(['survive', 'survival', 'survivor', 'endure', 'alive', 'peril', 'escape', 'stranded', 'wilderness', 'apocalypse', 'starvation', 'trapped', 'danger']),
    description: 'Physical endurance against mortal peril, isolation, hostile forces, or catastrophe.'
  },
  Family: {
    keywords: new Set(['family', 'father', 'mother', 'son', 'daughter', 'brother', 'sister', 'parent', 'child', 'children', 'ancestor', 'heritage', 'lineage', 'home']),
    description: 'Kinship, generational legacy, familial duties, and parental sacrifices.'
  },
  Mystery: {
    keywords: new Set(['mystery', 'secret', 'investigate', 'detective', 'clue', 'hidden', 'puzzle', 'unravel', 'truth', 'conspiracy', 'unknown', 'disappearance']),
    description: 'Investigating the unknown, unraveling hidden secrets, and solving enigmas.'
  },
  Adventure: {
    keywords: new Set(['adventure', 'journey', 'quest', 'travel', 'expedition', 'explore', 'voyage', 'uncharted', 'realm', 'odyssey', 'discover', 'mission', 'epic']),
    description: 'Epic physical journeys, daring exploits, and voyages across uncharted territory.'
  },
  Betrayal: {
    keywords: new Set(['betray', 'betrayal', 'treason', 'traitor', 'deceive', 'deception', 'double-cross', 'backstab', 'conspiracy', 'fraud', 'unfaithful']),
    description: 'Broken trust, treachery, and sabotage from within allied circles.'
  },
  Sacrifice: {
    keywords: new Set(['sacrifice', 'martyr', 'surrender', 'protect', 'shield', 'cost', 'forfeit', 'selfless', 'heroic', 'price', 'loss']),
    description: 'Willingly forfeiting personal safety, desires, or life for a greater cause.'
  },
  'Social issues': {
    keywords: new Set(['poverty', 'inequality', 'justice', 'society', 'class', 'corruption', 'oppression', 'racism', 'system', 'rebellion', 'protest', 'rights', 'politics', 'greed']),
    description: 'Societal friction, systemic injustice, economic disparity, and political struggle.'
  }
};

// -------------------------------------------------------------
// TEXT PREPROCESSING
// -------------------------------------------------------------
export function segmentSentences(text: string): string[] {
  if (!text) return [];
  const parts = text.split(/(?<=[.!?])\s+(?=[A-Z0-9"'‘“])/g);
  return parts.map(s => s.trim()).filter(s => s.length > 0);
}

export function tokenizeWords(text: string): string[] {
  if (!text) return [];
  const matches = text.match(/\b[a-zA-Z0-9']+\b/g);
  return matches ? Array.from(matches) : [];
}

export function lemmatizeWord(word: string): string {
  const w = word.toLowerCase();
  if (w.endsWith('ies') && w.length > 4) return w.slice(0, -3) + 'y';
  if (w.endsWith('sses') || w.endsWith('shes') || w.endsWith('ches')) return w.slice(0, -2);
  if (w.endsWith('s') && !w.endsWith('ss') && w.length > 3) return w.slice(0, -1);
  if (w.endsWith('ed') && w.length > 4) {
    if (w.endsWith('ied')) return w.slice(0, -3) + 'y';
    return w.endsWith('eed') ? w.slice(0, -1) : w.slice(0, -2);
  }
  if (w.endsWith('ing') && w.length > 5) return w.slice(0, -3);
  return w;
}

export function preprocessText(text: string): PreprocessingResult {
  const sentences = segmentSentences(text);
  const tokens = tokenizeWords(text);
  const cleaned = tokens.map(t => t.toLowerCase()).filter(t => !/^[.,\/#!$%\^&\*;:{}=\-_`~()]+$/.test(t));
  const filtered = cleaned.filter(t => !STOPWORDS.has(t) && !/^\d+$/.test(t));
  const lemmas = filtered.map(lemmatizeWord);

  return {
    raw_text: text,
    sentences,
    tokens,
    cleaned_tokens: cleaned,
    filtered_tokens: filtered,
    lemmas,
    processed_text_string: lemmas.join(' '),
    stats: {
      raw_chars: text.length,
      sentence_count: sentences.length,
      token_count: tokens.length,
      filtered_token_count: filtered.length,
      vocab_size: new Set(lemmas).size
    }
  };
}

// -------------------------------------------------------------
// TEXTRANK EXTRACTIVE SUMMARIZATION
// -------------------------------------------------------------
function cosineSimilarityVectors(v1: number[], v2: number[]): number {
  let dot = 0;
  let n1 = 0;
  let n2 = 0;
  for (let i = 0; i < v1.length; i++) {
    dot += v1[i] * v2[i];
    n1 += v1[i] * v1[i];
    n2 += v2[i] * v2[i];
  }
  if (n1 === 0 || n2 === 0) return 0;
  return dot / (Math.sqrt(n1) * Math.sqrt(n2));
}

export function textrankSummarize(text: string, topN: number = 3): SummaryResult {
  const sentences = segmentSentences(text);
  if (!sentences.length) {
    return {
      short_summary: '',
      detailed_summary: '',
      sentence_count: 0,
      selected_count: 0,
      ranked_sentences: [],
      method: 'TextRank Graph Centrality'
    };
  }

  if (sentences.length <= 2) {
    return {
      short_summary: sentences[0],
      detailed_summary: sentences.join(' '),
      sentence_count: sentences.length,
      selected_count: sentences.length,
      ranked_sentences: sentences.map((s, i) => ({ index: i, rank: i + 1, sentence: s, score: 1.0 })),
      method: 'Direct Extraction (Brief Narrative)'
    };
  }

  // Build sentence vectors over joint vocabulary
  const sentenceLemmas = sentences.map(s => preprocessText(s).lemmas);
  const vocabSet = new Set<string>();
  sentenceLemmas.forEach(lList => lList.forEach(l => vocabSet.add(l)));
  const vocab = Array.from(vocabSet).sort();
  const vMap = new Map(vocab.map((w, idx) => [w, idx]));

  const vectors: number[][] = sentenceLemmas.map(lList => {
    const vec = new Array(vocab.length).fill(0);
    lList.forEach(l => {
      const idx = vMap.get(l);
      if (idx !== undefined) vec[idx] += 1;
    });
    return vec;
  });

  const n = sentences.length;
  const simMatrix: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i !== j) {
        const sim = cosineSimilarityVectors(vectors[i], vectors[j]);
        simMatrix[i][j] = sim > 0.05 ? sim : 0;
      }
    }
  }

  // PageRank power iteration
  const damping = 0.85;
  const maxIter = 50;
  let scores = new Array(n).fill(1 / n);

  for (let iter = 0; iter < maxIter; iter++) {
    const nextScores = new Array(n).fill((1 - damping) / n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (i !== j && simMatrix[j][i] > 0) {
          const sumW = simMatrix[j].reduce((a, b) => a + b, 0);
          if (sumW > 0) {
            nextScores[i] += damping * (scores[j] * simMatrix[j][i] / sumW);
          }
        }
      }
    }
    scores = nextScores;
  }

  const rankedIndices = Array.from({ length: n }, (_, i) => i).sort((a, b) => scores[b] - scores[a]);
  const rankedSentences = rankedIndices.map((idx, r) => ({
    index: idx,
    rank: r + 1,
    sentence: sentences[idx],
    score: Math.round(scores[idx] * 1000) / 1000
  }));

  const selectedCount = Math.min(Math.max(1, topN), n);
  const chosenDetailed = rankedIndices.slice(0, selectedCount).sort((a, b) => a - b);
  const detailedSummary = chosenDetailed.map(i => sentences[i]).join(' ');

  const chosenShort = rankedIndices.slice(0, Math.min(2, n)).sort((a, b) => a - b);
  const shortSummary = chosenShort.map(i => sentences[i]).join(' ');

  return {
    short_summary: shortSummary,
    detailed_summary: detailedSummary,
    sentence_count: n,
    selected_count: selectedCount,
    ranked_sentences: rankedSentences,
    method: 'TextRank Graph Centrality with PageRank'
  };
}

// -------------------------------------------------------------
// VADER EXPLAINABLE SENTIMENT
// -------------------------------------------------------------
export function analyzeSentenceSentiment(sentence: string): SentenceSentiment {
  const words = tokenizeWords(sentence);
  if (!words.length) {
    return {
      sentence,
      compound: 0,
      sentiment: 'Neutral',
      pos_words: [],
      neg_words: []
    };
  }

  const valences: number[] = [];
  const posWords: [string, number][] = [];
  const negWords: [string, number][] = [];

  for (let i = 0; i < words.length; i++) {
    const w = words[i].toLowerCase();
    if (VADER_LEXICON[w] !== undefined) {
      let val = VADER_LEXICON[w];

      // Check negation in 3 preceding words
      let negated = false;
      for (let j = Math.max(0, i - 3); j < i; j++) {
        if (NEGATIONS.has(words[j].toLowerCase())) {
          negated = true;
          break;
        }
      }
      if (negated) val *= -0.74;

      if (i > 0 && BOOSTERS[words[i - 1].toLowerCase()]) {
        const boost = BOOSTERS[words[i - 1].toLowerCase()];
        val += val > 0 ? boost : -boost;
      }

      valences.push(val);
      if (val > 0.5) posWords.push([words[i], Math.round(val * 10) / 10]);
      if (val < -0.5) negWords.push([words[i], Math.round(val * 10) / 10]);
    }
  }

  if (!valences.length) {
    return {
      sentence,
      compound: 0,
      sentiment: 'Neutral',
      pos_words: [],
      neg_words: []
    };
  }

  const sumVal = valences.reduce((a, b) => a + b, 0);
  const alpha = 15;
  const compound = Math.round((sumVal / Math.sqrt(sumVal * sumVal + alpha)) * 1000) / 1000;

  let sentiment: 'Positive' | 'Negative' | 'Neutral' = 'Neutral';
  if (compound >= 0.15) sentiment = 'Positive';
  else if (compound <= -0.15) sentiment = 'Negative';

  return {
    sentence,
    compound,
    sentiment,
    pos_words: posWords,
    neg_words: negWords
  };
}

export function analyzeStorylineSentiment(text: string): SentimentResult {
  const sentences = segmentSentences(text);
  if (!sentences.length) {
    return {
      overall_sentiment: 'Neutral',
      compound_score: 0,
      counts: { Positive: 0, Negative: 0, Neutral: 0, Total: 0 },
      proportions: { pos: 0, neg: 0, neu: 1 },
      sentences: [],
      key_neg_sentences: [],
      key_pos_sentences: [],
      contributing_neg_words: [],
      contributing_pos_words: [],
      explanation: 'No text provided for analysis.',
      method: 'VADER Rule-Based Model'
    };
  }

  const sentenceResults = sentences.map(analyzeSentenceSentiment);
  let posCount = 0;
  let negCount = 0;
  let neuCount = 0;
  const allNegWords: [string, number][] = [];
  const allPosWords: [string, number][] = [];

  sentenceResults.forEach(r => {
    if (r.sentiment === 'Positive') posCount++;
    else if (r.sentiment === 'Negative') negCount++;
    else neuCount++;
    allNegWords.push(...r.neg_words);
    allPosWords.push(...r.pos_words);
  });

  const total = sentences.length;
  const avgCompound = Math.round((sentenceResults.reduce((sum, r) => sum + r.compound, 0) / total) * 1000) / 1000;

  let overall: 'Positive' | 'Negative' | 'Neutral' | 'Mixed' = 'Neutral';
  if (posCount > 0 && negCount > 0 && Math.abs(posCount - negCount) <= 1 && total >= 3) {
    overall = 'Mixed';
  } else if (avgCompound >= 0.12) {
    overall = 'Positive';
  } else if (avgCompound <= -0.12) {
    overall = 'Negative';
  }

  const uniqueNeg = Array.from(new Map(allNegWords).entries());
  const uniquePos = Array.from(new Map(allPosWords).entries());

  const sortedByCompound = [...sentenceResults].sort((a, b) => a.compound - b.compound);
  const keyNeg = sortedByCompound.filter(s => s.compound < -0.1).slice(0, 3);
  const keyPos = [...sortedByCompound].reverse().filter(s => s.compound > 0.1).slice(0, 3);

  let explanation = `The overall storyline sentiment was evaluated as ${overall} with a mean compound score of ${avgCompound >= 0 ? '+' : ''}${avgCompound.toFixed(2)} across ${total} sentences. `;
  if (overall === 'Negative') {
    const terms = uniqueNeg.slice(0, 5).map(([w]) => `'${w}'`).join(', ');
    explanation += `The narrative is dominated by adversity, conflict, or threat (${negCount} negative vs ${posCount} positive sentences). Contributing markers include ${terms || 'conflict vocabulary'}. Critical Note: Storyline sentiment measures dramatic plot peril and tragedy, NOT audience reception or cinematic quality.`;
  } else if (overall === 'Positive') {
    const terms = uniquePos.slice(0, 5).map(([w]) => `'${w}'`).join(', ');
    explanation += `The plot emphasizes triumph, discovery, romance, or rescue (${posCount} positive vs ${negCount} negative sentences). Contributing markers include ${terms || 'uplifting vocabulary'}.`;
  } else if (overall === 'Mixed') {
    explanation += `The plot features balanced dramatic polarity (${posCount} positive and ${negCount} negative sentences), depicting intense conflict followed by redemption or survival.`;
  } else {
    explanation += `The synopsis maintains an objective narrative tone (${neuCount} neutral sentences), describing events without heavy emotive polarity.`;
  }

  return {
    overall_sentiment: overall,
    compound_score: avgCompound,
    counts: {
      Positive: posCount,
      Negative: negCount,
      Neutral: neuCount,
      Total: total
    },
    proportions: {
      pos: Math.round((posCount / total) * 100) / 100,
      neg: Math.round((negCount / total) * 100) / 100,
      neu: Math.round((neuCount / total) * 100) / 100
    },
    sentences: sentenceResults,
    key_neg_sentences: keyNeg,
    key_pos_sentences: keyPos,
    contributing_neg_words: uniqueNeg.slice(0, 8),
    contributing_pos_words: uniquePos.slice(0, 8),
    explanation,
    method: 'VADER Lexicon-Rule Model with Sentence Attribution'
  };
}

// -------------------------------------------------------------
// 7 EMOTION DETECTION
// -------------------------------------------------------------
export function detectEmotions(text: string): EmotionResult {
  const words = tokenizeWords(text);
  const emotions = ['Happiness', 'Sadness', 'Fear', 'Anger', 'Surprise', 'Love', 'Hope'];

  const rawScores: Record<string, number> = {};
  const evidence: Record<string, [string, number][]> = {};
  emotions.forEach(e => {
    rawScores[e] = 0;
    evidence[e] = [];
  });

  words.forEach(w => {
    const wLower = w.toLowerCase();
    if (EMOTION_LEXICON[wLower]) {
      Object.entries(EMOTION_LEXICON[wLower]).forEach(([emo, weight]) => {
        rawScores[emo] += weight;
        evidence[emo].push([wLower, weight]);
      });
    }
  });

  const totalScore = Object.values(rawScores).reduce((a, b) => a + b, 0);
  const distribution: Record<string, number> = {};
  const percentages: Record<string, number> = {};

  if (totalScore === 0) {
    emotions.forEach(e => {
      distribution[e] = 0;
      percentages[e] = Math.round((100 / emotions.length) * 10) / 10;
    });
    return {
      raw_scores: rawScores,
      distribution,
      percentages,
      dominant_emotion: 'Objective / Neutral',
      secondary_emotion: 'None',
      evidence,
      total_emotion_markers: 0,
      method: 'Transparent Lexical Grounding'
    };
  }

  emotions.forEach(e => {
    distribution[e] = Math.round((rawScores[e] / totalScore) * 1000) / 1000;
    percentages[e] = Math.round((rawScores[e] / totalScore) * 1000) / 10;
  });

  const sorted = emotions.map(e => ({ emo: e, score: rawScores[e] })).sort((a, b) => b.score - a.score);
  const dominant = sorted[0].score > 0 ? sorted[0].emo : 'Neutral';
  const secondary = sorted.length > 1 && sorted[1].score > 0 ? sorted[1].emo : 'None';

  return {
    raw_scores: rawScores,
    distribution,
    percentages,
    dominant_emotion: dominant,
    secondary_emotion: secondary,
    evidence,
    total_emotion_markers: Object.values(evidence).reduce((sum, list) => sum + list.length, 0),
    method: 'Transparent Emotion Lexicon with Lexical Grounding'
  };
}

// -------------------------------------------------------------
// TF-IDF KEYWORDS & PHRASES
// -------------------------------------------------------------
export function extractKeywordsTfidf(text: string, topN: number = 10): KeywordsResult {
  const prep = preprocessText(text);
  const lemmas = prep.lemmas;
  const sentences = prep.sentences;

  if (!lemmas.length) {
    return {
      keywords: [],
      unigrams: [],
      bigrams: [],
      total_extracted: 0,
      method: 'TF-IDF Extraction'
    };
  }

  // Unigrams
  const unigrams = lemmas.filter(w => w.length > 2 && !STOPWORDS.has(w));

  // Bigrams
  const bigrams: string[] = [];
  sentences.forEach(s => {
    const sLemmas = preprocessText(s).lemmas.filter(w => w.length > 2);
    for (let i = 0; i < sLemmas.length - 1; i++) {
      bigrams.push(`${sLemmas[i]} ${sLemmas[i + 1]}`);
    }
  });

  const allCandidates = [...unigrams, ...bigrams];
  const counts = new Map<string, number>();
  allCandidates.forEach(c => counts.set(c, (counts.get(c) || 0) + 1));

  // Sentence-level document frequency
  const docTokens = sentences.map(s => new Set(preprocessText(s).lemmas));
  const totalDocs = Math.max(1, sentences.length);

  const tfidfScores: Array<{ keyword: string; type: 'Unigram' | 'Bigram'; score: number; frequency: number }> = [];

  counts.forEach((freq, term) => {
    const parts = term.split(' ');
    const isBigram = parts.length > 1;

    let df = 0;
    if (!isBigram) {
      df = docTokens.filter(set => set.has(parts[0])).length;
    } else {
      df = docTokens.filter(set => set.has(parts[0]) && set.has(parts[1])).length;
    }

    const tf = 1.0 + Math.log(freq);
    const idf = Math.log((totalDocs + 1.0) / (df + 1.0)) + 1.0;
    const boost = isBigram ? 1.25 : 1.0;
    const score = Math.round(tf * idf * boost * 100) / 100;

    tfidfScores.push({
      keyword: term,
      type: isBigram ? 'Bigram' : 'Unigram',
      score,
      frequency: freq
    });
  });

  tfidfScores.sort((a, b) => b.score - a.score);

  const topKeywords = tfidfScores.slice(0, topN);
  const topUnigrams = tfidfScores.filter(k => k.type === 'Unigram').slice(0, topN);
  const topBigrams = tfidfScores.filter(k => k.type === 'Bigram').slice(0, topN);

  return {
    keywords: topKeywords,
    unigrams: topUnigrams,
    bigrams: topBigrams,
    total_extracted: tfidfScores.length,
    method: 'Sublinear TF-IDF with Sentence Document Units'
  };
}

// -------------------------------------------------------------
// THEME DETECTION
// -------------------------------------------------------------
export function detectThemes(text: string): ThemesResult {
  const sentences = segmentSentences(text);
  const allScores: Record<string, number> = {};
  const evidenceWords: Record<string, Set<string>> = {};
  const supportingSents: Record<string, Set<string>> = {};

  Object.keys(THEMES_CONFIG).forEach(t => {
    allScores[t] = 0;
    evidenceWords[t] = new Set();
    supportingSents[t] = new Set();
  });

  sentences.forEach(s => {
    const words = tokenizeWords(s).map(w => w.toLowerCase());
    const matchedThemes = new Set<string>();

    Object.entries(THEMES_CONFIG).forEach(([theme, cfg]) => {
      words.forEach(w => {
        if (cfg.keywords.has(w)) {
          allScores[theme] += 1.5;
          evidenceWords[theme].add(w);
          matchedThemes.add(theme);
        }
      });
    });

    matchedThemes.forEach(t => {
      supportingSents[t].add(s);
    });
  });

  const total = Object.values(allScores).reduce((a, b) => a + b, 0);
  const detectedList: Array<{
    theme: string;
    score: number;
    percentage: number;
    description: string;
    keywords_found: string[];
    supporting_sentences: string[];
  }> = [];

  Object.entries(THEMES_CONFIG).forEach(([theme, cfg]) => {
    const score = allScores[theme];
    if (score > 0) {
      detectedList.push({
        theme,
        score: Math.round(score * 10) / 10,
        percentage: total > 0 ? Math.round((score / total) * 1000) / 10 : 0,
        description: cfg.description,
        keywords_found: Array.from(evidenceWords[theme]).sort(),
        supporting_sentences: Array.from(supportingSents[theme]).slice(0, 2)
      });
    }
  });

  detectedList.sort((a, b) => b.score - a.score);
  const dominant = detectedList.length > 0 ? detectedList[0].theme : 'None';

  return {
    themes: detectedList,
    all_scores: allScores,
    dominant_theme: dominant,
    total_theme_markers: Object.values(evidenceWords).reduce((sum, set) => sum + set.size, 0),
    method: 'Rule-Grounded Semantic Clustering with Sentence Proof'
  };
}

// -------------------------------------------------------------
// DUAL MOVIE COMPARISON
// -------------------------------------------------------------
export function compareTwoMovies(movieA: Movie, movieB: Movie): ComparisonResult {
  const storyA = movieA.storyline || '';
  const storyB = movieB.storyline || '';

  const prepA = preprocessText(storyA);
  const prepB = preprocessText(storyB);

  // TF-IDF Cosine Similarity
  const jointVocab = Array.from(new Set([...prepA.lemmas, ...prepB.lemmas])).sort();
  const vMap = new Map(jointVocab.map((w, idx) => [w, idx]));

  const vecA = new Array(jointVocab.length).fill(0);
  const vecB = new Array(jointVocab.length).fill(0);

  prepA.lemmas.forEach(w => {
    const idx = vMap.get(w);
    if (idx !== undefined) vecA[idx] += 1;
  });

  prepB.lemmas.forEach(w => {
    const idx = vMap.get(w);
    if (idx !== undefined) vecB[idx] += 1;
  });

  const sim = cosineSimilarityVectors(vecA, vecB);
  const simPercent = Math.round(sim * 1000) / 10;

  const sumA = textrankSummarize(storyA, 2);
  const sumB = textrankSummarize(storyB, 2);

  const sentA = analyzeStorylineSentiment(storyA);
  const sentB = analyzeStorylineSentiment(storyB);

  const emoA = detectEmotions(storyA);
  const emoB = detectEmotions(storyB);

  const kwA = extractKeywordsTfidf(storyA, 12);
  const kwB = extractKeywordsTfidf(storyB, 12);

  const thmA = detectThemes(storyA);
  const thmB = detectThemes(storyB);

  const termsA = kwA.keywords.map(k => k.keyword);
  const termsB = kwB.keywords.map(k => k.keyword);

  const setA = new Set(termsA);
  const setB = new Set(termsB);

  const sharedKw = Array.from(setA).filter(k => setB.has(k)).sort();
  const uniqueKwA = Array.from(setA).filter(k => !setB.has(k)).sort();
  const uniqueKwB = Array.from(setB).filter(k => !setA.has(k)).sort();

  const themesA = new Set(thmA.themes.map(t => t.theme));
  const themesB = new Set(thmB.themes.map(t => t.theme));

  const sharedThm = Array.from(themesA).filter(t => themesB.has(t)).sort();
  const uniqueThmA = Array.from(themesA).filter(t => !themesB.has(t)).sort();
  const uniqueThmB = Array.from(themesB).filter(t => !themesA.has(t)).sort();

  return {
    similarity_score: Math.round(sim * 10000) / 10000,
    similarity_percentage: simPercent,
    movie_a: {
      title: movieA.title,
      year: movieA.year || 'N/A',
      genres: movieA.genres,
      word_count: movieA.word_count,
      summary: sumA.short_summary,
      sentiment: sentA.overall_sentiment,
      compound: sentA.compound_score,
      dominant_emotion: emoA.dominant_emotion,
      dominant_theme: thmA.dominant_theme,
      top_keywords: termsA.slice(0, 8),
      emotions: emoA.percentages,
      theme_names: thmA.themes.slice(0, 4).map(t => t.theme)
    },
    movie_b: {
      title: movieB.title,
      year: movieB.year || 'N/A',
      genres: movieB.genres,
      word_count: movieB.word_count,
      summary: sumB.short_summary,
      sentiment: sentB.overall_sentiment,
      compound: sentB.compound_score,
      dominant_emotion: emoB.dominant_emotion,
      dominant_theme: thmB.dominant_theme,
      top_keywords: termsB.slice(0, 8),
      emotions: emoB.percentages,
      theme_names: thmB.themes.slice(0, 4).map(t => t.theme)
    },
    shared_keywords: sharedKw,
    unique_keywords_a: uniqueKwA,
    unique_keywords_b: uniqueKwB,
    shared_themes: sharedThm,
    unique_themes_a: uniqueThmA,
    unique_themes_b: uniqueThmB
  };
}

// -------------------------------------------------------------
// CSV INGESTION & DATASET VALIDATION
// -------------------------------------------------------------
export function parseCSV(csvContent: string): { success: boolean; message: string; movies: Movie[]; stats?: DatasetStats } {
  if (!csvContent || !csvContent.trim()) {
    return { success: false, message: 'Uploaded CSV file is empty.', movies: [] };
  }

  // Parse lines considering quotes
  const lines: string[] = [];
  let current = '';
  let insideQuote = false;

  for (let i = 0; i < csvContent.length; i++) {
    const char = csvContent[i];
    if (char === '"') {
      insideQuote = !insideQuote;
      current += char;
    } else if ((char === '\n' || char === '\r') && !insideQuote) {
      if (current.trim().length > 0) {
        lines.push(current);
      }
      current = '';
    } else {
      current += char;
    }
  }
  if (current.trim().length > 0) {
    lines.push(current);
  }

  if (lines.length < 2) {
    return { success: false, message: 'CSV must contain at least a header and one record.', movies: [] };
  }

  function parseLine(line: string): string[] {
    const cells: string[] = [];
    let cur = '';
    let inQ = false;
    for (let i = 0; i < line.length; i++) {
      const c = line[i];
      if (c === '"') {
        if (inQ && line[i + 1] === '"') {
          cur += '"';
          i++;
        } else {
          inQ = !inQ;
        }
      } else if (c === ',' && !inQ) {
        cells.push(cur.trim());
        cur = '';
      } else {
        cur += c;
      }
    }
    cells.push(cur.trim());
    return cells;
  }

  const rawHeaders = parseLine(lines[0]);
  const headers = rawHeaders.map(h => h.toLowerCase().replace(/[^a-z0-9_]/g, ''));

  // Alias maps
  const ALIASES: Record<string, string[]> = {
    title: ['title', 'movietitle', 'originaltitle', 'filmtitle', 'moviename', 'name'],
    storyline: ['overview', 'plot', 'description', 'storyline', 'synopsis', 'summary', 'story'],
    genres: ['genres', 'genre', 'genrenames', 'category'],
    year: ['year', 'releaseyear', 'releasedate', 'releasedyear', 'date'],
    director: ['director', 'directors', 'directedby'],
    cast: ['cast', 'actors', 'starring', 'stars'],
    poster_url: ['posterurl', 'posterpath', 'poster', 'imageurl', 'image']
  };

  const detected: Record<string, number> = {};
  const mappingReport: Record<string, string> = {};

  Object.entries(ALIASES).forEach(([canon, aliases]) => {
    for (let i = 0; i < headers.length; i++) {
      if (aliases.includes(headers[i])) {
        detected[canon] = i;
        mappingReport[canon] = rawHeaders[i];
        break;
      }
    }
  });

  if (detected.title === undefined || detected.storyline === undefined) {
    return {
      success: false,
      message: `Missing required column(s). Found: [${rawHeaders.join(', ')}]. Please ensure your CSV includes a movie title and a storyline/overview column.`,
      movies: []
    };
  }

  const movies: Movie[] = [];
  const seenTitles = new Set<string>();
  let duplicates = 0;

  for (let r = 1; r < lines.length; r++) {
    const row = parseLine(lines[r]);
    const titleVal = row[detected.title]?.trim();
    const storyVal = row[detected.storyline]?.trim();

    if (!titleVal || !storyVal || storyVal.length < 15) continue;

    const lowerTitle = titleVal.toLowerCase();
    if (seenTitles.has(lowerTitle)) {
      duplicates++;
      continue;
    }
    seenTitles.add(lowerTitle);

    // Year
    let year: number | null = null;
    if (detected.year !== undefined && row[detected.year]) {
      const match = row[detected.year].match(/\b(19\d{2}|20\d{2})\b/);
      if (match) year = parseInt(match[1], 10);
    }

    // Genres
    let genreList = ['General'];
    if (detected.genres !== undefined && row[detected.genres]) {
      const rawG = row[detected.genres].replace(/[\[\]'"{}]/g, '');
      const spl = rawG.split(/[,/|;]/).map(g => g.trim()).filter(g => g.length > 0);
      if (spl.length) genreList = spl;
    }

    const director = detected.director !== undefined && row[detected.director] ? row[detected.director].trim() : 'Unknown';
    const cast = detected.cast !== undefined && row[detected.cast] ? row[detected.cast].trim() : 'Unknown';

    let poster = '';
    if (detected.poster_url !== undefined && row[detected.poster_url]) {
      const p = row[detected.poster_url].trim();
      if (p.startsWith('http://') || p.startsWith('https://')) poster = p;
    }

    movies.push({
      movie_id: movies.length,
      title: titleVal,
      storyline: storyVal,
      storyline_length: storyVal.length,
      word_count: storyVal.split(/\s+/).length,
      genres: genreList,
      primary_genre: genreList[0] || 'General',
      year,
      director,
      cast,
      poster_url: poster
    });
  }

  if (!movies.length) {
    return { success: false, message: 'No valid movie records found with meaningful storylines.', movies: [] };
  }

  const allGenres = new Set<string>();
  movies.forEach(m => m.genres.forEach(g => allGenres.add(g)));

  const years = movies.map(m => m.year).filter((y): y is number => y !== null);
  const minYear = years.length ? Math.min(...years) : 'N/A';
  const maxYear = years.length ? Math.max(...years) : 'N/A';

  const stats: DatasetStats = {
    total_raw_rows: lines.length - 1,
    valid_movies: movies.length,
    duplicates_removed: duplicates,
    unique_genres_count: allGenres.size,
    unique_genres: Array.from(allGenres).sort(),
    year_range: [minYear, maxYear],
    avg_storyline_words: Math.round(movies.reduce((s, m) => s + m.word_count, 0) / movies.length),
    detected_columns: mappingReport
  };

  return {
    success: true,
    message: `Successfully validated ${movies.length} movies (${duplicates} duplicates filtered).`,
    movies,
    stats
  };
}

// -------------------------------------------------------------
// DYNAMIC CUSTOM MOVIE REGISTRATION
// -------------------------------------------------------------
export function addCustomMovieToCollection(
  currentMovies: Movie[],
  title: string,
  storyline: string,
  genres: string[] = ['Drama'],
  year: number | null = null,
  director: string = 'Custom / Independent',
  cast: string = 'Custom Cast',
  posterUrl: string = ''
): { success: boolean; message: string; movies: Movie[]; stats: DatasetStats; newMovie: Movie } {
  const cleanTitle = title.trim();
  const cleanStory = storyline.trim();

  const existingIdx = currentMovies.findIndex(
    m => m.title.trim().toLowerCase() === cleanTitle.toLowerCase()
  );

  let updatedList = [...currentMovies];
  let movieRecord: Movie;

  if (existingIdx >= 0) {
    movieRecord = {
      ...updatedList[existingIdx],
      storyline: cleanStory,
      storyline_length: cleanStory.length,
      word_count: cleanStory.split(/\s+/).length,
      genres: genres.length ? genres : ['Drama'],
      primary_genre: genres[0] || 'Drama',
      year: year || updatedList[existingIdx].year,
      director: director || updatedList[existingIdx].director,
      cast: cast || updatedList[existingIdx].cast,
      poster_url: posterUrl || updatedList[existingIdx].poster_url
    };
    updatedList[existingIdx] = movieRecord;
  } else {
    const newId = updatedList.length > 0 ? Math.max(...updatedList.map(m => m.movie_id)) + 1 : 0;
    movieRecord = {
      movie_id: newId,
      title: cleanTitle,
      storyline: cleanStory,
      storyline_length: cleanStory.length,
      word_count: cleanStory.split(/\s+/).length,
      genres: genres.length ? genres : ['Drama'],
      primary_genre: genres[0] || 'Drama',
      year: year || null,
      director: director || 'Custom / Independent',
      cast: cast || 'Custom Cast',
      poster_url: posterUrl || ''
    };
    updatedList = [movieRecord, ...updatedList];
  }

  const allGenres = new Set<string>();
  updatedList.forEach(m => m.genres.forEach(g => allGenres.add(g)));
  const years = updatedList.map(m => m.year).filter((y): y is number => y !== null);
  const minYear = years.length ? Math.min(...years) : 'N/A';
  const maxYear = years.length ? Math.max(...years) : 'N/A';

  const stats: DatasetStats = {
    total_raw_rows: updatedList.length,
    valid_movies: updatedList.length,
    duplicates_removed: 0,
    unique_genres_count: allGenres.size,
    unique_genres: Array.from(allGenres).sort(),
    year_range: [minYear, maxYear],
    avg_storyline_words: Math.round(
      updatedList.reduce((s, m) => s + m.word_count, 0) / updatedList.length
    ),
    detected_columns: { title: 'title', storyline: 'storyline' }
  };

  return {
    success: true,
    message: `Custom movie '${cleanTitle}' successfully loaded and ready for analysis!`,
    movies: updatedList,
    stats,
    newMovie: movieRecord
  };
}

// -------------------------------------------------------------
// CURATED INITIAL SAMPLE DATASET (15 Famous Multi-Genre Movies)
// -------------------------------------------------------------
export const SAMPLE_MOVIES_CSV = `title,overview,genres,year,director,cast,poster_url
"Inception","Dom Cobb is a skilled thief who steals valuable corporate secrets from deep within the subconscious during the dream state. His rare ability has made him a coveted player in industrial espionage, but it has also made him an international fugitive and cost him everything he has ever loved. Now Cobb is offered a chance at redemption with one final impossible job: inception, the reverse of theft, planting an idea into the mind of a CEO. If he succeeds with his elite team, it could be the perfect crime, but a dangerous enemy anticipates their every move.","Action, Sci-Fi, Thriller",2010,"Christopher Nolan","Leonardo DiCaprio, Joseph Gordon-Levitt, Elliot Page","https://images.unsplash.com/photo-1536440136628-849c177e76a1?w=500&auto=format&fit=crop&q=60"
"The Shawshank Redemption","Andy Dufresne, a successful banker, is wrongfully convicted of the murder of his wife and her lover and sentenced to two consecutive life terms at Shawshank State Penitentiary. Over the course of nearly two decades, Andy experiences the brutal horrors of prison life, corruption from the sadistic warden, and systemic despair. However, Andy befriends Red, a veteran inmate, and retains an unbreakable sense of hope, intellect, and dignity, secretly planning an ingenious escape that will reclaim his freedom and expose the prison's corruption.","Drama, Crime",1994,"Frank Darabont","Tim Robbins, Morgan Freeman, Bob Gunton","https://images.unsplash.com/photo-1534447677768-be436bb09401?w=500&auto=format&fit=crop&q=60"
"Interstellar","In the mid-21st century, Earth is plagued by devastating dust storms, catastrophic crop blights, and famine, threatening human civilization with imminent extinction. Joseph Cooper, a widowed former NASA pilot and farmer, is recruited for a secret space mission through a mysterious wormhole near Saturn. Leaving his beloved children behind with a promise to return, Cooper and a team of researchers embark on an interstellar voyage across uncharted galaxies to locate a viable habitable planet for human survival, wrestling with gravitational time dilation, betrayal, and the transcendental power of love across space and time.","Sci-Fi, Drama, Adventure",2014,"Christopher Nolan","Matthew McConaughey, Anne Hathaway, Jessica Chastain","https://images.unsplash.com/photo-1506703719100-a0f3a48c0f86?w=500&auto=format&fit=crop&q=60"
"The Dark Knight","With the help of Lieutenant Jim Gordon and newly elected District Attorney Harvey Dent, Batman sets out to dismantle the remaining criminal organizations that plague Gotham City. The partnership proves effective, but they soon find themselves targeted by a sadistic criminal mastermind known as the Joker, who plunges Gotham into terrifying anarchy and chaos. The Joker aims to test Batman's moral limits, forcing the Dark Knight to confront intense psychological dread, tragic personal sacrifice, and the razor-thin line between heroism and vigilantism as Harvey Dent falls from grace.","Action, Crime, Drama",2008,"Christopher Nolan","Christian Bale, Heath Ledger, Aaron Eckhart","https://images.unsplash.com/photo-1509198397868-475647b2a1e5?w=500&auto=format&fit=crop&q=60"
"Parasite","The Kim family lives in a cramped semi-basement apartment in Seoul, struggling with extreme poverty and erratic gig work. Opportunity arrives when the son, Ki-woo, is recommended to tutor the daughter of the affluent Park family. One by one, through cunning deceptions and orchestrated lies, the Kims infiltrate the luxurious Park household as indispensable domestic staff. However, their parasitic symbiotic scheme unravels when a dark, shocking secret hidden deep beneath the wealthy mansion surfaces, triggering a violent clash of social class, pride, and desperation.","Thriller, Drama, Comedy",2019,"Bong Joon-ho","Song Kang-ho, Lee Sun-kyun, Cho Yeo-jeong","https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=500&auto=format&fit=crop&q=60"
"Titanic","Eighty-four years after the historic maritime disaster, Rose DeWitt Bukater narrates her poignant journey aboard the ill-fated RMS Titanic. In 1912, seventeen-year-old aristocrat Rose is trapped in an unhappy engagement to an arrogant steel heir, contemplating ending her life until she is saved by Jack Dawson, a free-spirited, impoverished third-class artist. Despite harsh societal class divisions and her mother's disapproval, Jack and Rose fall deeply and passionately in love. Their forbidden romance turns into a harrowing battle for survival when the luxury liner strikes an iceberg in the freezing North Atlantic ocean, demanding heroic sacrifice.","Romance, Drama",1997,"James Cameron","Leonardo DiCaprio, Kate Winslet, Billy Zane","https://images.unsplash.com/photo-1518709268805-4e9042af9f23?w=500&auto=format&fit=crop&q=60"
"The Matrix","Thomas Anderson is a mild-mannered computer programmer by day and an underground hacker named Neo by night. Constantly questioning his own reality, Neo is contacted by Trinity and the enigmatic rebel leader Morpheus, who reveals a shocking, world-shattering revelation: the reality he knows is a simulated virtual construct called the Matrix, created by malevolent sentient machines to pacify humans while harvesting their bioelectric energy. Awakened into the grim, desolate real world, Neo must embrace his destiny as 'The One', waging war against deadly agents and liberating mankind from digital enslavement.","Sci-Fi, Action",1999,"Lana Wachowski, Lilly Wachowski","Keanu Reeves, Laurence Fishburne, Carrie-Anne Moss","https://images.unsplash.com/photo-1526374965328-7f61d4dc18c5?w=500&auto=format&fit=crop&q=60"
"Spirited Away","While moving to a new suburban neighborhood, sullen ten-year-old Chihiro Ogino and her parents inadvertently stumble into an abandoned amusement park that turns out to be a mystical bathhouse resort for spirits, gods, and supernatural beings. When her greedy parents are transformed into pigs after devouring spirit food, Chihiro must overcome her fears and find employment under the tyrannical sorceress Yubaba. Guided by the mysterious river spirit Haku, Chihiro displays remarkable resilience, loyalty, and empathy, embarking on an extraordinary journey to restore her family and reclaim her stolen identity.","Animation, Fantasy, Adventure",2001,"Hayao Miyazaki","Rumi Hiiragi, Miyu Irino, Mari Natsuki","https://images.unsplash.com/photo-1534447677768-be436bb09401?w=500&auto=format&fit=crop&q=60"
"Pulp Fiction","The lives of two Los Angeles mob hitmen, Vincent Vega and Jules Winnfield, a stylish gangster's wife named Mia Wallace, an aging prizefighter named Butch Coolidge, and a pair of diner bandits intertwine in four bizarre and violent tales of crime and redemption. Vincent must navigate a dangerous evening entertaining his boss's wife without crossing the line into deadly temptation. Meanwhile, Butch betrays the mob by winning a fixed boxing match, sparking a frantic escape, and Jules experiences a divine miracle during a shootout that inspires him to abandon the criminal underworld in pursuit of spiritual righteousness.","Crime, Drama",1994,"Quentin Tarantino","John Travolta, Samuel L. Jackson, Uma Thurman","https://images.unsplash.com/photo-1594909122845-11baa439b7bf?w=500&auto=format&fit=crop&q=60"
"Gladiator","In the year 180 AD, revered Roman general Maximus Decimus Meridius leads his legions to decisive victory against Germanic tribes. Dying Emperor Marcus Aurelius chooses Maximus to succeed him and restore the Roman Republic, but the Emperor's corrupt, jealous son Commodus murders his father and orders the execution of Maximus and his beloved family. Maximus escapes death but arrives too late to save his wife and son, falling into captivity as a slave. Trained as a fierce gladiator, Maximus fights his way to Rome's Colosseum, driven by an unyielding quest for revenge to avenge his murdered family and defy the tyrannical emperor.","Action, Adventure, Drama",2000,"Ridley Scott","Russell Crowe, Joaquin Phoenix, Connie Nielsen","https://images.unsplash.com/photo-1478720568477-152d9b164e26?w=500&auto=format&fit=crop&q=60"
"Whiplash","Andrew Neiman is an ambitious nineteen-year-old jazz drummer studying at the prestigious Shaffer Conservatory in New York City, determined to become one of the all-time greats like Buddy Rich. He is recruited into the top studio ensemble by Terence Fletcher, an extraordinarily ruthless and psychologically abusive conductor who believes in pushing students beyond human endurance through terrifying public humiliation and physical intimidation. As Andrew becomes consumed by singular obsession, he sacrifices his relationships, sanity, and physical well-being, culminating in an electrifying climactic performance that tests the boundary between artistic brilliance and monstrous self-destruction.","Drama, Music",2014,"Damien Chazelle","Miles Teller, J.K. Simmons, Paul Reiser","https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?w=500&auto=format&fit=crop&q=60"
"Coco","In the vibrant town of Santa Cecilia, twelve-year-old Miguel Rivera dreams of becoming an accomplished musician like his idol Ernesto de la Cruz, despite his family's generations-old strict ban on music. Desperate to prove his musical talent, Miguel accidentally crosses into the stunning, colorful Land of the Dead on Dia de los Muertos. Partnering with a charming trickster named Héctor, Miguel embarks on an extraordinary journey through the afterlife to uncover the true mystery behind his family history, the enduring power of unconditional love, and the sacred importance of never forgetting ancestors.","Animation, Family, Fantasy",2017,"Lee Unkrich, Adrian Molina","Anthony Gonzalez, Gael García Bernal, Benjamin Bratt","https://images.unsplash.com/photo-1514306191717-452ec28c7814?w=500&auto=format&fit=crop&q=60"
"The Prestige","In late 19th-century Victorian London, two ambitious stage magicians and former partners, Robert Angier and Alfred Borden, become bitter rivals following a tragic onstage accident that claims the life of Angier's wife. Their competition escalates into a fierce, obsessive feud to create the ultimate stage illusion: the Teleported Man. As Borden masters the trick through mysterious discipline, Angier turns to brilliant inventor Nikola Tesla to build a dangerous scientific apparatus. Their escalating rivalry drives them to betrayal, murder, and unimaginable sacrifice as both men surrender their morality for theatrical glory.","Drama, Mystery, Sci-Fi",2006,"Christopher Nolan","Hugh Jackman, Christian Bale, Scarlett Johansson","https://images.unsplash.com/photo-1517604931442-7e0c8ed2963c?w=500&auto=format&fit=crop&q=60"
"Forrest Gump","Sitting on a park bench in Savannah, Georgia, Forrest Gump, a gentle and kind-hearted man with a low IQ, recounts his remarkable life story to listening strangers. From running across Alabama as a boy to becoming a college football star, decorated Vietnam war hero, ping-pong champion, and shrimp boat entrepreneur, Forrest inadvertently witnesses and influences monumental historical events of the late 20th century. Through decades of triumph, hardship, and loss, Forrest remains loyal to his mother's wise lessons and steadfast in his lifelong, unrequited love for his troubled childhood sweetheart, Jenny Curran.","Drama, Romance",1994,"Robert Zemeckis","Tom Hanks, Robin Wright, Gary Sinise","https://images.unsplash.com/photo-1485846234645-a62644f84728?w=500&auto=format&fit=crop&q=60"
"Alien","The commercial deep-space towing vessel Nostromo is returning to Earth when its computer system intercepts a mysterious transmission from an uncharted moon. The crew is awakened from stasis to investigate the desolate alien moon, where executive officer Kane discovers a derelict extraterrestrial vessel containing thousands of organic eggs. A predatory parasitic organism attaches to Kane's face, and despite quarantine protocols, he is brought back aboard. Soon, a ferocious, acid-blooded alien creature bursts into the ship, stalking and slaughtering the isolated crew one by one in dark corridors as warrant officer Ellen Ripley fights for survival.","Horror, Sci-Fi",1979,"Ridley Scott","Sigourney Weaver, Tom Skerritt, John Hurt","https://images.unsplash.com/photo-1509281373149-e957c6296406?w=500&auto=format&fit=crop&q=60"`;
