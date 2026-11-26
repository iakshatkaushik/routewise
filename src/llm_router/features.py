"""Feature engineering (syllabus units: Data Transformation, Feature Scaling, Data Reduction).

A prompt becomes three blocks of features:
- word TF-IDF (unigrams + bigrams): what the prompt is about;
- character TF-IDF (3-5 grams): code, maths symbols, other languages, misspellings;
- 24 hand-made "prompt statistics" (length, code/maths markers, constraints ...), standardised.
"""
from __future__ import annotations

import math
import re

import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

_R = {
    "code": re.compile(r"```|\bdef |\bclass |\bimport |\bfunction\b|#include|\bpublic static\b|\bSELECT\b|\breturn\b|[{};]\s*$|=>|\bconsole\.|\bprint\(", re.I | re.M),
    "code_word": re.compile(r"\b(code|python|java|javascript|c\+\+|sql|function|script|program|regex|api|bug|error|compile)\b", re.I),
    "math": re.compile(r"\d\s*[-+*/^=]\s*\d|\b(solve|calculate|equation|integral|derivative|probability|prove|sum of|how many)\b|[∫√π∑≤≥]", re.I),
    "number": re.compile(r"\d+(\.\d+)?"),
    "url": re.compile(r"https?://|www\."),
    "wh": re.compile(r"^\s*(what|why|how|who|when|where|which|is|are|can|do|does|should)\b", re.I),
    "imperative": re.compile(r"^\s*(write|create|generate|make|give|list|explain|describe|summari[sz]e|translate|rewrite|compose|draft|tell)\b", re.I),
    "creative": re.compile(r"\b(story|poem|song|lyrics|essay|joke|novel|character|roleplay|role-play|pretend|imagine)\b", re.I),
    "constraint": re.compile(r"\b(must|exactly|at least|at most|no more than|only|do not|don't|without|format|words|sentences|bullet|json|table)\b", re.I),
    "reasoning": re.compile(r"\b(step by step|reason|logic|puzzle|riddle|explain why|analy[sz]e|compare|pros and cons)\b", re.I),
    "translate": re.compile(r"\b(translate|translation|in (french|spanish|german|chinese|hindi|japanese|russian))\b", re.I),
    "options": re.compile(r"^\s*(\(?[a-dA-D][).:]|[0-9]+[).])\s", re.M),
    "context": re.compile(r"\b(input|context|passage|article|text|document|paragraph|definition|task)\s*:", re.I),
}

STAT_NAMES = [
    "log_chars", "log_words", "n_lines", "avg_word_len", "frac_digit", "frac_upper",
    "frac_punct", "frac_non_ascii", "n_question_marks", "starts_wh_question",
    "starts_imperative", "code_markers", "code_words", "math_markers", "n_numbers",
    "has_url", "creative_words", "constraint_words", "reasoning_words", "translate",
    "mcq_options", "has_context_block", "n_sentences", "frac_space",
]


def prompt_stats(texts) -> np.ndarray:
    """Return an (n, 24) float32 matrix of hand-made features."""
    rows = []
    for t in texts:
        n = max(len(t), 1)
        words = t.split()
        nw = max(len(words), 1)
        rows.append([
            math.log1p(len(t)),
            math.log1p(len(words)),
            min(t.count("\n") + 1, 100),
            sum(map(len, words)) / nw,
            sum(c.isdigit() for c in t) / n,
            sum(c.isupper() for c in t) / n,
            sum((not c.isalnum()) and (not c.isspace()) for c in t) / n,
            sum(ord(c) > 127 for c in t) / n,
            min(t.count("?"), 20),
            1.0 if _R["wh"].search(t) else 0.0,
            1.0 if _R["imperative"].search(t) else 0.0,
            min(len(_R["code"].findall(t)), 50),
            min(len(_R["code_word"].findall(t)), 20),
            min(len(_R["math"].findall(t)), 20),
            math.log1p(len(_R["number"].findall(t))),
            1.0 if _R["url"].search(t) else 0.0,
            min(len(_R["creative"].findall(t)), 10),
            min(len(_R["constraint"].findall(t)), 20),
            min(len(_R["reasoning"].findall(t)), 10),
            1.0 if _R["translate"].search(t) else 0.0,
            min(len(_R["options"].findall(t)), 10),
            1.0 if _R["context"].search(t) else 0.0,
            min(len(re.findall(r"[.!?](\s|$)", t)), 100),
            sum(c.isspace() for c in t) / n,
        ])
    return np.asarray(rows, dtype=np.float32)


class FeatureNames:
    """Memory-light list of feature names: one UTF-8 blob plus int32 offsets.

    A NumPy unicode array of 300k names takes about 67 MB (fixed 56-char width x 4 bytes);
    this takes about 5 MB.
    """

    def __init__(self, names):
        encoded = [str(n).encode("utf-8") for n in names]
        self.offsets = np.zeros(len(encoded) + 1, dtype=np.int64)
        np.cumsum([len(e) for e in encoded], out=self.offsets[1:])
        self.blob = b"".join(encoded)

    def __len__(self):
        return len(self.offsets) - 1

    def __getitem__(self, i):
        i = int(i)
        return self.blob[self.offsets[i]:self.offsets[i + 1]].decode("utf-8")


class CompactProjector:
    """Deployment-sized replacement for a full TruncatedSVD (Data Reduction).

    A 256-component SVD over 300k sparse columns stores 307 MB of weights. This keeps only the
    `cols` most informative columns (chosen by the caller, e.g. by |logistic coefficient|) and
    fits the SVD on them, so the projection matrix is n_components x len(cols) in float32.
    """

    def __init__(self, cols, n_components=128, seed=42):
        self.cols = np.asarray(cols, dtype=np.int64)
        self.n_components = n_components
        self.seed = seed

    def fit(self, X):
        from sklearn.decomposition import TruncatedSVD
        svd = TruncatedSVD(self.n_components, algorithm="randomized", random_state=self.seed)
        svd.fit(X[:, self.cols])
        self.components_ = svd.components_.astype(np.float32)
        self.explained_variance_ratio_ = svd.explained_variance_ratio_
        return self

    def transform(self, X):
        return np.asarray(X[:, self.cols] @ self.components_.T, dtype=np.float32)


class PromptFeaturizer(BaseEstimator, TransformerMixin):
    """Turns raw prompts into one sparse matrix: [word TF-IDF | char TF-IDF | scaled stats]."""

    def __init__(self, word_max_features=60000, char_max_features=60000,
                 word_ngrams=(1, 2), char_ngrams=(3, 5), min_df=3, stats_weight=1.0):
        self.word_max_features = word_max_features
        self.char_max_features = char_max_features
        self.word_ngrams = word_ngrams
        self.char_ngrams = char_ngrams
        self.min_df = min_df
        self.stats_weight = stats_weight

    def fit(self, texts, y=None):
        texts = list(texts)
        self.word_ = TfidfVectorizer(ngram_range=self.word_ngrams, min_df=self.min_df,
                                     max_features=self.word_max_features, sublinear_tf=True,
                                     strip_accents="unicode", dtype=np.float32)
        self.char_ = TfidfVectorizer(analyzer="char_wb", ngram_range=self.char_ngrams,
                                     min_df=self.min_df, max_features=self.char_max_features,
                                     sublinear_tf=True, dtype=np.float32)
        self.word_.fit(texts)
        self.char_.fit(texts)
        self.scaler_ = StandardScaler().fit(prompt_stats(texts))
        return self

    def transform_parts(self, texts):
        texts = list(texts)
        stats = self.scaler_.transform(prompt_stats(texts)).astype(np.float32)
        return self.word_.transform(texts), self.char_.transform(texts), stats

    def transform(self, texts):
        w, c, s = self.transform_parts(texts)
        s = np.clip(s, -5, 5) * self.stats_weight
        return sp.hstack([w, c, sp.csr_matrix(s)], format="csr", dtype=np.float32)

    def feature_names(self) -> np.ndarray:
        return np.concatenate([
            ["word: " + f for f in self.word_.get_feature_names_out()],
            ["chars: '" + f + "'" for f in self.char_.get_feature_names_out()],
            ["stat: " + f for f in STAT_NAMES],
        ])
