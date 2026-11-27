"""Inference-time router: prompt -> P(cheap model fails) -> cheap or strong model.

The trained artifact (models/router.joblib) holds
- the fitted featurizer and SVD projection,
- one or more component models and a logistic "stacker" that combines them. A component reads the
  TF-IDF features, the SVD projection, sentence embeddings, or the raw text (the fine-tuned transformer,
  run with ONNX Runtime from models/text_router.onnx), and gives the stacker a log-odds, a logit, or
  minus a predicted 1-5 score,
- a table mapping "share of traffic sent to the strong model" to a risk threshold,
- the test-set routing curve (used to state expected quality at each budget),
- linear weights used to explain a decision word by word.
"""
from __future__ import annotations

import os
import time
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

from . import config
from .features import STAT_NAMES, prompt_stats


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class Router:
    """mode="full" uses every component; mode="light" uses the artifact's light variant: sparse
    (TF-IDF) components only, so neither the ONNX text model nor an embedding runtime is loaded
    (about 114 MB less memory on the server) at an accuracy cost."""

    def __init__(self, artifact: dict, mode: str | None = None):
        mode = (mode or os.getenv("ROUTER_MODE", "full")).lower()
        if mode == "light" and "light" in artifact:
            artifact = {**artifact, **artifact["light"]}
        self.mode = artifact["meta"].get("mode", "full")
        self.a = artifact
        self.featurizer = artifact["featurizer"]
        self.svd = artifact["svd"]
        self.components = artifact["components"]
        self.stacker = artifact["stacker"]
        self.shares = np.asarray(artifact["share_thresholds"]["shares"])
        self.thresholds = np.asarray(artifact["share_thresholds"]["thresholds"])
        self.curve = artifact["curve"]
        self.meta = artifact["meta"]
        self._names = artifact.get("feature_names")
        self._coef = artifact.get("explain_coef")

    @classmethod
    def load(cls, path: str | Path | None = None, mode: str | None = None) -> "Router":
        return cls(joblib.load(path or config.MODELS / "router.joblib"), mode)

    # ---- scoring -------------------------------------------------------------------------
    def _features(self, prompts) -> dict:
        kinds = {c["input"] for c in self.components}
        feats = {"sparse": self.featurizer.transform(prompts)}
        if "dense" in kinds:
            feats["dense"] = self.svd.transform(feats["sparse"]).astype(np.float32)
        if "embed" in kinds:
            from .embedder import embed
            stats = self.featurizer.scaler_.transform(prompt_stats(prompts)).clip(-5, 5)
            feats["embed"] = np.hstack([embed(prompts), stats]).astype(np.float32)
        if "text" in kinds:
            from . import text_model
            spec = self.a["text_model"]
            feats["text"] = text_model.load(spec["onnx"], spec["tokenizer"], spec["max_tokens"]).logits(prompts)
        return feats

    @staticmethod
    def _component_signal(c, feats) -> np.ndarray:
        """One stacker input per component: a log-odds, or minus a predicted 1-5 score."""
        kind = c.get("output", "proba")
        if kind == "text_logit":                       # fine-tuned transformer head (already a logit)
            return feats["text"][:, c["head"]]
        if kind == "score":                            # regressor of the cheap model's score
            return -c["model"].predict(feats[c["input"]])
        return _logit(c["model"].predict_proba(feats[c["input"]])[:, 1])

    def risk(self, prompts) -> np.ndarray:
        """Calibrated probability that the cheap model's answer is poor (score <= 3)."""
        prompts = [str(p)[: config.MAX_PROMPT_CHARS] for p in prompts]
        feats = self._features(prompts)
        cols = [self._component_signal(c, feats) for c in self.components]
        return self.stacker.predict_proba(np.column_stack(cols))[:, 1]

    def threshold_for(self, strong_share: float) -> float:
        strong_share = float(np.clip(strong_share, 0.0, 1.0))
        return float(np.interp(strong_share, self.shares, self.thresholds))

    def expected_at(self, strong_share: float) -> dict:
        x = np.asarray(self.curve["x"])
        return {"pgr": float(np.interp(strong_share, x, self.curve["pgr"])),
                "quality": float(np.interp(strong_share, x, self.curve["quality"]))}

    # ---- explanation ---------------------------------------------------------------------
    def explain(self, prompt: str, top: int = 6) -> dict:
        if self._coef is None:
            return {"towards_strong": [], "towards_cheap": []}
        x = self.featurizer.transform([prompt]).tocsr()
        contrib = x.data * self._coef[x.indices]
        order = np.argsort(contrib)
        def pick(ids):
            return [{"feature": str(self._names[x.indices[i]]), "weight": round(float(contrib[i]), 4)}
                    for i in ids if abs(contrib[i]) > 1e-4]
        return {"towards_strong": pick(order[::-1][:top]), "towards_cheap": pick(order[:top])}

    # ---- main entry point ----------------------------------------------------------------
    def route(self, prompt: str, strong_share: float = 0.3, explain: bool = True) -> dict:
        t = time.perf_counter()
        p = float(self.risk([prompt])[0])
        thr = self.threshold_for(strong_share)
        latency_ms = (time.perf_counter() - t) * 1000
        stats = dict(zip(STAT_NAMES, prompt_stats([prompt])[0].round(3).tolist()))
        out = {
            "route": "strong" if p >= thr else "cheap",
            "p_cheap_fails": round(p, 4),
            "threshold": round(thr, 4),
            "strong_share": strong_share,
            "expected": self.expected_at(strong_share),
            "router_latency_ms": round(latency_ms, 2),
            "prompt_stats": {k: stats[k] for k in ("log_chars", "code_markers", "math_markers",
                                                   "constraint_words", "reasoning_words", "n_question_marks")},
        }
        if explain:
            out["explanation"] = self.explain(prompt)
        return out


@lru_cache(maxsize=1)
def get_router() -> Router:
    return Router.load()
