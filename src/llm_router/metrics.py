"""Evaluation metrics.

Classification (syllabus): sensitivity/recall, specificity, precision, F1, accuracy, ROC-AUC.
Regression (syllabus): RMSE, MAE, Pearson correlation, R-squared.
Routing (from the RouteLLM paper, Ong et al. 2024):
- PGR(x): "performance gap recovered" when x% of prompts go to the strong model,
  (quality - cheap_only_quality) / (strong_only_quality - cheap_only_quality).
- APGR: area under the PGR curve over x in [0, 1]. A random router scores 0.5.
- CPT(p): the smallest share of strong-model calls that reaches PGR = p.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score,
                             roc_auc_score)

from . import config


def classification_report(y_true, proba, threshold) -> dict:
    y_pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def regression_report(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "pearson_r": float(np.corrcoef(y_true, y_pred)[0, 1]),
        "r2": float(r2_score(y_true, y_pred)),
    }


def routing_curve(risk, cheap_scores, strong_score=config.STRONG_SCORE, points=101):
    """Quality when the top-x share of prompts by `risk` go to the strong model.

    Returns (x, quality, pgr) arrays with x from 0 to 1.
    """
    risk = np.asarray(risk, float)
    cheap = np.asarray(cheap_scores, float)
    n = len(risk)
    order = np.argsort(-risk, kind="stable")
    # gain from sending prompt i to the strong model, accumulated in risk order
    gain = np.concatenate([[0.0], np.cumsum(strong_score - cheap[order])])
    x = np.linspace(0, 1, points)
    k = np.round(x * n).astype(int)
    quality = (cheap.sum() + gain[k]) / n
    q_cheap, q_strong = cheap.mean(), strong_score
    pgr = (quality - q_cheap) / (q_strong - q_cheap)
    return x, quality, pgr


def routing_report(risk, cheap_scores, strong_score=config.STRONG_SCORE) -> dict:
    x, quality, pgr = routing_curve(risk, cheap_scores, strong_score)

    def cpt(target):
        idx = np.argmax(pgr >= target)
        return float(x[idx]) if pgr[idx] >= target else 1.0

    def at(share):
        i = int(round(share * (len(x) - 1)))
        return {"strong_share": float(x[i]), "quality": float(quality[i]), "pgr": float(pgr[i])}

    return {
        "apgr": float(np.trapezoid(pgr, x)),
        "cpt50": cpt(0.5),
        "cpt80": cpt(0.8),
        "cpt90": cpt(0.9),
        "quality_cheap_only": float(quality[0]),
        "quality_strong_only": float(quality[-1]),
        "at_20pct": at(0.2), "at_30pct": at(0.3), "at_50pct": at(0.5),
    }


def oracle_and_random(cheap_scores, strong_score=config.STRONG_SCORE) -> dict:
    cheap = np.asarray(cheap_scores, float)
    oracle = routing_report(strong_score - cheap, cheap, strong_score)
    rng = np.random.default_rng(config.SEED)
    rand = [routing_report(rng.random(len(cheap)), cheap, strong_score)["apgr"] for _ in range(20)]
    return {"oracle": oracle, "random_apgr": float(np.mean(rand))}
