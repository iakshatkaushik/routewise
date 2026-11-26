import numpy as np
import pytest

from llm_router import metrics


def test_curve_endpoints_and_baselines():
    rng = np.random.default_rng(0)
    cheap = rng.choice([1, 2, 3, 4, 5], size=5000, p=[.05, .05, .05, .15, .7]).astype(float)
    x, q, pgr = metrics.routing_curve(rng.random(5000), cheap)
    assert pgr[0] == pytest.approx(0.0)
    assert pgr[-1] == pytest.approx(1.0)
    assert q[0] == pytest.approx(cheap.mean())
    base = metrics.oracle_and_random(cheap)
    assert base["random_apgr"] == pytest.approx(0.5, abs=0.02)
    assert base["oracle"]["apgr"] > 0.85
    # a router that is exactly backwards is worse than random
    assert metrics.routing_report(cheap, cheap)["apgr"] < 0.2


def test_classification_report_counts():
    y = np.array([1, 1, 0, 0, 0])
    p = np.array([0.9, 0.2, 0.8, 0.1, 0.1])
    r = metrics.classification_report(y, p, 0.5)
    assert r["confusion_matrix"] == {"tn": 2, "fp": 1, "fn": 1, "tp": 1}
    assert r["recall_sensitivity"] == pytest.approx(0.5)
    assert r["specificity"] == pytest.approx(2 / 3)
