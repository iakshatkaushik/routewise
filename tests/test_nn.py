import numpy as np
import pytest

torch = pytest.importorskip("torch")

from llm_router.nn import NumpyMLP, train_mlp  # noqa: E402


def test_numpy_mlp_learns_and_matches_shapes():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3000, 10)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0.8).astype(np.float32)
    m, hist, ep = train_mlp(X[:2400], y[:2400], X[2400:], y[2400:], hidden=(32,), lr=1e-2, max_epochs=60, patience=5)
    p = m.predict_proba(X[2400:])
    assert p.shape == (600, 2) and np.allclose(p.sum(1), 1)
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y[2400:], p[:, 1]) > 0.95


def test_numpy_forward_equals_manual():
    W1, b1 = np.eye(3, 2, dtype=np.float32), np.zeros(2, np.float32)
    W2, b2 = np.ones((2, 1), np.float32), np.array([-1.0], np.float32)
    m = NumpyMLP([W1, W2], [b1, b2], "relu")
    x = np.array([[1.0, -2.0, 5.0]], np.float32)
    logit = max(1, 0) + max(-2, 0) - 1.0
    assert m.predict_proba(x)[0, 1] == pytest.approx(1 / (1 + np.exp(-logit)))
