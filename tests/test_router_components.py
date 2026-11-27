"""Router component kinds ("proba", "score", "text_logit") on small synthetic artifacts, plus the
fine-tuned text model's ONNX path when the exported file is present."""
import json

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression, Ridge

from llm_router import config
from llm_router.features import PromptFeaturizer
from llm_router.router import Router, _logit

TEXTS = ["write a poem about cats", "fix this python bug: def f(x): return x", "what is 2+2",
         "translate hello to french", "prove that sqrt(2) is irrational", "hi there"] * 6
Y = np.array([0, 1, 0, 0, 1, 0] * 6)
SCORE = np.array([5, 2, 5, 4, 1, 5] * 6, dtype=float)


def _artifact(components, stacker):
    shares = np.linspace(0, 1, 11)
    return {"featurizer": FEAT, "svd": None, "components": components, "stacker": stacker,
            "share_thresholds": {"shares": shares.tolist(), "thresholds": (1 - shares).tolist()},
            "curve": {"x": shares.tolist(), "quality": shares.tolist(), "pgr": shares.tolist()},
            "meta": {"mode": "full", "components": [c["name"] for c in components]}}


FEAT = PromptFeaturizer(min_df=1).fit(TEXTS)
X = FEAT.transform(TEXTS)


def test_score_component_uses_minus_predicted_score():
    ridge = Ridge(alpha=1.0).fit(X, SCORE)
    lr = LogisticRegression().fit(X, Y)
    S = np.column_stack([_logit(lr.predict_proba(X)[:, 1]), -ridge.predict(X)])
    stacker = LogisticRegression().fit(S, Y)
    comps = [{"name": "LR", "input": "sparse", "model": lr},                       # default output: proba
             {"name": "Ridge", "input": "sparse", "output": "score", "model": ridge}]
    r = Router(_artifact(comps, stacker), mode="full")
    np.testing.assert_allclose(r.risk(TEXTS), stacker.predict_proba(S)[:, 1], rtol=1e-10)
    assert r.route("prove that pi is irrational", 0.5)["route"] in {"cheap", "strong"}


class _FakeText:
    """Stands in for the ONNX text model: deterministic two-column logits from the prompt length."""
    def logits(self, prompts):
        n = np.array([len(p) for p in prompts], dtype=np.float32)
        return np.column_stack([n / 10 - 2, 1 - n / 20]).astype(np.float32)


def test_text_logit_component_reads_the_chosen_head(monkeypatch):
    from llm_router import text_model
    monkeypatch.setattr(text_model, "load", lambda *a, **k: _FakeText())
    L = _FakeText().logits(TEXTS)
    stacker = LogisticRegression().fit(L[:, [1, 0]], Y)
    comps = [{"name": "deficit head", "input": "text", "output": "text_logit", "head": 1, "model": None},
             {"name": "fail head", "input": "text", "output": "text_logit", "head": 0, "model": None}]
    art = _artifact(comps, stacker) | {"text_model": {"onnx": "x.onnx", "tokenizer": "x.json", "max_tokens": 8}}
    r = Router(art, mode="full")
    np.testing.assert_allclose(r.risk(TEXTS), stacker.predict_proba(L[:, [1, 0]])[:, 1], rtol=1e-6)


ONNX = config.MODELS / "text_router.onnx"
TOK = config.MODELS / "text_tokenizer.json"
REF = config.ROOT / "tests" / "fixtures" / "text_model_reference.json"


@pytest.mark.skipif(not (ONNX.exists() and TOK.exists() and REF.exists()),
                    reason="exported text model not present (run scripts/07a_export_text_model.py)")
def test_text_model_onnx_matches_stored_reference():
    pytest.importorskip("onnxruntime")
    pytest.importorskip("tokenizers")
    from llm_router.text_model import TextRouterModel
    ref = json.loads(REF.read_text())
    m = TextRouterModel(ONNX, TOK, ref["max_tokens"], threads=1)
    # int8 kernels differ slightly between CPU types (x86 vs ARM): about 0.02 in logit units
    np.testing.assert_allclose(m.logits(ref["prompts"]), np.array(ref["logits"]), atol=0.05)
