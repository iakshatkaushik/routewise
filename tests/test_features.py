import numpy as np

from llm_router.data import clean_text
from llm_router.features import STAT_NAMES, PromptFeaturizer, prompt_stats


def test_stats_shape_and_signals():
    s = prompt_stats(["hi", "```python\ndef f(x):\n    return x\n```", "Solve 2 + 2 = ?"])
    assert s.shape == (3, len(STAT_NAMES))
    col = {n: s[:, i] for i, n in enumerate(STAT_NAMES)}
    assert col["code_markers"][1] > 0 and col["code_markers"][0] == 0
    assert col["math_markers"][2] > 0


def test_clean_text():
    assert clean_text("  a\x00b \t c\r\n\n\n\nd ") == "ab c\n\nd"


def test_featurizer_round_trip():
    texts = ["write a poem about cats", "fix this python bug", "what is 2+2", "translate hello to french"] * 5
    f = PromptFeaturizer(min_df=1).fit(texts)
    X = f.transform(texts)
    assert X.shape[0] == len(texts)
    assert X.shape[1] == len(f.feature_names())
    assert np.isfinite(X.data).all()
