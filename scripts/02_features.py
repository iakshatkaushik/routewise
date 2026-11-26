"""Step 2 - feature engineering ablation and caching (Data Transformation / Reduction / Scaling).

The ablation uses only the training data (80/20 split inside train), never the test set.
The winning configuration is fitted on all training data and cached to data/processed/.
"""
import joblib
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from _common import Timer, save_json
from llm_router import config, data
from llm_router.features import PromptFeaturizer

CONFIGS = {
    "word_only":        dict(char_max_features=1, stats_weight=0.0),
    "char_only":        dict(word_max_features=1, stats_weight=0.0),
    "word+char":        dict(stats_weight=0.0),
    "word+char+stats":  dict(),
    "small_vocab_20k":  dict(word_max_features=20000, char_max_features=20000),
    "large_vocab_150k": dict(word_max_features=150000, char_max_features=150000),
}


def score(cfg, tr, va):
    f = PromptFeaturizer(**cfg).fit(tr.prompt)
    Xtr, Xva = f.transform(tr.prompt), f.transform(va.prompt)
    m = LogisticRegression(C=0.5, max_iter=3000, class_weight="balanced").fit(Xtr, tr.label)
    return roc_auc_score(va.label, m.predict_proba(Xva)[:, 1]), Xtr.shape[1]


if __name__ == "__main__":
    train, test = data.load("train"), data.load("test")
    tr, va = train_test_split(train, test_size=0.2, random_state=config.SEED, stratify=train.label)
    results = {}
    for name, cfg in CONFIGS.items():
        with Timer() as t:
            auc, nfeat = score(cfg, tr, va)
        results[name] = {"val_auc": round(auc, 4), "n_features": nfeat, "seconds": round(t.s, 1)}
        print(name, results[name])
    best = max(["word+char+stats", "small_vocab_20k", "large_vocab_150k"], key=lambda k: results[k]["val_auc"])
    # prefer the smaller model when the AUC difference is negligible (< 0.002)
    if results["small_vocab_20k"]["val_auc"] >= results[best]["val_auc"] - 0.002:
        best = "small_vocab_20k"
    print("chosen:", best)

    feat = PromptFeaturizer(**CONFIGS[best]).fit(train.prompt)
    Xtr, Xte = feat.transform(train.prompt), feat.transform(test.prompt)
    sp.save_npz(config.DATA_PROCESSED / "X_train.npz", Xtr)
    sp.save_npz(config.DATA_PROCESSED / "X_test.npz", Xte)
    # Data reduction: 256-dim dense representation for models that need dense input
    # (decision tree, random forest, K-NN, neural network, clustering).
    with Timer() as t:
        svd = TruncatedSVD(256, algorithm="randomized", random_state=config.SEED).fit(Xtr)
    Ztr, Zte = svd.transform(Xtr).astype(np.float32), svd.transform(Xte).astype(np.float32)
    np.save(config.DATA_PROCESSED / "Z_train.npy", Ztr)
    np.save(config.DATA_PROCESSED / "Z_test.npy", Zte)
    joblib.dump({"featurizer": feat, "svd": svd, "config_name": best}, config.MODELS / "featurizer.joblib", compress=3)
    save_json("02_features.json", {"ablation": results, "chosen": best, "n_features": int(Xtr.shape[1]),
                                   "svd_components": 256,
                                   "svd_explained_variance": float(svd.explained_variance_ratio_.sum()),
                                   "svd_seconds": round(t.s, 1)})
