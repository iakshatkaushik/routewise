"""Step 3b - neural network on sentence embeddings, trained on the GPU (ANN unit).

Input: 384-dim bge-small sentence embedding + the 24 standardised prompt statistics.
Model: PyTorch MLP trained on the GPU with class-balanced BCE loss,
AdamW, cosine learning-rate schedule and early stopping on validation AUC.
Same protocol as step 3: tune on the validation split, refit on all training data for the
best number of epochs, score once on test. Results are appended to step 3's reports.
"""
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from _common import Timer, save_json
from llm_router import config, data, metrics
from llm_router.nn import torch_device, train_mlp

NAME = "Embedding Neural Network (GPU)"

if __name__ == "__main__":
    train, test = data.load("train"), data.load("test")
    feat = joblib.load(config.MODELS / "featurizer.joblib")["featurizer"]
    from llm_router.features import prompt_stats
    S = feat.scaler_.transform(prompt_stats(train.prompt)).clip(-5, 5)
    St = feat.scaler_.transform(prompt_stats(test.prompt)).clip(-5, 5)
    X = np.hstack([np.load(config.DATA_PROCESSED / "E_train.npy"), S]).astype(np.float32)
    Xt = np.hstack([np.load(config.DATA_PROCESSED / "E_test.npy"), St]).astype(np.float32)
    y, yt = train.label.values, test.label.values
    idx_tr, idx_va = train_test_split(np.arange(len(y)), test_size=0.2, random_state=config.SEED, stratify=y)
    print("device:", torch_device(), "input dims:", X.shape[1], flush=True)

    grid = [dict(hidden=h, dropout=d, lr=lr) for h in [(512,), (1024, 256)] for d in [0.2, 0.4] for lr in [1e-3, 3e-4]]
    tried, best = [], None
    for g in grid:
        with Timer() as t:
            m, hist, ep = train_mlp(X[idx_tr], y[idx_tr], X[idx_va], y[idx_va], **g)
        auc = roc_auc_score(y[idx_va], m.predict_proba(X[idx_va])[:, 1])
        tried.append({**{k: str(v) for k, v in g.items()}, "val_auc": round(auc, 4), "best_epoch": ep, "seconds": round(t.s, 1)})
        print(" ", tried[-1], flush=True)
        if best is None or auc > best[0]:
            best = (auc, g, ep, m)
    auc_va, g, ep, m_val = best

    act = {}
    for a in ["logistic", "tanh", "relu", "gelu"]:
        m_a, hist, ep_a = train_mlp(X[idx_tr], y[idx_tr], X[idx_va], y[idx_va], **{**g, "activation": a})
        act[a] = {"val_auc": round(roc_auc_score(y[idx_va], m_a.predict_proba(X[idx_va])[:, 1]), 4), "epochs": ep_a}
        print("activation", a, act[a], flush=True)

    with Timer() as t_fit:
        model, _, _ = train_mlp(X, y, None, None, **g, fixed_epochs=ep)
    t = time.perf_counter()
    s_te = model.predict_proba(Xt)[:, 1]
    pred_s = time.perf_counter() - t
    s_va = m_val.predict_proba(X[idx_va])[:, 1]
    thr = float(np.quantile(s_va, 0.70))
    rep = metrics.classification_report(yt, s_te, thr)
    rout = metrics.routing_report(s_te, test.score.values)
    print(f"==> {NAME}: test AUC={rep['roc_auc']:.4f} APGR={rout['apgr']:.4f}", flush=True)
    joblib.dump(model, config.MODELS / "clf_embedding_neural_network_gpu.joblib", compress=3)

    # append to step 3's reports so every later step sees one combined comparison
    path = config.REPORTS / "03_classifiers.json"
    c3 = json.loads(path.read_text())
    c3["models"][NAME] = {"best_params": {k: str(v) for k, v in g.items()} | {"epochs": str(ep), "device": torch_device()},
                          "val_auc": round(auc_va, 4), "test": rep, "routing": rout,
                          "fit_seconds": round(t_fit.s, 1),
                          "predict_ms_per_1k": round(1000 * pred_s / len(yt) * 1000, 2),
                          "trained_on_rows": int(len(y)), "input": "384-dim bge-small embedding + 24 prompt stats"}
    c3["activation_functions_gpu_network"] = act
    path.write_text(json.dumps(c3, indent=2))
    for fname, col in (("val_scores.parquet", s_va), ("test_scores.parquet", s_te)):
        df = pd.read_parquet(config.REPORTS / fname)
        df[NAME] = col
        df.to_parquet(config.REPORTS / fname)
    save_json("03b_gpu_network.json", {"grid": tried, "best": {k: str(v) for k, v in g.items()}, "best_epoch": ep,
                                       "activation_functions": act, "test_auc": rep["roc_auc"], "test_apgr": rout["apgr"]})
