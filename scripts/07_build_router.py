"""Step 7 - build, optimise and export the production router.

1. Candidate components are the tuned classifiers from step 3 that can run cheaply online
   (K-NN and Random Forest are excluded: K-NN must keep all 109k training vectors in memory
   and the forest is hundreds of MB).
2. Greedy forward selection: start from the best single model and add a component only if
   the stacked validation AUC improves by at least 0.002.
3. A logistic-regression stacker combines the components' log-odds into one calibrated
   probability "P(cheap model fails)". It is trained on validation predictions only.
4. Budget table: for every strong-model share s in 0..100%, the risk threshold is the
   (1 - s) quantile of validation risks. The test set checks that the realised share matches.
5. Everything is scored once on the untouched test set and exported to models/router.joblib.
"""
import time

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_predict

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from _common import load_json, save_json, slug  # noqa: E402
from llm_router import config, data, metrics  # noqa: E402
from llm_router.features import FeatureNames  # noqa: E402
from llm_router.router import Router, _logit  # noqa: E402

COMPACT_COLS, COMPACT_DIMS = 40000, 128

DEPLOYABLE = {  # result name in step 3 -> (saved model file, input kind)
    "Logistic Regression": (None, "sparse"),
    "Neural Network (MLP)": (None, "dense"),
    "Gradient Boosting (beyond syllabus)": (None, "dense"),
    "Multinomial Naive Bayes": (None, "sparse_nb"),
    "Decision Tree (Entropy / information gain)": (None, "dense"),
    "Embedding Neural Network (GPU)": (None, "embed"),
}


def stacked_val_auc(cols, y):
    S = np.column_stack([_logit(c) for c in cols])
    p = cross_val_predict(LogisticRegression(), S, y, cv=5, method="predict_proba")[:, 1]
    return roc_auc_score(y, p)


if __name__ == "__main__":
    train, test = data.load("train"), data.load("test")
    val = pd.read_parquet(config.REPORTS / "val_scores.parquet")
    tst = pd.read_parquet(config.REPORTS / "test_scores.parquet")
    yv = train.label.values[val["row"].values]
    yt = test.label.values
    clf = load_json("03_classifiers.json")["models"]

    cands = [n for n in DEPLOYABLE if n in val and DEPLOYABLE[n][1] != "sparse_nb"]
    single = {n: roc_auc_score(yv, val[n]) for n in cands}
    chosen = [max(single, key=single.get)]
    best = stacked_val_auc([val[chosen[0]].values], yv)
    log = [{"components": list(chosen), "val_auc": round(best, 4)}]
    improved = True
    while improved and len(chosen) < 3:
        improved = False
        trials = {n: stacked_val_auc([val[c].values for c in chosen + [n]], yv) for n in cands if n not in chosen}
        if trials:
            n, auc = max(trials.items(), key=lambda kv: kv[1])
            if auc >= best + 0.002:
                chosen.append(n)
                best = auc
                improved = True
                log.append({"components": list(chosen), "val_auc": round(best, 4)})
    print("selection:", log)

    # ---- compression: dense components get a deployment-sized projection ----------------
    import scipy.sparse as sp
    from sklearn.base import clone
    from sklearn.model_selection import train_test_split
    from llm_router.features import CompactProjector

    lr = joblib.load(config.MODELS / "clf_logistic_regression.joblib")
    feat = joblib.load(config.MODELS / "featurizer.joblib")
    projector, compact_models, compression = feat["svd"], {}, {}
    dense = [n for n in chosen if DEPLOYABLE[n][1] == "dense"]
    if dense:
        X = sp.load_npz(config.DATA_PROCESSED / "X_train.npz").tocsr()
        Xt = sp.load_npz(config.DATA_PROCESSED / "X_test.npz").tocsr()
        y = train.label.values
        idx_tr, idx_va = train_test_split(np.arange(len(y)), test_size=0.2, random_state=config.SEED, stratify=y)
        assert np.array_equal(idx_va, val["row"].values), "validation split differs from step 3"
        cols = np.sort(np.argsort(-np.abs(lr.coef_.ravel()))[:COMPACT_COLS])
        projector = CompactProjector(cols, COMPACT_DIMS, config.SEED).fit(X)
        Zc, Zct = projector.transform(X), projector.transform(Xt)
        for n in dense:
            base_model = joblib.load(config.MODELS / f"clf_{slug(n)}.joblib")
            v_before = roc_auc_score(yv, val[n])
            val[n] = clone(base_model).fit(Zc[idx_tr], y[idx_tr]).predict_proba(Zc[idx_va])[:, 1]
            compact_models[n] = clone(base_model).fit(Zc, y)
            tst[n] = compact_models[n].predict_proba(Zct)[:, 1]
            compression[n] = {"val_auc_full_svd256": round(v_before, 4),
                              "val_auc_compact": round(roc_auc_score(yv, val[n]), 4)}
        compression["projection"] = {
            "full_svd_mb": round(feat["svd"].components_.astype(np.float32).nbytes / 1e6, 1),
            "compact_mb": round(projector.components_.nbytes / 1e6, 1),
            "columns_kept": int(len(cols)), "dims": COMPACT_DIMS}
        print("compression:", compression)

    Sv = np.column_stack([_logit(val[c].values) for c in chosen])
    St = np.column_stack([_logit(tst[c].values) for c in chosen])
    stacker = LogisticRegression().fit(Sv, yv)
    risk_v = stacker.predict_proba(Sv)[:, 1]
    risk_t = stacker.predict_proba(St)[:, 1]

    shares = np.linspace(0, 1, 101)
    thresholds = np.quantile(risk_v, 1 - shares)
    thresholds[0] = 1.0 + 1e-9          # 0% strong: never route
    thresholds[-1] = 0.0                # 100% strong: always route
    realised = [(risk_t >= t).mean() for t in thresholds]

    rep = metrics.classification_report(yt, risk_t, float(np.interp(0.3, shares, thresholds)))
    rout = metrics.routing_report(risk_t, test.score.values)
    x, quality, pgr = metrics.routing_curve(risk_t, test.score.values)
    base = metrics.oracle_and_random(test.score.values)
    best_single = max(clf, key=lambda n: clf[n]["routing"]["apgr"])
    print(f"final router: test AUC={rep['roc_auc']:.4f}  APGR={rout['apgr']:.4f}  "
          f"CPT50={rout['cpt50']:.2f}  CPT80={rout['cpt80']:.2f}  (best single {best_single}: "
          f"APGR={clf[best_single]['routing']['apgr']:.4f}; random {base['random_apgr']:.3f}; oracle {base['oracle']['apgr']:.3f})")

    components = [{"name": n, "input": DEPLOYABLE[n][1],
                   "model": compact_models.get(n) or joblib.load(config.MODELS / f"clf_{slug(n)}.joblib")}
                  for n in chosen]
    artifact = {
        "featurizer": feat["featurizer"], "svd": projector if dense else None, "components": components, "stacker": stacker,
        "share_thresholds": {"shares": shares.tolist(), "thresholds": thresholds.tolist()},
        "curve": {"x": x.tolist(), "quality": quality.tolist(), "pgr": pgr.tolist()},
        "explain_coef": lr.coef_.ravel().astype(np.float32),
        "feature_names": FeatureNames(feat["featurizer"].feature_names()),
        "meta": {"components": chosen, "test_auc": rep["roc_auc"], "test_apgr": rout["apgr"],
                 "cpt50": rout["cpt50"], "cpt80": rout["cpt80"], "trained_rows": int(len(train)),
                 "cheap_model_in_data": "Mixtral-8x7B-Instruct", "strong_model_in_data": "GPT-4",
                 "built": time.strftime("%Y-%m-%d")},
    }
    # Light variant for very small servers: logistic regression only (no embedding runtime).
    lv = _logit(val["Logistic Regression"].values)[:, None]
    lt = _logit(tst["Logistic Regression"].values)[:, None]
    light_stacker = LogisticRegression().fit(lv, yv)
    lrisk_v, lrisk_t = light_stacker.predict_proba(lv)[:, 1], light_stacker.predict_proba(lt)[:, 1]
    lthr = np.quantile(lrisk_v, 1 - shares)
    lthr[0], lthr[-1] = 1.0 + 1e-9, 0.0
    lx, lq, lp = metrics.routing_curve(lrisk_t, test.score.values)
    lrout = metrics.routing_report(lrisk_t, test.score.values)
    artifact["light"] = {
        "components": [{"name": "Logistic Regression", "input": "sparse",
                        "model": joblib.load(config.MODELS / "clf_logistic_regression.joblib")}],
        "stacker": light_stacker,
        "share_thresholds": {"shares": shares.tolist(), "thresholds": lthr.tolist()},
        "curve": {"x": lx.tolist(), "quality": lq.tolist(), "pgr": lp.tolist()},
        "meta": {**artifact["meta"], "components": ["Logistic Regression"], "mode": "light",
                 "test_auc": float(roc_auc_score(yt, lrisk_t)), "test_apgr": lrout["apgr"],
                 "cpt50": lrout["cpt50"], "cpt80": lrout["cpt80"]},
    }
    artifact["meta"]["mode"] = "full"
    print(f"light router: test AUC={artifact['light']['meta']['test_auc']:.4f} APGR={lrout['apgr']:.4f}")

    # Memory optimisation: plain-int vocabularies and an uncompressed file. Decompressing a
    # compressed joblib file briefly needs ~150 MB that the OS does not get back, which matters
    # on a 512 MB server; uncompressed, loading adds ~66 MB instead of ~198 MB.
    for vec in (artifact["featurizer"].word_, artifact["featurizer"].char_):
        vec.vocabulary_ = {t: int(i) for t, i in vec.vocabulary_.items()}
    path = config.MODELS / "router.joblib"
    joblib.dump(artifact, path, compress=0)
    size_mb = path.stat().st_size / 1e6

    # end-to-end check: the exported artifact must reproduce the offline test risks
    router = Router.load(path)
    sample = test.prompt.values[:2000].tolist()
    t = time.perf_counter()
    online = router.risk(sample)
    batch_ms = (time.perf_counter() - t) * 1000 / len(sample)
    diff = np.abs(online - risk_t[:2000])
    max_diff, mean_diff = float(diff.max()), float(diff.mean())
    thr30 = float(np.interp(0.3, shares, thresholds))
    agree30 = float(((online >= thr30) == (risk_t[:2000] >= thr30)).mean())
    lat = []
    for p in sample[:300]:
        t = time.perf_counter()
        router.risk([p])
        lat.append((time.perf_counter() - t) * 1000)
    perf = {"artifact_mb": round(size_mb, 2), "batch_ms_per_prompt": round(batch_ms, 3),
            "single_p50_ms": round(float(np.percentile(lat, 50)), 2),
            "single_p95_ms": round(float(np.percentile(lat, 95)), 2),
            "max_abs_diff_vs_offline": max_diff, "mean_abs_diff_vs_offline": mean_diff,
            "decision_agreement_at_30pct": agree30,
            "note": "online embeddings use ONNX on CPU; offline used PyTorch on the GPU" if any(
                DEPLOYABLE[n][1] == "embed" for n in chosen) else "sparse/dense path is bit-exact"}
    print("performance:", perf)
    uses_embed = any(DEPLOYABLE[n][1] == "embed" for n in chosen)
    assert (agree30 > 0.98 and mean_diff < 0.01) if uses_embed else max_diff < 1e-4, \
        "exported router does not reproduce offline scores"

    # ---- figures ------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(x * 100, pgr * 100, lw=2.5, label=f"Our router (APGR {rout['apgr']:.3f})")
    for n in ["Logistic Regression", "Random Forest", "Multinomial Naive Bayes"]:
        if n in tst:
            xx, _, pp = metrics.routing_curve(tst[n].values, test.score.values)
            ax.plot(xx * 100, pp * 100, lw=1, alpha=0.7, label=f"{n} ({clf[n]['routing']['apgr']:.3f})")
    xo, _, po = metrics.routing_curve(5 - test.score.values, test.score.values)
    ax.plot(xo * 100, po * 100, ls=":", c="k", label=f"Oracle ({base['oracle']['apgr']:.3f})")
    ax.plot([0, 100], [0, 100], ls="--", c="grey", label="Random (0.500)")
    ax.set(xlabel="% of prompts sent to the strong model", ylabel="% of quality gap recovered",
           title="Routing curve on the held-out test set", xlim=(0, 100), ylim=(0, 101))
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(config.FIGURES / "routing_curve.png", dpi=130)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for n in list(tst.columns):
        fpr, tpr, _ = roc_curve(yt, tst[n].values)
        ax.plot(fpr, tpr, lw=1, label=f"{n.split(' (')[0]} ({roc_auc_score(yt, tst[n]):.3f})")
    fpr, tpr, _ = roc_curve(yt, risk_t)
    ax.plot(fpr, tpr, lw=2.5, c="k", label=f"Final router ({rep['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", c="grey")
    ax.set(xlabel="False positive rate (1 - specificity)", ylabel="True positive rate (sensitivity)", title="ROC curves (test)")
    ax.legend(fontsize=6.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(config.FIGURES / "roc_curves.png", dpi=130)

    names = sorted(clf, key=lambda n: clf[n]["routing"]["apgr"])
    fig, ax = plt.subplots(figsize=(7.5, 0.38 * len(names) + 1.2))
    ax.barh([n for n in names] + ["FINAL ROUTER"], [clf[n]["routing"]["apgr"] for n in names] + [rout["apgr"]],
            color=["tab:blue"] * len(names) + ["tab:green"])
    ax.axvline(0.5, ls="--", c="grey")
    ax.set(xlim=(0.45, max(0.8, rout["apgr"] + 0.03)), xlabel="APGR on test (0.5 = random)", title="Model comparison")
    fig.tight_layout()
    fig.savefig(config.FIGURES / "model_comparison.png", dpi=130)

    save_json("07_router.json", {
        "selection_log": log, "compression": compression, "light_mode": artifact["light"]["meta"], "components": chosen, "test_classification_at_30pct": rep, "test_routing": rout,
        "baselines": base, "budget_check": {f"{int(s*100)}%": round(float(r), 4) for s, r in zip(shares[::10], realised[::10])},
        "performance": perf,
    })
