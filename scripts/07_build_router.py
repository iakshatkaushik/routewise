"""Step 7 - build, optimise and export the production router.

1. Candidate components are the tuned models from steps 3, 3b, 3c and 4 that can run cheaply online
   (K-NN and Random Forest are excluded: K-NN must keep all 109k training vectors in memory
   and the forest is hundreds of MB). Each has an input kind (sparse TF-IDF, dense SVD, sentence
   embedding, raw text) and an output kind (probability, predicted 1-5 score, or a fine-tuned
   transformer's logit).
2. Greedy forward selection on validation: the criterion is the 5-fold cross-validated APGR of the
   stacked model (AUC breaks ties); a component is added only if APGR improves by at least 0.001,
   up to 4 components. Every step is logged.
3. A logistic-regression stacker combines the components' signals (log-odds, or minus a predicted
   score) into one calibrated probability "P(cheap model fails)". It is trained on validation
   predictions only, built with the same transform the Router uses online (Router._component_signal).
4. Budget table: for every strong-model share s in 0..100%, the risk threshold is the
   (1 - s) quantile of validation risks. The test set checks that the realised share matches.
5. A light variant (sparse components only, no embedding or ONNX model) is chosen by the same rule.
6. Everything is scored once on the untouched test set and exported to models/router.joblib.
"""
import json
import subprocess
import sys
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
from llm_router.router import Router  # noqa: E402

COMPACT_COLS, COMPACT_DIMS = 40000, 128

TEXT = "Fine-tuned Transformer (GPU)"
TEXT_DEFICIT = "Fine-tuned Transformer (GPU), score-deficit head"
RIDGE = "Ridge regression (predicted score)"
# score-table column -> how the component is served. file None = models/clf_<slug>.joblib
DEPLOYABLE = {
    "Logistic Regression": {"input": "sparse", "output": "proba"},
    RIDGE: {"input": "sparse", "output": "score", "file": "reg_ridge.joblib"},
    "Embedding Neural Network (GPU)": {"input": "embed", "output": "proba"},
    TEXT: {"input": "text", "output": "text_logit", "head": 0},
    TEXT_DEFICIT: {"input": "text", "output": "text_logit", "head": 1},
    "Neural Network (MLP)": {"input": "dense", "output": "proba"},
    "Gradient Boosting (beyond syllabus)": {"input": "dense", "output": "proba"},
    "Multinomial Naive Bayes": {"input": "sparse_nb", "output": "proba"},
    "Decision Tree (Entropy / information gain)": {"input": "dense", "output": "proba"},
}
LIGHT_INPUTS = {"sparse"}           # light mode: no embedding runtime, no ONNX text model
MIN_GAIN, MAX_COMPONENTS = 0.001, 4


class _Stored:
    """Stand-in model that returns stored predictions, so offline stacker inputs go through the exact
    Router._component_signal transform used online."""
    def __init__(self, v):
        self.v = np.asarray(v, dtype=np.float64)

    def predict_proba(self, _):
        return np.column_stack([1 - self.v, self.v])

    def predict(self, _):
        return self.v


def offline_signal(name, values):
    spec = DEPLOYABLE[name]
    c = {"input": spec["input"], "output": spec["output"], "head": spec.get("head"), "model": _Stored(values)}
    feats = {spec["input"]: None}
    if spec["output"] == "text_logit":              # score tables store sigmoid(logit): invert exactly
        v = np.asarray(values, dtype=np.float64)
        L = np.zeros((len(v), 2))
        L[:, spec["head"]] = np.log(v) - np.log1p(-v)
        feats["text"] = L
    return Router._component_signal(c, feats)


def stacked_cv(names, table, y, s):
    """5-fold cross-validated stacked predictions on validation -> (APGR, AUC)."""
    S = np.column_stack([offline_signal(n, table[n].values) for n in names])
    p = cross_val_predict(LogisticRegression(), S, y, cv=5, method="predict_proba")[:, 1]
    return metrics.routing_report(p, s)["apgr"], roc_auc_score(y, p)


def forward_select(cands, table, y, s, label):
    chosen, best, log = [], (-1.0, -1.0), []
    while len(chosen) < MAX_COMPONENTS:
        trials = {n: stacked_cv(chosen + [n], table, y, s) for n in cands if n not in chosen}
        if not trials:
            break
        n, (apgr, auc) = max(trials.items(), key=lambda kv: kv[1])     # APGR first, AUC breaks ties
        step = {"step": len(chosen) + 1,
                "trials": {k: {"cv_val_apgr": round(a, 4), "cv_val_auc": round(u, 4)} for k, (a, u) in
                           sorted(trials.items(), key=lambda kv: kv[1], reverse=True)},
                "best_candidate": n, "gain": None if not chosen else round(apgr - best[0], 4)}
        if chosen and apgr < best[0] + MIN_GAIN:
            step["decision"] = f"stop: gain below {MIN_GAIN}"
            log.append(step)
            print(f"[{label}] step {step['step']}: best add {n} -> APGR {apgr:.4f} (gain {apgr - best[0]:+.4f}) - stop", flush=True)
            break
        chosen.append(n)
        best = (apgr, auc)
        step.update(decision="add", components=list(chosen), cv_val_apgr=round(apgr, 4), cv_val_auc=round(auc, 4))
        log.append(step)
        print(f"[{label}] step {step['step']}: add {n} -> CV val APGR {apgr:.4f}, AUC {auc:.4f}", flush=True)
    return chosen, best, log


def load_component_model(name):
    spec = DEPLOYABLE[name]
    if spec["input"] == "text":
        return None
    return joblib.load(config.MODELS / (spec.get("file") or f"clf_{slug(name)}.joblib"))


def make_component(name, model=None):
    spec = DEPLOYABLE[name]
    c = {"name": name, "input": spec["input"], "output": spec["output"],
         "model": model if model is not None else load_component_model(name)}
    if "head" in spec:
        c["head"] = spec["head"]
    return c


RSS_SNIPPET = """
import json, sys
sys.path.insert(0, {src!r})
from llm_router.router import Router
r = Router.load(mode={mode!r})
for p in json.load(open({prompts!r})):
    r.route(p, 0.3)
status = dict(l.split(":", 1) for l in open("/proc/self/status"))
print(int(status["VmRSS"].split()[0]) / 1024, int(status["VmHWM"].split()[0]) / 1024)
"""


def measure_rss(mode, prompts):
    """Resident memory (Linux /proc: current VmRSS and peak VmHWM) of a fresh process after loading
    the router and routing 20 prompts. Only the router's own imports are loaded."""
    import os
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(list(prompts), f)
    code = RSS_SNIPPET.format(src=str(config.ROOT / "src"), mode=mode, prompts=f.name)
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True)
    os.unlink(f.name)
    rss, peak = map(float, out.stdout.split()[-2:])
    return {"rss_mb": round(rss, 1), "peak_rss_mb": round(peak, 1), "prompts_routed": len(prompts)}


if __name__ == "__main__":
    train, test = data.load("train"), data.load("test")
    val = pd.read_parquet(config.REPORTS / "val_scores.parquet")
    tst = pd.read_parquet(config.REPORTS / "test_scores.parquet")
    yv = train.label.values[val["row"].values]
    yt = test.label.values
    clf = load_json("03_classifiers.json")["models"]

    sv = train.score.values[val["row"].values]
    cands = [n for n in DEPLOYABLE if n in val and DEPLOYABLE[n]["input"] != "sparse_nb"]
    print("candidates:", cands, flush=True)
    chosen, (best_apgr, best_auc), log = forward_select(cands, val, yv, sv, "full")
    light_cands = [n for n in cands if DEPLOYABLE[n]["input"] in LIGHT_INPUTS]
    light_chosen, (light_apgr, light_auc), light_log = forward_select(light_cands, val, yv, sv, "light")

    # ---- compression: dense components get a deployment-sized projection ----------------
    import scipy.sparse as sp
    from sklearn.base import clone
    from sklearn.model_selection import train_test_split
    from llm_router.features import CompactProjector

    lr = joblib.load(config.MODELS / "clf_logistic_regression.joblib")
    feat = joblib.load(config.MODELS / "featurizer.joblib")
    projector, compact_models, compression = feat["svd"], {}, {}
    dense = [n for n in chosen if DEPLOYABLE[n]["input"] == "dense"]
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

    Sv = np.column_stack([offline_signal(c, val[c].values) for c in chosen])
    St = np.column_stack([offline_signal(c, tst[c].values) for c in chosen])
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

    components = [make_component(n, compact_models.get(n)) for n in chosen]
    artifact = {
        "featurizer": feat["featurizer"], "svd": projector if dense else None, "components": components, "stacker": stacker,
        "share_thresholds": {"shares": shares.tolist(), "thresholds": thresholds.tolist()},
        "curve": {"x": x.tolist(), "quality": quality.tolist(), "pgr": pgr.tolist()},
        "explain_coef": lr.coef_.ravel().astype(np.float32),
        "feature_names": FeatureNames(feat["featurizer"].feature_names()),
        "meta": {"components": chosen, "val_cv_apgr": round(best_apgr, 4), "val_cv_auc": round(best_auc, 4),
                 "test_auc": rep["roc_auc"], "test_apgr": rout["apgr"],
                 "cpt50": rout["cpt50"], "cpt80": rout["cpt80"], "trained_rows": int(len(train)),
                 "cheap_model_in_data": "Mixtral-8x7B-Instruct", "strong_model_in_data": "GPT-4",
                 "built": time.strftime("%Y-%m-%d")},
    }
    uses_text = any(DEPLOYABLE[n]["input"] == "text" for n in chosen)
    if uses_text:
        text_spec = load_json("07a_text_export.json")
        artifact["text_model"] = {"onnx": text_spec["served_file"], "tokenizer": text_spec["tokenizer_file"],
                                  "max_tokens": int(text_spec["max_tokens"])}
    # Light variant for very small servers: sparse components only (no embedding runtime, no ONNX model).
    lv = np.column_stack([offline_signal(c, val[c].values) for c in light_chosen])
    lt = np.column_stack([offline_signal(c, tst[c].values) for c in light_chosen])
    light_stacker = LogisticRegression().fit(lv, yv)
    lrisk_v, lrisk_t = light_stacker.predict_proba(lv)[:, 1], light_stacker.predict_proba(lt)[:, 1]
    lthr = np.quantile(lrisk_v, 1 - shares)
    lthr[0], lthr[-1] = 1.0 + 1e-9, 0.0
    lx, lq, lp = metrics.routing_curve(lrisk_t, test.score.values)
    lrout = metrics.routing_report(lrisk_t, test.score.values)
    artifact["light"] = {
        "components": [make_component(n) for n in light_chosen],
        "stacker": light_stacker,
        "share_thresholds": {"shares": shares.tolist(), "thresholds": lthr.tolist()},
        "curve": {"x": lx.tolist(), "quality": lq.tolist(), "pgr": lp.tolist()},
        "meta": {**artifact["meta"], "components": light_chosen, "mode": "light",
                 "val_cv_apgr": round(light_apgr, 4), "val_cv_auc": round(light_auc, 4),
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

    # end-to-end check: the exported artifact must reproduce the offline test risks. Prompts are scored
    # one at a time, as the web service does (the int8 text model's outputs depend on batch padding).
    def online_check(mode, offline_risk, thr_table, n_check=2000, n_lat=300):
        router = Router.load(path, mode=mode)
        sample = test.prompt.values[:n_check].tolist()
        router.risk(sample[:5])                                    # warm-up (loads ONNX sessions)
        online, lat = np.zeros(len(sample)), []
        for i, p in enumerate(sample):
            t = time.perf_counter()
            online[i] = router.risk([p])[0]
            lat.append((time.perf_counter() - t) * 1000)
        t = time.perf_counter()
        router.risk(sample[:500])
        batch_ms = (time.perf_counter() - t) * 1000 / 500
        diff = np.abs(online - offline_risk[:n_check])
        thr30 = float(np.interp(0.3, shares, thr_table))
        return {"artifact_mb": round(size_mb, 2), "batch_ms_per_prompt": round(batch_ms, 3),
                "single_p50_ms": round(float(np.percentile(lat[:n_lat], 50)), 2),
                "single_p95_ms": round(float(np.percentile(lat[:n_lat], 95)), 2),
                "single_p50_ms_2000": round(float(np.percentile(lat, 50)), 2),
                "single_p95_ms_2000": round(float(np.percentile(lat, 95)), 2),
                "max_abs_diff_vs_offline": float(diff.max()), "mean_abs_diff_vs_offline": float(diff.mean()),
                "decision_agreement_at_30pct": float(((online >= thr30) == (offline_risk[:n_check] >= thr30)).mean()),
                "threads": 1, "prompts_checked": n_check}

    import os
    os.environ.setdefault("EMBED_THREADS", "1")
    os.environ.setdefault("TEXT_MODEL_THREADS", "1")
    uses_embed = any(DEPLOYABLE[n]["input"] == "embed" for n in chosen)
    perf = online_check("full", risk_t, thresholds)
    perf["note"] = ("online embeddings use ONNX on CPU; offline used PyTorch on the GPU" if uses_embed else
                    "sparse/dense path is bit-exact") + (
                    "; the text model is the same ONNX file online and offline" if uses_text else "")
    print("performance (full):", perf, flush=True)
    assert (perf["decision_agreement_at_30pct"] > 0.98 and perf["mean_abs_diff_vs_offline"] < 0.01) if uses_embed \
        else perf["max_abs_diff_vs_offline"] < 1e-4, "exported router does not reproduce offline scores"
    if uses_text and not uses_embed:
        assert perf["max_abs_diff_vs_offline"] < 1e-4, "online text path differs from the offline one"
    light_perf = online_check("light", lrisk_t, lthr)
    print("performance (light):", light_perf, flush=True)
    assert light_perf["max_abs_diff_vs_offline"] < 1e-4, "light router does not reproduce offline scores"
    first20 = test.prompt.values[:20].tolist()
    memory = {"full": measure_rss("full", first20), "light": measure_rss("light", first20), "platform": sys.platform,
              "note": "fresh Python process: load the router, route 20 test prompts, read /proc/self/status"}
    print("memory:", memory, flush=True)

    # ---- figures ------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(x * 100, pgr * 100, lw=2.5, label=f"Our router (APGR {rout['apgr']:.3f})")
    for n in [TEXT, "Logistic Regression", "Random Forest", "Multinomial Naive Bayes"]:
        if n in tst:
            xx, _, pp = metrics.routing_curve(tst[n].values, test.score.values)
            a = metrics.routing_report(tst[n].values, test.score.values)["apgr"]
            ax.plot(xx * 100, pp * 100, lw=1, alpha=0.7, label=f"{n} ({a:.3f})")
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
        sig = -tst[n].values if DEPLOYABLE.get(n, {}).get("output") == "score" else tst[n].values  # risk = -score
        fpr, tpr, _ = roc_curve(yt, sig)
        short = n.split(" (")[0] + (", deficit head" if n == TEXT_DEFICIT else "")
        ax.plot(fpr, tpr, lw=1, label=f"{short} ({roc_auc_score(yt, sig):.3f})")
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
        "selection_rule": f"greedy forward selection on 5-fold cross-validated stacked validation APGR "
                          f"(AUC breaks ties); add only if APGR improves by >= {MIN_GAIN}; at most {MAX_COMPONENTS}",
        "selection_log": log, "light_selection_log": light_log, "candidates": cands,
        "validation_cv": {"full": {"apgr": round(best_apgr, 4), "auc": round(best_auc, 4)},
                          "light": {"apgr": round(light_apgr, 4), "auc": round(light_auc, 4)}},
        "text_model": artifact.get("text_model"),
        "compression": compression, "light_mode": artifact["light"]["meta"], "components": chosen,
        "test_classification_at_30pct": rep, "test_routing": rout,
        "baselines": base, "budget_check": {f"{int(s*100)}%": round(float(r), 4) for s, r in zip(shares[::10], realised[::10])},
        "performance": perf, "light_performance": light_perf, "memory": memory,
    })
