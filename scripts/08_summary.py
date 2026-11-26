"""Step 8 - collect every report into reports/summary.json for the web dashboard."""
import time

import numpy as np
import pandas as pd

from _common import load_json, save_json
from llm_router import config, data, metrics

if __name__ == "__main__":
    test = data.load("test")
    d1, d2 = load_json("01_data_report.json"), load_json("02_features.json")
    c3, r4 = load_json("03_classifiers.json"), load_json("04_regression.json")
    k5, a6, f7 = load_json("05_clustering.json"), load_json("06_association.json"), load_json("07_router.json")

    classifiers = [{
        "name": n, "params": ", ".join(f"{k}={v}" for k, v in m["best_params"].items()),
        "val_auc": m["val_auc"], "test_auc": m["test"]["roc_auc"], "apgr": m["routing"]["apgr"],
        "cpt80": m["routing"]["cpt80"], "precision": m["test"]["precision"],
        "recall": m["test"]["recall_sensitivity"], "specificity": m["test"]["specificity"],
        "f1": m["test"]["f1"], "accuracy": m["test"]["accuracy"], "fit_seconds": m["fit_seconds"],
        "predict_ms_per_1k": m["predict_ms_per_1k"]} for n, m in c3["models"].items()]

    regression = [{"name": n, **m["test"], "apgr": m["routing"]["apgr"],
                   "note": m.get("equation") or (f"{m['nonzero_coefficients']:,} of {m['total_features']:,} coefficients non-zero"
                                                  if "nonzero_coefficients" in m else m.get("note"))}
                  for n, m in r4.items() if not n.startswith("_")]

    # curves for the dashboard chart, 51 points each
    scores = pd.read_parquet(config.REPORTS / "test_scores.parquet")
    best_single = max(c3["models"], key=lambda n: c3["models"][n]["routing"]["apgr"])
    import joblib
    from llm_router.router import Router
    router = Router(joblib.load(config.MODELS / "router.joblib"))
    final_curve = router.curve
    xs = np.linspace(0, 1, 51)
    _, _, pgr_best = metrics.routing_curve(scores[best_single].values, test.score.values, points=51)
    _, _, pgr_oracle = metrics.routing_curve(5 - test.score.values, test.score.values, points=51)

    rr = f7["test_routing"]
    summary = {
        "built": time.strftime("%Y-%m-%d"),
        "data": {"train_rows": d1["train"]["rows_out"], "test_rows": d1["test"]["rows_out"],
                 "removed": sum(v for k, v in d1["train"].items() if k.startswith("dropped"))
                            + sum(v for k, v in d1["test"].items() if k.startswith("dropped")),
                 "fail_rate_train": d1["fail_rate_train"], "fail_rate_test": d1["fail_rate_test"]},
        "features": d2,
        "classifiers": classifiers,
        "activation_functions": {f"MLP on SVD (CPU), {k}": v for k, v in c3["activation_functions"].items()}
                                | {f"Embedding network (GPU), {k}": v
                                   for k, v in c3.get("activation_functions_gpu_network", {}).items()},
        "baselines": {"random_apgr": c3["baselines"]["random_apgr"], "oracle_apgr": c3["baselines"]["oracle"]["apgr"]},
        "router": {"components": f7["components"], "auc": f7["test_classification_at_30pct"]["roc_auc"],
                   "classification_at_30pct": f7["test_classification_at_30pct"], **rr,
                   "performance": f7["performance"], "budget_check": f7["budget_check"]},
        "curves": {"x": xs.tolist(),
                   "final": np.interp(xs, final_curve["x"], final_curve["pgr"]).tolist(),
                   "best_single": pgr_best.tolist(), "best_single_name": best_single,
                   "best_single_apgr": c3["models"][best_single]["routing"]["apgr"],
                   "oracle": pgr_oracle.tolist()},
        "regression": regression, "regression_baseline": r4["_baseline_predict_mean"],
        "clustering": {"k": k5["k"], "clusters": k5["clusters"], "hierarchical": k5["hierarchical"],
                       "dbscan": {k: v for k, v in k5["dbscan"].items() if k != "largest_groups"}},
        "association": {k: a6[k] for k in ("apriori", "naive", "base_fail_rate", "min_support")}
                       | {"rules_to_fail": a6["rules_to_fail"][:12]},
    }
    save_json("summary.json", summary)
