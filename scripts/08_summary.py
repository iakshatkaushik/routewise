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

    # fine-tuned transformer (step 3c): validation-only sweep, final run, ONNX export check (step 7a)
    def opt(name):
        path = config.REPORTS / name
        return load_json(name) if path.exists() else None
    sweep, ft, export = opt("finetune_sweep.json"), opt("03c_finetune.json"), opt("07a_text_export.json")
    finetune = None
    if ft:
        finetune = {
            "base_model": ft["base"], "max_tokens": ft["max_tokens"], "hyperparameters": ft["hyperparameters"],
            "seed": ft["seed"], "teacher": ft.get("teacher"), "distill_alpha": ft.get("distill_alpha"),
            "pooling": ft.get("pooling", "cls"),
            "precision": ft["precision"], "gpu": ft.get("gpu"), "train_minutes": ft["train_minutes"],
            "history": ft["history"], "results": ft["results"],
            # every validation-only run except the teacher's fold runs (summarised separately)
            "sweep": [{"tag": r["tag"], "base": r["base"], "lr": r["lr"], "epochs": r["epochs"],
                       "batch_size": r["batch_size"], "max_tokens": r["max_tokens"],
                       "deficit_weight": r["deficit_weight"], "seed": r["seed"], "precision": r["precision"],
                       "pooling": r.get("pooling", "cls"), "params_millions": r.get("params_millions"),
                       "teacher": r.get("teacher"), "distill_alpha": r.get("distill_alpha"),
                       "kind": ("study" if r["tag"].startswith("study:") else
                                "distill" if r.get("teacher") else "sweep"),
                       "train_minutes": r["train_minutes"], **r["best_checkpoint"]}
                      for r in (sweep or {}).get("runs", []) if not r.get("n_folds")],
            "teacher_folds": [{"tag": r["tag"], "train_minutes": r["train_minutes"], **r["best_checkpoint"]}
                              for r in (sweep or {}).get("runs", []) if r.get("n_folds")],
            # fold runs of the teacher the deployed student learned from (its out-of-fold file is ft["teacher"])
            "teacher_name": {"models/teacher/teacher_oof.npz": "Qwen3-1.7B",
                             "models/teacher/teacher_oof_qwen3_4b.npz": "Qwen3-4B",
                             "models/teacher/teacher_oof_qwen3_8b_lora.npz": "Qwen3-8B"}.get(ft.get("teacher")),
            "export": export and {k: export[k] for k in ("served_variant", "served_mb", "variants_validation",
                                                         "int8_val_auc_loss_vs_fp32", "latency_1_thread_ms", "rule")},
        }
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
                   "performance": f7["performance"] | ({"text_model_mb": export["served_mb"]}
                                                      if export and f7.get("text_model") else {}),
                   "budget_check": f7["budget_check"],
                   "validation_cv": f7.get("validation_cv"), "selection_log": f7.get("selection_log"),
                   "selection_rule": f7.get("selection_rule"),
                   "memory": f7.get("memory"),
                   "light": {k: f7["light_mode"].get(k) for k in ("components", "test_auc", "test_apgr", "cpt50", "cpt80",
                                                                  "val_cv_apgr", "val_cv_auc")}
                            | {"performance": f7.get("light_performance")}},
        "finetune": finetune,
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
