"""Step 3g - would serving two distilled students together help? (validation only)

Uses the validation logits saved by the student runs (models/study_logits/, steps 3e and 3f). For each
student, the deficit head (the head the router selects) is stacked exactly as in step 7: a logistic-regression
stacker, 5-fold cross-validated on the validation split, scored by APGR (AUC reported too). Then every pair
of the top students is stacked the same way. A pair would need two ONNX models online (about twice the
text-model latency), so it is only worth it for a clear gain. Writes reports/student_ensemble_check.json.
"""
import glob
import itertools

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

from _common import save_json
from llm_router import config, data, metrics

TOP = 6

if __name__ == "__main__":
    train = data.load("train")
    val = pd.read_parquet(config.REPORTS / "val_scores.parquet")
    y, s = train.label.values[val.row.values], train.score.values[val.row.values]
    logits = {}
    for f in sorted(glob.glob(str(config.MODELS / "study_logits" / "*.npz"))):
        name = f.split("/")[-1][:-4]
        if not name.startswith(("search_", "distill_")):
            continue
        d = np.load(f)
        assert np.array_equal(d["val_rows"], val.row.values), f"{name}: validation rows differ"
        logits[name] = d["val_logits"]

    def stacked(cols):
        S = np.column_stack(cols)
        p = cross_val_predict(LogisticRegression(), S, y, cv=5, method="predict_proba")[:, 1]
        return {"cv_val_apgr": round(metrics.routing_report(p, s)["apgr"], 4), "cv_val_auc": round(roc_auc_score(y, p), 4)}

    single = {k: stacked([v[:, 1]]) for k, v in logits.items()}
    top = sorted(single, key=lambda k: (single[k]["cv_val_apgr"], single[k]["cv_val_auc"]), reverse=True)[:TOP]
    pairs = sorted(({"students": [a, b], **stacked([logits[a][:, 1], logits[b][:, 1]])}
                    for a, b in itertools.combinations(top, 2)),
                   key=lambda r: (r["cv_val_apgr"], r["cv_val_auc"]), reverse=True)
    best_single = single[top[0]]
    out = {"students_scored": len(single), "top_single": [{"student": k, **single[k]} for k in top],
           "pairs": pairs, "best_pair_gain_apgr": round(pairs[0]["cv_val_apgr"] - best_single["cv_val_apgr"], 4),
           "both_heads_of_best_single": stacked([logits[top[0]][:, 0], logits[top[0]][:, 1]])}
    print(f"best single {top[0]}: {best_single}")
    print(f"best pair {pairs[0]}")
    print(f"gain of the best pair: {out['best_pair_gain_apgr']:+.4f} APGR")
    save_json("student_ensemble_check.json", out)
