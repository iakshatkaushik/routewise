"""Step 3 - tune and compare every syllabus classifier (Classification + ANN units).

Protocol
- Hyper-parameters are chosen on a validation split (20% of the training data).
- The best setting of each model is refitted on all training data and scored once on the
  untouched test set.
- Sparse TF-IDF features go to the linear models and Naive Bayes; the 256-dim SVD
  representation goes to the tree, neighbour and neural models.
"""
import itertools

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import normalize
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

from _common import Timer, save_json, slug
from llm_router import config, data, metrics


def score_of(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def grid(**kw):
    keys = list(kw)
    for vals in itertools.product(*kw.values()):
        yield dict(zip(keys, vals))


if __name__ == "__main__":
    train, test = data.load("train"), data.load("test")
    X = sp.load_npz(config.DATA_PROCESSED / "X_train.npz").tocsr()
    Xt = sp.load_npz(config.DATA_PROCESSED / "X_test.npz").tocsr()
    Z = np.load(config.DATA_PROCESSED / "Z_train.npy")
    Zt = np.load(config.DATA_PROCESSED / "Z_test.npy")
    Zn, Ztn = normalize(Z), normalize(Zt)
    y, yt = train.label.values, test.label.values
    n_stats = 24
    Xnb, Xtnb = X[:, :-n_stats], Xt[:, :-n_stats]  # Naive Bayes needs non-negative counts/TF-IDF

    idx_tr, idx_va = train_test_split(np.arange(len(y)), test_size=0.2, random_state=config.SEED, stratify=y)
    rng = np.random.default_rng(config.SEED)
    sub20k = rng.choice(idx_tr, 20000, replace=False)

    def balanced_weights(lbl):
        w = np.where(lbl == 1, 0.5 / lbl.mean(), 0.5 / (1 - lbl.mean()))
        return w

    # name: (factory, param grid, train matrix, test matrix, train subset or None, uses sample_weight)
    specs = {
        "Logistic Regression": (lambda p: LogisticRegression(max_iter=4000, class_weight="balanced", **p),
                                grid(C=[0.1, 0.25, 0.5, 1.0]), X, Xt, None),
        "Linear SVM": (lambda p: LinearSVC(class_weight="balanced", **p),
                       grid(C=[0.01, 0.03, 0.1]), X, Xt, None),
        "RBF-kernel SVM (20k sample)": (lambda p: SVC(kernel="rbf", class_weight="balanced", cache_size=2000, **p),
                                        grid(C=[0.5, 2.0], gamma=["scale"]), Zn, Ztn, sub20k),
        "Multinomial Naive Bayes": (lambda p: MultinomialNB(**p), grid(alpha=[0.03, 0.1, 0.3, 1.0]), Xnb, Xtnb, None),
        "Complement Naive Bayes": (lambda p: ComplementNB(**p), grid(alpha=[0.1, 0.3, 1.0]), Xnb, Xtnb, None),
        "Decision Tree (Gini)": (lambda p: DecisionTreeClassifier(criterion="gini", class_weight="balanced", random_state=config.SEED, **p),
                                 grid(max_depth=[8, 12, 16], min_samples_leaf=[50, 200]), Z, Zt, None),
        "Decision Tree (Entropy / information gain)": (lambda p: DecisionTreeClassifier(criterion="entropy", class_weight="balanced", random_state=config.SEED, **p),
                                                       grid(max_depth=[8, 12, 16], min_samples_leaf=[50, 200]), Z, Zt, None),
        "Random Forest": (lambda p: RandomForestClassifier(n_estimators=200, n_jobs=-1, class_weight="balanced_subsample", random_state=config.SEED, **p),
                          grid(max_depth=[None], min_samples_leaf=[2, 8], max_features=["sqrt"]), Z, Zt, None),
        "K-Nearest Neighbours": (lambda p: KNeighborsClassifier(n_jobs=-1, algorithm="brute", **p),
                                 grid(n_neighbors=[50, 150, 300], weights=["distance"]), Zn, Ztn, None),
        "Neural Network (MLP)": (lambda p: MLPClassifier(early_stopping=True, max_iter=200, random_state=config.SEED, batch_size=512, **p),
                                 grid(hidden_layer_sizes=[(256,), (256, 128)], alpha=[1e-3, 1e-2], activation=["relu"]), Z, Zt, None),
        "Gradient Boosting (beyond syllabus)": (lambda p: HistGradientBoostingClassifier(class_weight="balanced", early_stopping=True, random_state=config.SEED, **p),
                                                grid(learning_rate=[0.05, 0.1], max_leaf_nodes=[31, 63]), Z, Zt, None),
    }

    results, test_scores, val_scores = {}, {}, {}
    for name, (make, params, A, At, subset) in specs.items():
        tr_rows = subset if subset is not None else idx_tr
        best = None
        for p in params:
            with Timer() as t:
                m = make(p).fit(A[tr_rows], y[tr_rows])
            s_va = score_of(m, A[idx_va])
            auc = roc_auc_score(y[idx_va], s_va)
            print(f"  {name:45s} {p} val AUC={auc:.4f} ({t.s:.0f}s)", flush=True)
            if best is None or auc > best[0]:
                best = (auc, p, s_va)
        auc_va, p, s_va = best
        val_scores[name] = s_va
        fit_rows = subset if subset is not None else np.arange(len(y))
        with Timer() as t_fit:
            m = make(p).fit(A[fit_rows], y[fit_rows])
        with Timer() as t_pred:
            s_te = score_of(m, At)
        test_scores[name] = s_te
        # decision threshold chosen on validation: the value that sends 30% of prompts to the strong model
        thr = float(np.quantile(s_va, 0.70))
        rep = metrics.classification_report(yt, s_te, thr)
        rout = metrics.routing_report(s_te, test.score.values)
        results[name] = {"best_params": {k: str(v) for k, v in p.items()}, "val_auc": round(auc_va, 4),
                         "test": rep, "routing": rout, "fit_seconds": round(t_fit.s, 1),
                         "predict_ms_per_1k": round(1000 * t_pred.s / len(yt) * 1000, 2),
                         "trained_on_rows": int(len(fit_rows))}
        print(f"==> {name}: test AUC={rep['roc_auc']:.4f} APGR={rout['apgr']:.4f}", flush=True)
        joblib.dump(m, config.MODELS / f"clf_{slug(name)}.joblib", compress=3)

    # Activation-function study for the neural network (syllabus: Activation Functions)
    act = {}
    for a in ["logistic", "tanh", "relu"]:
        with Timer() as t:
            m = MLPClassifier(hidden_layer_sizes=(256,), activation=a, alpha=1e-2, early_stopping=True,
                              max_iter=200, random_state=config.SEED, batch_size=512).fit(Z[idx_tr], y[idx_tr])
        act[a] = {"val_auc": round(roc_auc_score(y[idx_va], m.predict_proba(Z[idx_va])[:, 1]), 4),
                  "epochs": int(m.n_iter_), "seconds": round(t.s, 1)}
        print("activation", a, act[a], flush=True)

    base = metrics.oracle_and_random(test.score.values)
    save_json("03_classifiers.json", {"models": results, "activation_functions": act, "baselines": base,
                                      "protocol": "tuned on 20% validation split of train; scored once on test"})
    pd.DataFrame(test_scores).to_parquet(config.REPORTS / "test_scores.parquet")
    pd.DataFrame(val_scores).assign(row=idx_va).to_parquet(config.REPORTS / "val_scores.parquet")
