"""Step 4 - regression: predict the cheap model's 1-5 quality score (Regression unit).

Models: simple linear (one feature), multiple linear, polynomial (degree 2), and the
regularised Ridge (L2), Lasso (L1) and Elastic Net on the full feature set.
Each regressor is also used as a router: prompts with the lowest predicted score go to the
strong model, so we can compare it with the classifiers on the same routing metric.
"""
import warnings

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from _common import Timer, save_json
from llm_router import config, data, metrics
from llm_router.features import STAT_NAMES, prompt_stats

warnings.filterwarnings("ignore", message="Objective did not converge")

if __name__ == "__main__":
    train, test = data.load("train"), data.load("test")
    y, yt = train.score.values.astype(float), test.score.values.astype(float)
    S, St = prompt_stats(train.prompt), prompt_stats(test.prompt)
    X = sp.load_npz(config.DATA_PROCESSED / "X_train.npz").tocsr()
    Xt = sp.load_npz(config.DATA_PROCESSED / "X_test.npz").tocsr()
    idx_tr, idx_va = train_test_split(np.arange(len(y)), test_size=0.2, random_state=config.SEED)
    out = {}

    def record(name, model, A, At, extra=None):
        with Timer() as t:
            model.fit(A, y)
        pred = np.clip(model.predict(At), 1, 5)
        out[name] = {"test": metrics.regression_report(yt, pred),
                     "routing": metrics.routing_report(-pred, yt),
                     "fit_seconds": round(t.s, 1), **(extra or {})}
        print(f"{name:40s} RMSE={out[name]['test']['rmse']:.3f} R2={out[name]['test']['r2']:.3f} "
              f"r={out[name]['test']['pearson_r']:.3f} APGR={out[name]['routing']['apgr']:.3f}", flush=True)
        return model

    # 1. simple linear regression: score ~ log(prompt length)
    j = STAT_NAMES.index("log_chars")
    m = record("Simple Linear (log prompt length)", LinearRegression(), S[:, [j]], St[:, [j]])
    out["Simple Linear (log prompt length)"]["equation"] = f"score = {m.intercept_:.3f} + {m.coef_[0]:.3f} * log(1 + chars)"

    # 2. multiple linear regression on the 24 prompt statistics
    m = record("Multiple Linear (24 prompt stats)", make_pipeline(StandardScaler(), LinearRegression()), S, St)
    coefs = dict(zip(STAT_NAMES, m[-1].coef_.round(4).tolist()))
    out["Multiple Linear (24 prompt stats)"]["top_coefficients"] = dict(sorted(coefs.items(), key=lambda kv: -abs(kv[1]))[:8])

    # 3. polynomial regression (degree 2: squares and pairwise interactions)
    record("Polynomial deg-2 (24 prompt stats)",
           make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False), StandardScaler(), Ridge(alpha=1.0)), S, St,
           {"note": "324 polynomial terms; a tiny ridge penalty (alpha=1) keeps the solution stable"})

    # 4. regularisation: alpha chosen on the validation split
    def tune(make, alphas):
        best = None
        for a in alphas:
            mm = make(a).fit(X[idx_tr], y[idx_tr])
            rmse = metrics.regression_report(y[idx_va], np.clip(mm.predict(X[idx_va]), 1, 5))["rmse"]
            print(f"   alpha={a:g} val RMSE={rmse:.4f}", flush=True)
            if best is None or rmse < best[0]:
                best = (rmse, a)
        return best[1]

    a = tune(lambda a: Ridge(alpha=a, solver="sparse_cg"), [3, 10, 30, 100])
    record("Ridge (L2), full features", Ridge(alpha=a, solver="sparse_cg"), X, Xt, {"alpha": a})
    a = tune(lambda a: Lasso(alpha=a, max_iter=1000, tol=1e-3), [1e-4, 3e-4, 1e-3])
    m = record("Lasso (L1), full features", Lasso(alpha=a, max_iter=1000, tol=1e-3), X, Xt, {"alpha": a})
    out["Lasso (L1), full features"]["nonzero_coefficients"] = int((m.coef_ != 0).sum())
    out["Lasso (L1), full features"]["total_features"] = int(X.shape[1])
    record("Elastic Net (L1+L2), full features", ElasticNet(alpha=a * 2, l1_ratio=0.5, max_iter=1000, tol=1e-3), X, Xt,
           {"alpha": a * 2, "l1_ratio": 0.5})

    out["_baseline_predict_mean"] = metrics.regression_report(yt, np.full_like(yt, y.mean()))
    save_json("04_regression.json", out)
