"""Step 5 - discover prompt types with clustering (Clustering unit).

- K-Means (partitioned): k chosen by silhouette score, with the elbow curve saved too.
- Agglomerative / Ward (hierarchical): agreement with K-Means measured by adjusted Rand index.
- DBSCAN (density-based): finds tight groups of near-template prompts and labels the rest noise.
Each K-Means cluster gets a name (its top words) and its cheap-model failure rate, which shows
which kinds of prompt the cheap model struggles with.
"""
import joblib
import matplotlib
import numpy as np
import scipy.sparse as sp

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from _common import save_json  # noqa: E402
from llm_router import clustering as C  # noqa: E402
from llm_router import config, data  # noqa: E402

if __name__ == "__main__":
    train, test = data.load("train"), data.load("test")
    Z = C.unit(np.load(config.DATA_PROCESSED / "Z_train.npy")[:, :64])  # first 64 SVD dims
    Zt = C.unit(np.load(config.DATA_PROCESSED / "Z_test.npy")[:, :64])
    X = sp.load_npz(config.DATA_PROCESSED / "X_train.npz").tocsr()
    feat = joblib.load(config.MODELS / "featurizer.joblib")["featurizer"]

    sweep = C.kmeans_sweep(Z, list(range(6, 31, 2)))
    best_k = max(sweep, key=lambda r: r["silhouette"])["k"]
    print("K-Means sweep:", [(r["k"], round(r["silhouette"], 3)) for r in sweep], "-> k =", best_k)
    km = C.fit_kmeans(Z, best_k)
    labels, labels_t = km.labels_, km.predict(Zt)
    names = C.top_terms(feat, X, labels, best_k)

    clusters = []
    for c in range(best_k):
        m, mt = labels == c, labels_t == c
        clusters.append({"cluster": c, "top_words": names[c], "size_train": int(m.sum()),
                         "fail_rate_train": float(train.label.values[m].mean()),
                         "fail_rate_test": float(test.label.values[mt].mean()) if mt.any() else None,
                         "mean_score_train": float(train.score.values[m].mean()),
                         "main_source": str(train.source[m].value_counts().index[0])})
    clusters.sort(key=lambda r: -r["fail_rate_train"])
    for r in clusters:
        print(f"  c{r['cluster']:2d} n={r['size_train']:6d} fail={r['fail_rate_train']:.3f} {', '.join(r['top_words'][:6])}")

    hier, h_idx, h_labels = C.hierarchical_vs_kmeans(Z, best_k)
    dens, d_idx, d_labels = C.dbscan_dense_topics(Z)
    dens_groups = []
    for c in range(dens["n_clusters"]):
        rows = d_idx[d_labels == c]
        dens_groups.append({"size": int(len(rows)), "fail_rate": float(train.label.values[rows].mean()),
                            "example": train.prompt.values[rows[0]][:120]})
    dens_groups.sort(key=lambda r: -r["size"])
    dens["largest_groups"] = dens_groups[:10]
    noise = d_idx[d_labels == -1]
    dens["fail_rate_noise"] = float(train.label.values[noise].mean()) if len(noise) else None
    print("hierarchical:", hier)
    print("dbscan:", {k: v for k, v in dens.items() if k != "largest_groups"})

    joblib.dump({"kmeans": km, "svd_dims": 64}, config.MODELS / "kmeans.joblib", compress=3)
    np.save(config.DATA_PROCESSED / "cluster_train.npy", labels)
    np.save(config.DATA_PROCESSED / "cluster_test.npy", labels_t)

    # figures: elbow + silhouette, and failure rate per cluster
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ks = [r["k"] for r in sweep]
    ax[0].plot(ks, [r["inertia"] for r in sweep], marker="o")
    ax[0].set(title="K-Means elbow curve", xlabel="k", ylabel="inertia")
    ax[1].plot(ks, [r["silhouette"] for r in sweep], marker="o", color="tab:orange")
    ax[1].axvline(best_k, ls="--", c="grey")
    ax[1].set(title="Silhouette score", xlabel="k")
    fig.tight_layout()
    fig.savefig(config.FIGURES / "kmeans_selection.png", dpi=130)
    fig, ax = plt.subplots(figsize=(9, 0.32 * best_k + 1))
    lbl = [f"{', '.join(r['top_words'][:3])} (n={r['size_train']})" for r in clusters]
    ax.barh(lbl[::-1], [r["fail_rate_train"] for r in clusters][::-1])
    ax.axvline(train.label.mean(), ls="--", c="grey", label="overall")
    ax.set(xlabel="cheap-model failure rate", title="Which prompt types does the cheap model fail on?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIGURES / "cluster_failure_rates.png", dpi=130)

    save_json("05_clustering.json", {"kmeans_sweep": sweep, "k": best_k, "clusters": clusters,
                                     "hierarchical": hier, "dbscan": dens})
