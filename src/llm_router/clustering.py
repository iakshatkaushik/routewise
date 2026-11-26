"""Clustering helpers (syllabus unit: Clustering - partitioned, hierarchical, density-based)."""
from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering, MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from . import config


def unit(Z):
    """L2-normalise so that Euclidean distance behaves like cosine distance."""
    return normalize(Z.astype(np.float32))


def kmeans_sweep(Z, ks, sample_for_silhouette=8000):
    rng = np.random.default_rng(config.SEED)
    idx = rng.choice(len(Z), min(sample_for_silhouette, len(Z)), replace=False)
    out = []
    for k in ks:
        km = MiniBatchKMeans(k, batch_size=4096, n_init=3, random_state=config.SEED).fit(Z)
        sil = silhouette_score(Z[idx], km.labels_[idx])
        out.append({"k": k, "inertia": float(km.inertia_), "silhouette": float(sil)})
    return out


def fit_kmeans(Z, k):
    return MiniBatchKMeans(k, batch_size=4096, n_init=5, random_state=config.SEED).fit(Z)


def hierarchical_vs_kmeans(Z, k, n=4000):
    """Ward agglomerative clustering on a sample, compared with K-Means via adjusted Rand index."""
    rng = np.random.default_rng(config.SEED)
    idx = rng.choice(len(Z), min(n, len(Z)), replace=False)
    hc = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(Z[idx])
    km = MiniBatchKMeans(k, batch_size=2048, n_init=5, random_state=config.SEED).fit(Z[idx])
    return {
        "sample_size": int(len(idx)),
        "silhouette_hierarchical": float(silhouette_score(Z[idx], hc.labels_)),
        "silhouette_kmeans": float(silhouette_score(Z[idx], km.labels_)),
        "adjusted_rand_index": float(adjusted_rand_score(hc.labels_, km.labels_)),
    }, idx, hc.labels_


def dbscan_dense_topics(Z, n=15000, min_samples=15, quantile=0.10):
    """DBSCAN finds tight groups of near-identical prompt types and marks the rest as noise.

    eps is chosen from the k-distance curve: the `quantile` of each point's distance to its
    `min_samples`-th neighbour.
    """
    rng = np.random.default_rng(config.SEED)
    idx = rng.choice(len(Z), min(n, len(Z)), replace=False)
    X = Z[idx]
    dist, _ = NearestNeighbors(n_neighbors=min_samples).fit(X).kneighbors(X)
    eps = float(np.quantile(dist[:, -1], quantile))
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
    return {"eps": eps, "n_clusters": int(labels.max() + 1),
            "noise_fraction": float((labels == -1).mean())}, idx, labels


def top_terms(featurizer, X_sparse, labels, k, n_terms=8):
    """Name each cluster by its most *distinctive* words.

    Score = mean TF-IDF inside the cluster minus mean TF-IDF overall, ignoring stop words and
    bigrams made only of stop words, so names are not dominated by "the", "of", "and".
    """
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    names = featurizer.word_.get_feature_names_out()
    n_word = len(names)
    W = X_sparse[:, :n_word]
    overall = np.asarray(W.mean(axis=0)).ravel()
    keep = np.array([not all(tok in ENGLISH_STOP_WORDS or tok.isdigit() for tok in n.split()) for n in names])
    out = {}
    for c in range(k):
        rows = np.where(labels == c)[0]
        if len(rows) == 0:
            out[c] = []
            continue
        score = np.asarray(W[rows].mean(axis=0)).ravel() - overall
        score[~keep] = -np.inf
        out[c] = [str(names[i]) for i in np.argsort(-score)[:n_terms]]
    return out
