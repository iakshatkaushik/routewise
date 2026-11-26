"""CPU sentence embeddings for the web service (ONNX Runtime via fastembed).

Training computes the same bge-small-en-v1.5 embeddings with PyTorch on the GPU
(scripts/02b_embeddings.py); scripts/09_check_onnx_parity.py checks the two agree.
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

MODEL = "BAAI/bge-small-en-v1.5"


@lru_cache(maxsize=1)
def _model():
    from fastembed import TextEmbedding
    threads = int(os.getenv("EMBED_THREADS", "1"))
    # The CPU memory arena pre-allocates large buffers; turning it off saves memory on small
    # servers at a small speed cost. Set EMBED_MEM_ARENA=1 to turn it back on.
    arena = os.getenv("EMBED_MEM_ARENA", "0") == "1"
    return TextEmbedding(MODEL, threads=threads, cache_dir=os.getenv("FASTEMBED_CACHE", None),
                         enable_cpu_mem_arena=arena)


def embed(texts) -> np.ndarray:
    texts = list(texts)
    order = np.argsort([len(t) for t in texts])  # similar lengths per batch = less padding
    vecs = np.asarray(list(_model().embed([texts[i] for i in order], batch_size=32)), dtype=np.float32)
    out = np.empty_like(vecs)
    out[order] = vecs
    return out / np.linalg.norm(out, axis=1, keepdims=True).clip(1e-12)
