"""Step 2b - neural sentence embeddings on the GPU (Apple Silicon "mps", CUDA, or CPU fallback).

Model: BAAI/bge-small-en-v1.5 (33M parameters, 384-dim, L2-normalised). The same model is
available as ONNX through `fastembed`, which is how the web service computes it on a CPU;
`scripts/09_check_onnx_parity.py` verifies the two give the same vectors.
"""
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from _common import save_json
from llm_router import config, data

MODEL = "BAAI/bge-small-en-v1.5"


def device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


if __name__ == "__main__":
    dev = device()
    print("device:", dev, flush=True)
    model = SentenceTransformer(MODEL, device=dev)
    model.max_seq_length = 512
    out = {"model": MODEL, "device": dev, "dims": model.get_sentence_embedding_dimension()}
    for split in ("train", "test"):
        df = data.load(split)
        texts = df.prompt.tolist()
        # sort by length so each batch has similar lengths (much less padding = faster)
        order = np.argsort([len(t) for t in texts])
        t = time.perf_counter()
        emb = model.encode([texts[i] for i in order], batch_size=128, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=False)
        secs = time.perf_counter() - t
        E = np.empty_like(emb)
        E[order] = emb
        np.save(config.DATA_PROCESSED / f"E_{split}.npy", E.astype(np.float32))
        out[split] = {"rows": len(texts), "seconds": round(secs, 1), "ms_per_prompt": round(1000 * secs / len(texts), 2)}
        print(split, out[split], flush=True)
    save_json("02b_embeddings.json", out)
