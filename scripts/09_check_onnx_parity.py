"""Step 9 - check that CPU (ONNX) embeddings used online match GPU (PyTorch) embeddings used in training."""
import time

import numpy as np

from _common import save_json
from llm_router import config, data
from llm_router.embedder import embed

if __name__ == "__main__":
    test = data.load("test")
    n = 1000
    E_gpu = np.load(config.DATA_PROCESSED / "E_test.npy")[:n]
    t = time.perf_counter()
    E_cpu = embed(test.prompt.values[:n].tolist())
    secs = time.perf_counter() - t
    cos = (E_gpu * E_cpu).sum(1)
    out = {"prompts": n, "cosine_min": float(cos.min()), "cosine_mean": float(cos.mean()),
           "cosine_p01": float(np.quantile(cos, 0.01)), "cpu_ms_per_prompt": round(1000 * secs / n, 2)}
    print(out)
    save_json("09_onnx_parity.json", out)
