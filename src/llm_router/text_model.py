"""CPU inference for the fine-tuned transformer router (ONNX Runtime, no PyTorch needed).

The exported model file holds the encoder and both heads; it returns two logits per prompt:
column 0 = "cheap model fails", column 1 = expected score deficit.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import config

THREADS = int(os.getenv("TEXT_MODEL_THREADS", "1"))


class TextRouterModel:
    def __init__(self, onnx_path: Path, tokenizer_path: Path, max_tokens: int, threads: int | None = None):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads or THREADS
        opts.inter_op_num_threads = 1
        opts.enable_cpu_mem_arena = os.getenv("EMBED_MEM_ARENA", "0") == "1"
        self.session = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
        self.tok = Tokenizer.from_file(str(tokenizer_path))
        self.tok.no_padding()
        self.tok.enable_truncation(max_tokens)
        self.max_tokens = max_tokens

    def logits(self, texts, batch_size: int = 1) -> np.ndarray:
        """Two logits per text. batch_size=1 (the default, as in the web service) avoids padding: int8
        dynamic quantisation scales activations per batch, so padded batches give slightly different outputs."""
        texts = list(texts)
        enc = [e.ids for e in self.tok.encode_batch(texts)]
        out = np.zeros((len(texts), 2), dtype=np.float32)
        order = np.argsort([len(e) for e in enc], kind="stable")      # similar lengths share a batch
        for s in range(0, len(order), batch_size):
            ix = order[s:s + batch_size]
            L = max(len(enc[i]) for i in ix)
            ids = np.zeros((len(ix), L), dtype=np.int64)
            att = np.zeros((len(ix), L), dtype=np.int64)
            for r, i in enumerate(ix):
                ids[r, :len(enc[i])] = enc[i]
                att[r, :len(enc[i])] = 1
            out[ix] = self.session.run(["logits"], {"input_ids": ids, "attention_mask": att})[0]
        return out


@lru_cache(maxsize=2)
def load(onnx_file: str, tokenizer_file: str, max_tokens: int) -> TextRouterModel:
    return TextRouterModel(config.MODELS / onnx_file, config.MODELS / tokenizer_file, max_tokens)
