"""Step 7a - export the fine-tuned transformer (step 3c) to ONNX for CPU serving.

1. Loads models/finetuned/ and exports it to ONNX twice: fp32 and int8 (dynamic quantisation).
2. Saves the base model's tokenizer as models/text_tokenizer.json.
3. Scores the validation set with both files through the serving code (text_model.TextRouterModel).
   int8 is served unless it loses more than 0.003 validation AUC; then an fp16 file is tried
   (onnxconverter-common), and fp32 is the last resort. The served file must be under 95 MB.
4. Recomputes the validation and test head outputs with the exact served file and stores them, as
   sigmoid probabilities, in the "Fine-tuned Transformer (GPU)" columns of the score tables, so step 7's
   stacker is trained on what production computes.
5. Measures single-prompt latency on 1 CPU thread (300 test prompts) and writes reports/07a_text_export.json.

The choice between int8, fp16 and fp32 uses validation only; test outputs are computed for the served file only.
"""
import json
import os
import shutil
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from _common import save_json
from llm_router import config, data, metrics
from llm_router.finetune import export_onnx, load_finetuned
from llm_router.text_model import TextRouterModel

SRC = config.MODELS / "finetuned"
ONNX = config.MODELS / "text_router.onnx"
FP32 = config.MODELS / "text_router.fp32.onnx"      # comparison only, not deployed (ignored by git)
FP16 = config.MODELS / "text_router.fp16.onnx"
TOKENIZER = config.MODELS / "text_tokenizer.json"
NAME = "Fine-tuned Transformer (GPU)"
NAME_DEFICIT = "Fine-tuned Transformer (GPU), score-deficit head"
MAX_AUC_LOSS = 0.003
MAX_MB = 95
WORKERS = min(48, os.cpu_count() or 1)


def _init(path, tok, max_tokens):
    global _M
    _M = TextRouterModel(path, tok, max_tokens, threads=1)


def _score(chunk):
    return _M.logits(chunk, batch_size=1)


def score(path, prompts, max_tokens):
    """Score prompts one at a time (batch size 1, exactly as the web service does), spread over
    WORKERS single-threaded processes."""
    from concurrent.futures import ProcessPoolExecutor
    chunks = [prompts[i:i + 256] for i in range(0, len(prompts), 256)]
    with ProcessPoolExecutor(WORKERS, initializer=_init, initargs=(str(path), str(TOKENIZER), max_tokens)) as ex:
        return np.vstack(list(ex.map(_score, chunks)))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def val_report(logits, yv, sv):
    return {"val_auc_fail": round(float(roc_auc_score(yv, logits[:, 0])), 5),
            "val_auc_deficit": round(float(roc_auc_score(yv, logits[:, 1])), 5),
            "val_apgr_fail": round(metrics.routing_report(logits[:, 0], sv)["apgr"], 5),
            "val_apgr_deficit": round(metrics.routing_report(logits[:, 1], sv)["apgr"], 5)}


if __name__ == "__main__":
    meta = json.loads((SRC / "router_meta.json").read_text())
    max_tokens = int(meta["max_tokens"])
    train, test = data.load("train"), data.load("test")
    val = pd.read_parquet(config.REPORTS / "val_scores.parquet")
    rows = val["row"].values
    yv, sv = train.label.values[rows], train.score.values[rows]
    clip = lambda ps: [str(p)[: config.MAX_PROMPT_CHARS] for p in ps]   # as Router.risk does  # noqa: E731
    val_prompts = clip(train.prompt.values[rows])

    model = load_finetuned(SRC, "cpu")
    export_onnx(model, FP32, quantize=False)
    export_onnx(model, ONNX, quantize=True)
    shutil.copy(SRC / "tokenizer.json", TOKENIZER)
    mb = lambda p: round(p.stat().st_size / 1e6, 2)  # noqa: E731
    print(f"exported: fp32 {mb(FP32)} MB, int8 {mb(ONNX)} MB", flush=True)

    # PyTorch reference on a sample, to check the export itself (fp32 should match closely)
    import torch
    from transformers import AutoTokenizer
    hf_tok = AutoTokenizer.from_pretrained(str(SRC))
    sample = val_prompts[:256]
    with torch.no_grad():
        ref = []
        for p in sample:
            e = hf_tok([p], truncation=True, max_length=max_tokens, return_tensors="pt")
            ref.append(model(e["input_ids"], e["attention_mask"]).numpy()[0])
    ref = np.array(ref)

    variants = {"int8": ONNX, "fp32": FP32}
    results, logits_val = {}, {}
    for name, path in variants.items():
        m = TextRouterModel(path, TOKENIZER, max_tokens, threads=4)
        t = time.perf_counter()
        logits_val[name] = score(path, val_prompts, max_tokens)
        secs = time.perf_counter() - t
        results[name] = {"file_mb": mb(path), **val_report(logits_val[name], yv, sv),
                         "max_abs_logit_diff_vs_pytorch_256": round(float(np.abs(m.logits(sample) - ref).max()), 5),
                         # padded batches of 32: checks the attention mask survives export
                         "max_abs_logit_diff_vs_pytorch_256_padded_batches":
                             round(float(np.abs(m.logits(sample, batch_size=32) - ref).max()), 5),
                         "val_scoring_seconds": round(secs, 1), "workers": WORKERS, "batch_size": 1}
        print(name, results[name], flush=True)

    loss = results["fp32"]["val_auc_fail"] - results["int8"]["val_auc_fail"]
    served = "int8" if loss <= MAX_AUC_LOSS and results["int8"]["file_mb"] < MAX_MB else None
    if served is None:
        print(f"int8 loses {loss:.4f} val AUC (> {MAX_AUC_LOSS}); trying fp16", flush=True)
        import onnx
        from onnxconverter_common import float16
        onnx.save(float16.convert_float_to_float16(onnx.load(str(FP32)), keep_io_types=True), str(FP16))
        logits_val["fp16"] = score(FP16, val_prompts, max_tokens)
        results["fp16"] = {"file_mb": mb(FP16), **val_report(logits_val["fp16"], yv, sv)}
        print("fp16", results["fp16"], flush=True)
        for cand in ("fp16", "fp32"):
            if results[cand]["file_mb"] < MAX_MB and results["fp32"]["val_auc_fail"] - results[cand]["val_auc_fail"] <= MAX_AUC_LOSS:
                served = cand
                break
        assert served is not None, "no ONNX variant is both accurate enough and under 95 MB"
        shutil.copy({"fp16": FP16, "fp32": FP32}[served], ONNX)
    print(f"served variant: {served} ({mb(ONNX)} MB); int8 val AUC loss vs fp32: {loss:.5f}", flush=True)

    # outputs of the exact served file, for the stacker (validation) and the final evaluation (test)
    lv = score(ONNX, val_prompts, max_tokens)
    lt = score(ONNX, clip(test.prompt.values), max_tokens)
    assert np.array_equal(lv, logits_val[served]) or served != "int8" or np.abs(lv - logits_val[served]).max() < 1e-5
    for fname, L in (("val_scores.parquet", lv), ("test_scores.parquet", lt)):
        df = pd.read_parquet(config.REPORTS / fname)
        df[NAME], df[NAME_DEFICIT] = sigmoid(L[:, 0]), sigmoid(L[:, 1])
        df.to_parquet(config.REPORTS / fname)

    # step 3's comparison table shows the served model: its entries get the ONNX outputs' metrics, and the
    # PyTorch numbers written by step 3c are kept under "pytorch".
    c3_path = config.REPORTS / "03_classifiers.json"
    c3 = json.loads(c3_path.read_text())
    for name, col in ((NAME, 0), (NAME_DEFICIT, 1)):
        e = c3["models"][name]
        if "pytorch" not in e:
            e["pytorch"] = {k: e[k] for k in ("val_auc", "test", "routing")}
        pv, pt = sigmoid(lv[:, col]), sigmoid(lt[:, col])
        e["val_auc"] = round(float(roc_auc_score(yv, pv)), 4)
        e["test"] = metrics.classification_report(test.label.values, pt, float(np.quantile(pv, 0.70)))
        e["routing"] = metrics.routing_report(pt, test.score.values)
        e["served_as"] = f"ONNX {served}, CPU, one prompt at a time"
        print(f"{name}: served {served} test AUC {e['test']['roc_auc']:.4f} APGR {e['routing']['apgr']:.4f} "
              f"(PyTorch: {e['pytorch']['test']['roc_auc']:.4f} / {e['pytorch']['routing']['apgr']:.4f})", flush=True)
    c3_path.write_text(json.dumps(c3, indent=2))

    # single-prompt latency on one CPU thread, as in production
    one = TextRouterModel(ONNX, TOKENIZER, max_tokens, threads=1)
    prompts = clip(test.prompt.values[:300])
    one.logits(prompts[:5])                                   # warm-up
    lat = []
    for p in prompts:
        t = time.perf_counter()
        one.logits([p])
        lat.append((time.perf_counter() - t) * 1000)

    # small reference for tests/test_router_components.py: served-model logits for a few prompts
    ref_prompts = ["Write a haiku about autumn.", "Prove that there are infinitely many primes.",
                   "Fix this bug: def add(a, b): return a - b", "hi"] + clip(test.prompt.values[300:302])
    fx = config.ROOT / "tests" / "fixtures"
    fx.mkdir(exist_ok=True)
    (fx / "text_model_reference.json").write_text(json.dumps(
        {"max_tokens": max_tokens, "onnx": ONNX.name, "prompts": ref_prompts,
         "logits": one.logits(ref_prompts).round(6).tolist()}, indent=1))

    save_json("07a_text_export.json", {
        "source": "models/finetuned", "base_model": meta["base"], "max_tokens": max_tokens,
        "served_variant": served, "served_file": ONNX.name, "served_mb": mb(ONNX), "tokenizer_file": TOKENIZER.name,
        "variants_validation": results, "int8_val_auc_loss_vs_fp32": round(float(loss), 5),
        "rule": f"serve int8 unless it loses more than {MAX_AUC_LOSS} validation AUC; file must be < {MAX_MB} MB",
        "served_validation": val_report(lv, yv, sv),
        "latency_1_thread_ms": {"p50": round(float(np.percentile(lat, 50)), 2),
                                "p95": round(float(np.percentile(lat, 95)), 2),
                                "mean": round(float(np.mean(lat)), 2), "prompts": len(lat)},
        "cpu": os.uname().machine,
    })
