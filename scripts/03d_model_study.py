"""Step 3d - validation-only study: does a larger pretrained model route better?

Fine-tunes larger encoders and decoder-only language models with the same two-head setup as step 3c
(`03c_finetune.py --tune`), one standard configuration per model (not tuned per model). Results are
appended to reports/finetune_sweep.json; the test set is never scored. Validation logits are saved to
models/study_logits/ (not in git) for the ensemble check printed at the end.

None of these models is deployed: their int8 ONNX files would be far over the 95 MB file limit (bge-base
alone is 109.7 MB) and slower than the latency budget. The best one is used as a teacher in step 3e.

Runs share one GPU, two at a time (four large models at once ran out of memory on an 80 GB H100); run
Qwen3-1.7B and Qwen3-4B on their own (about 62 GB and 74 GB of GPU memory).

  python scripts/03d_model_study.py            # run the study, then print the table
  python scripts/03d_model_study.py --table    # only print the table and the ensemble check
"""
import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
LOGITS = "models/study_logits"

RUNS = [
    ["--base", "BAAI/bge-base-en-v1.5", "--lr", "3e-5", "--batch-size", "32", "--tag", "study: bge-base, lr 3e-5, batch 32"],
    ["--base", "BAAI/bge-large-en-v1.5", "--lr", "2e-5", "--batch-size", "32", "--tag", "study: bge-large, lr 2e-5, batch 32"],
    ["--base", "answerdotai/ModernBERT-large", "--lr", "2e-5", "--batch-size", "32",
     "--tag", "study: ModernBERT-large, lr 2e-5, batch 32"],
    ["--base", "microsoft/deberta-v3-large", "--lr", "1e-5", "--batch-size", "32", "--eval-batch-size", "128",
     "--tag", "study: deberta-v3-large, lr 1e-5, batch 32 (fp32 master weights)"],
    ["--base", "Qwen/Qwen3-0.6B", "--pooling", "last", "--lr", "1e-5", "--batch-size", "32", "--eval-batch-size", "128",
     "--tag", "study: Qwen3-0.6B (decoder, last-token), lr 1e-5, batch 32"],
    # Qwen3-0.6B peaked after 1 of 2 epochs, so the larger decoders train for 1 epoch
    ["--base", "Qwen/Qwen3-1.7B", "--pooling", "last", "--lr", "1e-5", "--batch-size", "32", "--eval-batch-size", "128",
     "--epochs", "1", "--tag", "study: Qwen3-1.7B (decoder, last-token), lr 1e-5, batch 32, 1 epoch"],
    # Qwen3-4B: fp32 weights, gradients and Adam state take ~64 GB, so it runs with gradient checkpointing
    ["--base", "Qwen/Qwen3-4B", "--pooling", "last", "--lr", "1e-5", "--batch-size", "32", "--eval-batch-size", "64",
     "--grad-ckpt", "--epochs", "1", "--tag", "study: Qwen3-4B (decoder, last-token), lr 1e-5, batch 32, 1 epoch, grad-ckpt"],
    # Qwen3-8B: a full fine-tune does not fit in 80 GB (~128 GB of fp32 weights, gradients and Adam state), so the
    # weights stay frozen in bf16 and LoRA adapters (rank 16) are trained
    ["--base", "Qwen/Qwen3-8B", "--pooling", "last", "--lr", "1e-4", "--batch-size", "32", "--eval-batch-size", "64",
     "--grad-ckpt", "--epochs", "1", "--lora-r", "16",
     "--tag", "study: Qwen3-8B (decoder, last-token), LoRA r16, lr 1e-4, batch 32, 1 epoch, grad-ckpt"],
]


def run(args):
    cmd = [sys.executable, "-u", str(HERE / "03c_finetune.py"), "--tune", "--save-logits", LOGITS, *args]
    print("running:", " ".join(cmd[2:]), flush=True)
    env = {**__import__("os").environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    import re
    (ROOT / "logs").mkdir(exist_ok=True)
    stem = re.sub(r"[^a-z0-9]+", "_", args[args.index("--tag") + 1].lower()).strip("_")
    with open(ROOT / "logs" / f"{stem}.log", "w") as log:          # logs/ is not in git
        return subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env).returncode


def table():
    runs = [r for r in json.loads((ROOT / "reports" / "finetune_sweep.json").read_text())["runs"]
            if r["tag"].startswith("study:")]
    print(f"{'tag':78s} {'params':>7} {'min':>6} {'ep*':>4} {'AUC fail':>8} {'APGR fail':>9} {'APGR def':>9}")
    for r in runs:
        b = r["best_checkpoint"]
        print(f"{r['tag'][:78]:78s} {r.get('params_millions', 0):>6.0f}M {r['train_minutes']:>6.1f} {b['epoch']:>4} "
              f"{b['val_auc_fail']:>8.4f} {b['val_apgr_fail']:>9.4f} {b['val_apgr_deficit']:>9.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", action="store_true")
    if not ap.parse_args().table:
        big = [r for r in RUNS if r[1] in ("Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B", "Qwen/Qwen3-8B")]
        with ThreadPoolExecutor(2) as ex:
            codes = list(ex.map(run, [r for r in RUNS if r not in big]))
        codes += [run(r) for r in big]                     # one at a time
        print("exit codes:", codes)
    table()
