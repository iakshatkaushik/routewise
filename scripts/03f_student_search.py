"""Step 3f - wider search over the distilled student (validation only).

Starting point: the best student of step 3e (bge-small, alpha 0.2, 4 epochs, 512 tokens, lr 5e-5, batch 64,
deficit weight 1, CLS pooling). Stage A changes one setting at a time, including other small base models whose
int8 ONNX file fits the 95 MB limit; stage B combines the stage-A winners and repeats the best configuration with
other seeds (for a seed-ensemble check). Every run is `03c_finetune.py --tune` (the test set is never scored);
results go to reports/finetune_sweep.json, validation logits to models/study_logits/ (not in git).

Students are small (about 5 GB of GPU memory each), so many share the GPU at once.

  python scripts/03f_student_search.py --teacher models/teacher/teacher_oof_qwen3_4b.npz --stage A
  python scripts/03f_student_search.py --teacher ... --stage B --seeds "" --config "--base thenlper/gte-small" ...
"""
import argparse
import os
import re
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
BASE = {"--base": "BAAI/bge-small-en-v1.5", "--lr": "5e-5", "--batch-size": "64", "--epochs": "4",
        "--max-tokens": "512", "--deficit-weight": "1", "--distill-alpha": "0.2", "--pooling": "cls"}
STAGE_A = [
    ({"--lr": "3e-5"}, "lr 3e-5"),
    ({"--lr": "8e-5"}, "lr 8e-5"),
    ({"--lr": "1e-4"}, "lr 1e-4"),
    ({"--batch-size": "32"}, "batch 32"),
    ({"--batch-size": "128"}, "batch 128"),
    ({"--deficit-weight": "0.5"}, "deficit weight 0.5"),
    ({"--deficit-weight": "2"}, "deficit weight 2"),
    ({"--pooling": "mean"}, "mean pooling"),
    ({"--ema": "0.999"}, "EMA 0.999"),
    ({"--distill-alpha": "0.1"}, "alpha 0.1"),
    ({"--distill-alpha": "0.3"}, "alpha 0.3"),
    ({"--epochs": "6"}, "6 epochs"),
    ({"--base": "intfloat/e5-small-v2"}, "base e5-small-v2"),
    ({"--base": "thenlper/gte-small"}, "base gte-small"),
    ({"--base": "sentence-transformers/all-MiniLM-L12-v2"}, "base all-MiniLM-L12-v2"),
    # bigger students that still fit the 95 MB file limit as int8 ONNX (measured: 66.8-82.0 MB)
    ({"--base": "BAAI/bge-base-en-v1.5", "--num-layers": "6"}, "base bge-base, first 6 layers"),
    ({"--base": "BAAI/bge-base-en-v1.5", "--num-layers": "8"}, "base bge-base, first 8 layers"),
    ({"--base": "distilbert-base-uncased"}, "base distilbert-base-uncased"),
    ({"--base": "distilroberta-base"}, "base distilroberta-base"),
    ({}, "reference (stage-A starting point)"),
]
ENV = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}


def run(args):
    cmd = [sys.executable, "-u", str(HERE / "03c_finetune.py"), "--tune", *args]
    stem = re.sub(r"[^a-z0-9]+", "_", args[args.index("--tag") + 1].lower()).strip("_")
    (ROOT / "logs").mkdir(exist_ok=True)
    print("running:", " ".join(cmd[2:]), flush=True)
    with open(ROOT / "logs" / f"{stem}.log", "w") as log:          # logs/ is not in git
        return subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=ENV).returncode


def build(change, label, teacher, teacher_name):
    opts = BASE | change
    return [*[x for kv in opts.items() for x in kv], "--teacher", teacher, "--save-logits", "models/study_logits",
            "--tag", f"search: student of {teacher_name}, {label}"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--teacher-name", required=True)
    ap.add_argument("--stage", choices=["A", "B"], required=True)
    ap.add_argument("--config", action="append", default=[],
                    help="stage B: extra args (on top of the start point) to combine; repeat for several")
    ap.add_argument("--seeds", default="43,44,45", help="stage B: extra seeds of the combined configuration")
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--seeds-only", action="store_true", help="stage B: only the extra seeds (seed 42 already run)")
    a = ap.parse_args()
    if a.stage == "A":
        jobs = [build(c, lab, a.teacher, a.teacher_name) for c, lab in STAGE_A]
    else:
        jobs = []
        for cfg in a.config:
            toks = shlex.split(cfg)
            change = dict(zip(toks[::2], toks[1::2]))
            label = "combined: " + (", ".join(f"{k.strip('-')} {v}" for k, v in change.items()) or "start point")
            if not a.seeds_only:
                jobs.append(build(change, label, a.teacher, a.teacher_name))
            jobs += [build(change | {"--seed": s}, f"{label}, seed {s}", a.teacher, a.teacher_name)
                     for s in a.seeds.split(",") if s]
    with ThreadPoolExecutor(a.parallel) as ex:
        codes = list(ex.map(run, jobs))
    print("exit codes:", codes)
