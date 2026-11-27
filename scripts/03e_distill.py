"""Step 3e - knowledge distillation: a large teacher trains the small, deployable bge-small model.

1. Teacher: Qwen3-1.7B (the strongest model of step 3d whose five fold runs fit the time budget) is
   fine-tuned 5 times, each time holding out one
   fold of the training part (`03c_finetune.py --tune --n-folds 5 --fold k`). Every training prompt
   then has an out-of-fold teacher prediction, i.e. one from a model that never saw it. Checkpoints are
   chosen on the validation split, which is never part of the teacher file.
2. The out-of-fold logits are merged into models/teacher/teacher_oof.npz (not in git).
3. Students: bge-small with the step-3c configuration, trained on
       target = alpha * label + (1 - alpha) * teacher probability   (for both heads)
   for a few alpha values (`--tune`, validation only; results in reports/finetune_sweep.json).

  python scripts/03e_distill.py                 # teacher folds, merge, students
  python scripts/03e_distill.py --students      # merge + students only (teacher folds already run)
  python scripts/03e_distill.py --round2        # second and third student rounds only
  python scripts/03e_distill.py --round3        # third student round only
  python scripts/03e_distill.py --teacher-4b    # Qwen3-4B teacher folds + ensemble teacher + their students
  python scripts/03e_distill.py --teacher-8b    # Qwen3-8B LoRA teacher folds + 3 seeds of the step-3f student
"""
import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parent
TEACHER_DIR = ROOT / "models" / "teacher"
TEACHER_FILE = TEACHER_DIR / "teacher_oof.npz"            # Qwen3-1.7B teacher (the first one)
K = 5
TEACHER = ["--base", "Qwen/Qwen3-1.7B", "--pooling", "last", "--lr", "1e-5", "--batch-size", "32",
           "--eval-batch-size", "128", "--epochs", "1"]
TEACHER_PARALLEL = 1          # one Qwen3-1.7B run uses about 62 GB of the 80 GB GPU
# second teacher: Qwen3-4B (the strongest model of step 3d), same settings as its study run, one validation
# checkpoint per fold (its study run peaked at the end of the epoch); about 74 GB of GPU memory, one at a time
TEACHER_4B = ["--base", "Qwen/Qwen3-4B", "--pooling", "last", "--lr", "1e-5", "--batch-size", "32",
              "--eval-batch-size", "64", "--epochs", "1", "--grad-ckpt", "--evals-per-epoch", "1"]
TEACHER_4B_FILE = TEACHER_DIR / "teacher_oof_qwen3_4b.npz"
TEACHER_ENS_FILE = TEACHER_DIR / "teacher_oof_ensemble.npz"   # mean of the two teachers' logits
STUDENT = ["--lr", "5e-5", "--batch-size", "64", "--epochs", "2", "--max-tokens", "256"]   # step-3c choice
ALPHAS = ["0.5", "0.2", "0.0"]
# round 2, from the best alpha of round 1 (0.2): one change at a time, plus a seed-noise check
ROUND2 = [
    (["--epochs", "3"], "alpha 0.2, 3 epochs"),
    (["--max-tokens", "512"], "alpha 0.2, max_tokens 512"),
    (["--distill-alpha", "0.1"], "alpha 0.1"),
    (["--seed", "43"], "alpha 0.2, seed 43 (noise check)"),
]
# round 3: round 2's 3-epoch run was still improving at its last checkpoint
ROUND3 = [
    (["--epochs", "4"], "alpha 0.2, 4 epochs"),
    (["--epochs", "5"], "alpha 0.2, 5 epochs"),
    (["--epochs", "4", "--max-tokens", "512"], "alpha 0.2, 4 epochs, max_tokens 512"),
]
ENV = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}


def run(args):
    cmd = [sys.executable, "-u", str(HERE / "03c_finetune.py"), "--tune", *args]
    print("running:", " ".join(cmd[2:]), flush=True)
    import re
    (ROOT / "logs").mkdir(exist_ok=True)
    stem = re.sub(r"[^a-z0-9]+", "_", args[args.index("--tag") + 1].lower()).strip("_")
    with open(ROOT / "logs" / f"{stem}.log", "w") as log:          # logs/ is not in git
        return subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=ENV).returncode


def teacher_tag(k):
    return f"teacher: Qwen3-1.7B, lr 1e-5, batch 32, 1 epoch, fold {k} of {K}"


def teacher_tag_4b(k):
    return f"teacher: Qwen3-4B, lr 1e-5, batch 32, 1 epoch, grad-ckpt, fold {k} of {K}"


def merge(tag=teacher_tag, out=TEACHER_FILE):
    import re
    rows, logits = [], []
    for k in range(K):
        stem = re.sub(r"[^a-z0-9]+", "_", tag(k).lower()).strip("_")
        d = np.load(TEACHER_DIR / f"{stem}.npz")
        rows.append(d["oof_rows"])
        logits.append(d["oof_logits"])
    rows, logits = np.concatenate(rows), np.concatenate(logits)
    assert len(np.unique(rows)) == len(rows), "folds overlap"
    np.savez(out, rows=rows, logits=logits)
    print(f"teacher file: {out} ({len(rows)} training rows)")


def ensemble():
    """Mean of the two teachers' out-of-fold logits, aligned by training row."""
    a, b = np.load(TEACHER_FILE), np.load(TEACHER_4B_FILE)
    ia, ib = np.argsort(a["rows"]), np.argsort(b["rows"])
    assert np.array_equal(a["rows"][ia], b["rows"][ib]), "teachers cover different rows"
    np.savez(TEACHER_ENS_FILE, rows=a["rows"][ia], logits=(a["logits"][ia] + b["logits"][ib]) / 2)
    print(f"teacher file: {TEACHER_ENS_FILE}")


# third teacher: Qwen3-8B with LoRA (frozen bf16 weights + rank-16 adapters; ~22 GB while training, so three folds
# share the GPU). The first attempts ran out of memory in the prediction pass because 03c_finetune.py predicted the
# saved logits with a hard-coded batch of 512 (fixed: it now uses --eval-batch-size). Inference batch size changes
# no result (fold 2 was saved by the first attempt).
TEACHER_8B = ["--base", "Qwen/Qwen3-8B", "--pooling", "last", "--lr", "1e-4", "--batch-size", "32",
              "--eval-batch-size", "32", "--epochs", "1", "--grad-ckpt", "--lora-r", "16", "--evals-per-epoch", "1"]
TEACHER_8B_FILE = TEACHER_DIR / "teacher_oof_qwen3_8b_lora.npz"
# student configuration chosen by the step-3f search (3-seed means on validation)
SEARCH_STUDENT = ["--base", "BAAI/bge-small-en-v1.5", "--pooling", "mean", "--lr", "8e-5", "--batch-size", "64",
                  "--epochs", "4", "--max-tokens", "512", "--deficit-weight", "1", "--distill-alpha", "0.1"]


def teacher_tag_8b(k):
    return f"teacher: Qwen3-8B LoRA r16, lr 1e-4, batch 32, 1 epoch, grad-ckpt, fold {k} of {K}"


def teacher_8b_round():
    import re
    todo = [k for k in range(K) if not (TEACHER_DIR / (re.sub(r"[^a-z0-9]+", "_", teacher_tag_8b(k).lower()).strip("_")
                                                       + ".npz")).exists()]   # folds without a saved file
    print("Qwen3-8B folds to run:", todo, flush=True)
    with ThreadPoolExecutor(3) as ex:
        codes = list(ex.map(run, [[*TEACHER_8B, "--n-folds", str(K), "--fold", str(k), "--save-logits", str(TEACHER_DIR),
                                   "--tag", teacher_tag_8b(k)] for k in todo]))
    assert all(c == 0 for c in codes), f"Qwen3-8B teacher fold failed: {codes}"
    merge(teacher_tag_8b, TEACHER_8B_FILE)
    # the step-3f student configuration with this teacher, three seeds (compare with the Qwen3-4B 3-seed mean)
    with ThreadPoolExecutor(3) as ex:
        codes = list(ex.map(run, [
            [*SEARCH_STUDENT, "--seed", sd, "--teacher", str(TEACHER_8B_FILE), "--save-logits", "models/study_logits",
             "--tag", f"distill: bge-small student of Qwen3-8B LoRA, mean pooling, lr 8e-5, alpha 0.1, seed {sd}"]
            for sd in ("42", "43", "44")]))
    print("student exit codes (Qwen3-8B teacher):", codes)


# best student configuration from rounds 1-3 (alpha 0.2, 4 epochs, 512 tokens), trained with the new teachers
FINAL_STUDENT = ["--lr", "5e-5", "--batch-size", "64", "--epochs", "4", "--max-tokens", "512", "--distill-alpha", "0.2"]


def teacher_4b_round():
    for k in range(K):                                          # one at a time (GPU memory)
        rc = run([*TEACHER_4B, "--n-folds", str(K), "--fold", str(k), "--save-logits", str(TEACHER_DIR),
                  "--tag", teacher_tag_4b(k)])
        assert rc == 0, f"Qwen3-4B teacher fold {k} failed"
    merge(teacher_tag_4b, TEACHER_4B_FILE)
    ensemble()
    with ThreadPoolExecutor(2) as ex:
        codes = list(ex.map(run, [
            [*FINAL_STUDENT, "--teacher", str(f), "--save-logits", "models/study_logits",
             "--tag", f"distill: bge-small student of {name}, alpha 0.2, 4 epochs, max_tokens 512"]
            for f, name in ((TEACHER_4B_FILE, "Qwen3-4B"), (TEACHER_ENS_FILE, "Qwen3-1.7B + Qwen3-4B (mean logits)"))]))
    print("student exit codes (new teachers):", codes)


def round2(rounds=None):
    def args(change, label):
        opts = dict(zip(STUDENT[::2], STUDENT[1::2])) | {"--distill-alpha": "0.2"}
        opts |= dict(zip(change[::2], change[1::2]))           # the one setting this run changes
        return [*[x for kv in opts.items() for x in kv], "--teacher", str(TEACHER_FILE),
                "--save-logits", "models/study_logits", "--tag", f"distill: bge-small student of Qwen3-1.7B, {label}"]
    for rnd in rounds or (ROUND2, ROUND3):
        with ThreadPoolExecutor(len(rnd)) as ex:
            codes = list(ex.map(run, [args(c, lab) for c, lab in rnd]))
        print("student round exit codes:", codes)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--students", action="store_true", help="skip the teacher folds")
    ap.add_argument("--round2", action="store_true", help="only the second and third student rounds")
    ap.add_argument("--round3", action="store_true", help="only the third student round")
    ap.add_argument("--teacher-8b", action="store_true",
                    help="only the Qwen3-8B LoRA teacher folds and three seeds of the step-3f student with it")
    ap.add_argument("--teacher-4b", action="store_true",
                    help="only the Qwen3-4B teacher folds, the ensemble teacher and their two students")
    a = ap.parse_args()
    if a.teacher_4b:
        teacher_4b_round()
        raise SystemExit(0)
    if a.teacher_8b:
        teacher_8b_round()
        raise SystemExit(0)
    if not (a.round2 or a.round3 or a.students):
        with ThreadPoolExecutor(TEACHER_PARALLEL) as ex:
            codes = list(ex.map(run, [[*TEACHER, "--n-folds", str(K), "--fold", str(k),
                                       "--save-logits", str(TEACHER_DIR), "--tag", teacher_tag(k)] for k in range(K)]))
        assert all(c == 0 for c in codes), f"teacher fold failed: {codes}"
    if a.round2 or a.round3:
        round2([ROUND3] if a.round3 else None)
        raise SystemExit(0)
    merge()
    with ThreadPoolExecutor(len(ALPHAS)) as ex:
        codes = list(ex.map(run, [[*STUDENT, "--teacher", str(TEACHER_FILE), "--distill-alpha", al,
                                   "--save-logits", "models/study_logits",
                                   "--tag", f"distill: bge-small student of Qwen3-1.7B, alpha {al}"] for al in ALPHAS]))
    print("student exit codes:", codes)
    round2()
