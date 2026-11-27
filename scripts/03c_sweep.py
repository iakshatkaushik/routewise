"""Step 3c sweep - the validation-only fine-tuning sweep, in the order it was run.

Each run is `03c_finetune.py --tune ...`; results are appended to reports/finetune_sweep.json and the
test set is never scored. Runs inside a round are independent and share one GPU in parallel (a single
bge-small run does not fill an H100). Each round changed one or two settings from the best-so-far config.

  python scripts/03c_sweep.py            # run every round, then print the table
  python scripts/03c_sweep.py --table    # only print the table from reports/finetune_sweep.json
"""
import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
SWEEP = HERE.parent / "reports" / "finetune_sweep.json"

ROUNDS = [
    [  # baseline: the default configuration, in bf16 (used from here on) and in fp32 to check bf16 doesn't hurt
        ["--tag", "baseline default (bf16)"],
        ["--precision", "fp32", "--tag", "baseline default (fp32 check)"],
    ],
    [  # one change at a time from the baseline
        ["--epochs", "3", "--tag", "r1: epochs 3"],
        ["--lr", "5e-5", "--tag", "r1: lr 5e-5"],
        ["--lr", "2e-5", "--tag", "r1: lr 2e-5"],
        ["--max-tokens", "512", "--tag", "r1: max_tokens 512"],
    ],
    [  # from the round-1 best (lr 5e-5)
        ["--lr", "5e-5", "--deficit-weight", "0.5", "--tag", "r2: lr 5e-5, deficit_weight 0.5"],
        ["--lr", "5e-5", "--deficit-weight", "2", "--tag", "r2: lr 5e-5, deficit_weight 2"],
        ["--lr", "5e-5", "--batch-size", "64", "--tag", "r2: lr 5e-5, batch 64"],
        ["--lr", "5e-5", "--max-tokens", "512", "--tag", "r2: lr 5e-5, max_tokens 512"],
    ],
    [  # combine the two round-2 winners; larger base model; seed-noise check of the best config
        ["--lr", "5e-5", "--batch-size", "64", "--max-tokens", "512", "--tag", "r3: lr 5e-5, batch 64, max_tokens 512"],
        ["--base", "BAAI/bge-base-en-v1.5", "--lr", "5e-5", "--batch-size", "64",
         "--tag", "r3: bge-base, lr 5e-5, batch 64 (validation-only; int8 file 109.7 MB > 95 MB)"],
        ["--lr", "5e-5", "--batch-size", "64", "--seed", "43", "--tag", "r3: noise check, lr 5e-5, batch 64, seed 43"],
    ],
]


def table():
    runs = json.loads(SWEEP.read_text())["runs"]
    head = f"{'#':>2}  {'tag':72s} {'min':>5} {'ep*':>4} {'AUC fail':>8} {'AUC def':>8} {'APGR fail':>9} {'APGR def':>9}"
    print(head)
    print("-" * len(head))
    for i, r in enumerate(runs, 1):
        b = r["best_checkpoint"]
        print(f"{i:>2}  {r['tag'][:72]:72s} {r['train_minutes']:>5.1f} {b['epoch']:>4} {b['val_auc_fail']:>8.4f} "
              f"{b['val_auc_deficit']:>8.4f} {b['val_apgr_fail']:>9.4f} {b['val_apgr_deficit']:>9.4f}")


def run(args):
    cmd = [sys.executable, "-u", str(HERE / "03c_finetune.py"), "--tune", *args]
    print("running:", " ".join(cmd[2:]), flush=True)
    return subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT).returncode


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", action="store_true")
    if not ap.parse_args().table:
        for rnd in ROUNDS:
            with ThreadPoolExecutor(len(rnd)) as ex:
                assert all(rc == 0 for rc in ex.map(run, rnd)), "a sweep run failed"
    table()
