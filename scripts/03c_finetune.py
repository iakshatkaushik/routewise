"""Step 3c - fine-tune a small transformer as a router, on the GPU (ANN unit).

Same protocol as steps 3 and 3b: train on the training part of step 3's split, pick the checkpoint with
the best validation AUC of the fail head (evaluated twice per epoch).

Two modes:
  --tune   train and report validation metrics only. One record per run is appended to
           reports/finetune_sweep.json. The test set is never scored, and the score tables and
           03_classifiers.json are left alone. Used for the hyperparameter sweep.
  --final  the chosen configuration: also scores the 10,000 test prompts once, saves models/finetuned/
           (rebuilt by this step, not stored in git), appends both heads' predictions to step 3's
           score tables and writes reports/03c_finetune.json. Step 7a exports the model to ONNX.

Examples:
  python scripts/03c_finetune.py --tune --lr 3e-5 --epochs 2
  python scripts/03c_finetune.py --final --lr 3e-5 --epochs 3 --max-tokens 512

On an NVIDIA H100 a 2-epoch run of bge-small takes a few minutes (see reports/finetune_sweep.json).
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from _common import Timer, save_json
from llm_router import config, data, metrics
from llm_router.finetune import BASE_MODEL, MAX_TOKENS, predict_logits, train
from llm_router.nn import torch_device

NAME = "Fine-tuned Transformer (GPU)"
NAME_DEFICIT = "Fine-tuned Transformer (GPU), score-deficit head"
OUT = config.MODELS / "finetuned"
SWEEP = config.REPORTS / "finetune_sweep.json"


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--tune", action="store_true", help="validation only; append a record to finetune_sweep.json")
    mode.add_argument("--final", action="store_true", help="also score the test set and save models/finetuned/")
    ap.add_argument("--base", default=BASE_MODEL)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--deficit-weight", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--precision", choices=["bf16", "fp32"], default="bf16",
                    help="bf16: bf16 autocast + TF32 on CUDA; fp32: full fp32 (TF32 off)")
    ap.add_argument("--pooling", choices=["cls", "mean", "last"], default="cls")
    ap.add_argument("--ema", type=float, default=0.0, help="EMA decay of the weights (0 = off), e.g. 0.999")
    ap.add_argument("--save-logits", default="", help="--tune: directory to save validation (and out-of-fold) logits")
    ap.add_argument("--n-folds", type=int, default=0, help="--tune: K-fold mode over the training part (teacher)")
    ap.add_argument("--fold", type=int, default=-1, help="--tune: which fold to hold out (0..K-1)")
    ap.add_argument("--teacher", default="", help="npz with out-of-fold teacher logits for the training rows")
    ap.add_argument("--distill-alpha", type=float, default=0.5,
                    help="target = alpha * label + (1 - alpha) * teacher probability")
    ap.add_argument("--grad-ckpt", action="store_true", help="gradient checkpointing (for the largest models)")
    ap.add_argument("--eval-batch-size", type=int, default=512)
    ap.add_argument("--evals-per-epoch", type=int, default=2, help="validation checkpoints per epoch")
    ap.add_argument("--lora-r", type=int, default=0, help="LoRA rank (0 = full fine-tune); for the largest models")
    ap.add_argument("--num-layers", type=int, default=0, help="keep only the first N transformer layers (0 = all)")
    ap.add_argument("--tag", default="", help="free-text label stored with the sweep record")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    import torch
    from transformers import AutoTokenizer

    device = torch_device()
    amp = args.precision == "bf16" and device == "cuda"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = amp
        torch.backends.cudnn.allow_tf32 = amp

    train_df = data.load("train")
    y = train_df.label.values
    deficit = ((5 - train_df.score.values) / 4).astype(np.float32)
    idx_tr, idx_va = train_test_split(np.arange(len(y)), test_size=0.2, random_state=config.SEED, stratify=y)
    val_ref = pd.read_parquet(config.REPORTS / "val_scores.parquet")
    assert np.array_equal(idx_va, val_ref["row"].values), "validation split differs from step 3"

    tok = AutoTokenizer.from_pretrained(args.base)
    enc = lambda texts: tok(list(texts), truncation=True, max_length=args.max_tokens)["input_ids"]  # noqa: E731
    with Timer() as t_tok:
        enc_all = enc(train_df.prompt)
    rows_fit, rows_oof = idx_tr, None
    if args.n_folds:                                   # teacher mode: hold out one fold of the training part
        assert args.tune and 0 <= args.fold < args.n_folds
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(args.n_folds, shuffle=True, random_state=config.SEED)
        f_fit, f_oof = list(skf.split(idx_tr, y[idx_tr]))[args.fold]
        rows_fit, rows_oof = idx_tr[f_fit], idx_tr[f_oof]
    target_fail = y.astype(np.float32).copy()
    target_def = deficit.copy()
    if args.teacher:                                   # distillation: blend labels with teacher probabilities
        t = np.load(args.teacher)
        tp = np.full((len(y), 2), np.nan, dtype=np.float32)
        tp[t["rows"]] = 1 / (1 + np.exp(-t["logits"]))
        assert not np.isnan(tp[rows_fit]).any(), "teacher file does not cover every training row"
        assert not np.isin(t["rows"], idx_va).any(), "teacher file must not contain validation rows"
        a = args.distill_alpha
        target_fail[rows_fit] = a * target_fail[rows_fit] + (1 - a) * tp[rows_fit, 0]
        target_def[rows_fit] = a * target_def[rows_fit] + (1 - a) * tp[rows_fit, 1]
    enc_tr = [enc_all[i] for i in rows_fit]
    enc_va = [enc_all[i] for i in idx_va]
    print(f"device: {device} (precision {args.precision}); tokenised in {t_tok.s:.0f}s; "
          f"train {len(enc_tr)}, val {len(enc_va)}", flush=True)

    yv, sv = y[idx_va], train_df.score.values[idx_va]

    def val_metric(logits):
        r = {"select": roc_auc_score(yv, logits[:, 0]),
             "val_auc_fail": roc_auc_score(yv, logits[:, 0]),
             "val_auc_deficit": roc_auc_score(yv, logits[:, 1]),
             "val_apgr_fail": metrics.routing_report(logits[:, 0], sv)["apgr"],
             "val_apgr_deficit": metrics.routing_report(logits[:, 1], sv)["apgr"]}
        print("    " + "  ".join(f"{k} {v:.4f}" for k, v in r.items() if k != "select"), flush=True)
        return r

    hp = dict(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, weight_decay=0.01, warmup=0.06,
              deficit_weight=args.deficit_weight)
    with Timer() as t_fit:
        model, history = train(enc_tr, target_fail[rows_fit], target_def[rows_fit], enc_va, val_metric, device=device,
                               seed=args.seed, base=args.base, amp=amp, log=lambda m: print(m, flush=True),
                               pooling=args.pooling, ema=args.ema, grad_ckpt=args.grad_ckpt,
                               eval_batch_size=args.eval_batch_size, evals_per_epoch=args.evals_per_epoch,
                               lora_r=args.lora_r, num_layers=args.num_layers, **hp)
    if device == "cuda":
        torch.cuda.empty_cache()     # hand training-time cached memory back before predicting (runs share the GPU)
    best = max(history, key=lambda h: h["select"])
    record = {"tag": args.tag, "base": args.base, "max_tokens": args.max_tokens, "seed": args.seed,
              "precision": args.precision, **hp, "device": device, "pooling": args.pooling, "ema": args.ema,
              "grad_ckpt": args.grad_ckpt, "lora_r": args.lora_r, "num_layers": args.num_layers, "n_folds": args.n_folds, "fold": args.fold, "teacher": args.teacher or None,
              "distill_alpha": args.distill_alpha if args.teacher else None, "trained_on_rows": int(len(rows_fit)),
              "params_millions": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
              "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
              "train_minutes": round(t_fit.s / 60, 2),
              "best_checkpoint": {k: v for k, v in best.items() if k not in ("select", "best_so_far")},
              "history": history, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    print(f"==> best checkpoint: {record['best_checkpoint']}  ({record['train_minutes']} min)", flush=True)

    if args.tune and args.save_logits:
        import re
        from pathlib import Path
        out = Path(args.save_logits)
        out.mkdir(parents=True, exist_ok=True)
        stem = re.sub(r"[^a-z0-9]+", "_", (args.tag or "run").lower()).strip("_")
        np.savez(out / f"{stem}.npz", val_rows=idx_va, val_logits=predict_logits(model, enc_va, device, args.eval_batch_size, amp),
                 **({"oof_rows": rows_oof, "oof_logits": predict_logits(model, [enc_all[i] for i in rows_oof],
                                                                          device, args.eval_batch_size, amp)} if rows_oof is not None else {}))
        record["saved_logits"] = str(out / f"{stem}.npz")
        print(f"saved logits to {record['saved_logits']}", flush=True)

    if args.tune:
        # validation only: append to the sweep log, never touch the test set or the score tables
        import fcntl
        import tempfile
        with open(f"{tempfile.gettempdir()}/routewise_sweep.lock", "w") as lock:   # several runs may share one GPU in parallel
            fcntl.flock(lock, fcntl.LOCK_EX)
            sweep = json.loads(SWEEP.read_text()) if SWEEP.exists() else {"runs": []}
            sweep["note"] = ("Validation-only fine-tuning sweep (scripts/03c_finetune.py --tune). Checkpoint chosen "
                             "by validation AUC of the fail head; the test set is not scored in these runs.")
            sweep["runs"].append(record)
            SWEEP.write_text(json.dumps(sweep, indent=2))
        print(f"appended run to {SWEEP.relative_to(config.ROOT)}")
        raise SystemExit(0)

    # ---- --final: score the test set once with the chosen configuration ------------------
    test = data.load("test")
    enc_te = enc(test.prompt)
    lv = predict_logits(model, enc_va, device, args.eval_batch_size, amp)
    t = time.perf_counter()
    lt = predict_logits(model, enc_te, device, args.eval_batch_size, amp)
    pred_s = time.perf_counter() - t
    p_va, p_te = sigmoid(lv), sigmoid(lt)

    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUT / "weights.pt")
    tok.save_pretrained(OUT)                       # tokenizer of the chosen base model
    model.encoder.config.save_pretrained(OUT)      # encoder config of the chosen base model
    (OUT / "router_meta.json").write_text(json.dumps({"base": args.base, "max_tokens": args.max_tokens,
                                                      "hyperparameters": hp, "precision": args.precision,
                                                      "pooling": args.pooling, "ema": args.ema,
                                                      "teacher": args.teacher or None,
                                                      "distill_alpha": args.distill_alpha if args.teacher else None},
                                                     indent=2))

    c3_path = config.REPORTS / "03_classifiers.json"
    c3 = json.loads(c3_path.read_text())
    report = {}
    for name, col in ((NAME, 0), (NAME_DEFICIT, 1)):
        thr = float(np.quantile(p_va[:, col], 0.70))
        rep = metrics.classification_report(test.label.values, p_te[:, col], thr)
        rout = metrics.routing_report(p_te[:, col], test.score.values)
        va_auc = roc_auc_score(yv, p_va[:, col])
        va_apgr = metrics.routing_report(p_va[:, col], sv)["apgr"]
        print(f"==> {name}: val AUC={va_auc:.4f} val APGR={va_apgr:.4f}  test AUC={rep['roc_auc']:.4f}  "
              f"APGR={rout['apgr']:.4f}", flush=True)
        c3["models"][name] = {"best_params": {k: str(v) for k, v in hp.items()} | {
                                  "base": args.base, "max_tokens": str(args.max_tokens), "device": device,
                                  "precision": args.precision},
                              "val_auc": round(va_auc, 4), "test": rep, "routing": rout,
                              "fit_seconds": round(t_fit.s, 1),
                              "predict_ms_per_1k": round(1000 * pred_s / len(test) * 1000, 2),
                              "trained_on_rows": int(len(idx_tr)),
                              "input": f"raw prompt text (first {args.max_tokens} tokens), fine-tuned {args.base}"}
        report[name] = {"val_auc": round(va_auc, 4), "val_apgr": round(va_apgr, 4),
                        "test_auc": rep["roc_auc"], "test_apgr": rout["apgr"],
                        "test_cpt50": rout["cpt50"], "test_cpt80": rout["cpt80"]}
    c3_path.write_text(json.dumps(c3, indent=2))
    for fname, P in (("val_scores.parquet", p_va), ("test_scores.parquet", p_te)):
        df = pd.read_parquet(config.REPORTS / fname)
        df[NAME], df[NAME_DEFICIT] = P[:, 0], P[:, 1]
        df.to_parquet(config.REPORTS / fname)
    save_json("03c_finetune.json", {**{k: v for k, v in record.items() if k != "tag"},
                                    "hyperparameters": hp, "results": report,
                                    "note": "Probabilities in the score tables come from PyTorch here; step 7a "
                                            "overwrites them with the served ONNX model's outputs."})
