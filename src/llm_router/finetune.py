"""Fine-tuned transformer router (ANN unit, beyond the frozen-embedding network).

A small pretrained sentence encoder (BAAI/bge-small-en-v1.5, 33M parameters) is fine-tuned end to end
on the routing task with two output heads on the [CLS] vector:

- "fail":    P(cheap model's answer scores <= 3), trained with binary cross-entropy;
- "deficit": expected score shortfall (5 - score) / 4 in [0, 1], trained with BCE on that soft target.
  APGR rewards routing a score-1 prompt more than a score-3 one, and this head sees that difference.

Training runs on a GPU (NVIDIA CUDA; other backends are picked up automatically) with length-bucketed batches (prompts of similar
length share a batch, so little compute is wasted on padding). On CUDA it can use bf16 autocast and TF32
matrix maths for speed. Serving uses an ONNX export on the CPU.
"""
from __future__ import annotations

import math
import time

import numpy as np

BASE_MODEL = "BAAI/bge-small-en-v1.5"
MAX_TOKENS = 256


def _torch():
    import torch
    return torch


def build_model(base: str = BASE_MODEL, config_dir=None, pooling: str = "cls", lora_r: int = 0, num_layers: int = 0):
    """Two-head model on top of `base`. With `config_dir` (a saved fine-tune), the encoder is built from
    the saved config without downloading pretrained weights; load the fine-tuned state dict afterwards.
    pooling="cls" reads the first token's vector; "mean" averages the token vectors (padding excluded);
    "last" reads the last real token (for decoder-only language models).
    lora_r > 0: the encoder's weights stay frozen in bf16 and only low-rank LoRA adapters (rank lora_r) on
    its attention and MLP projections are trained, plus the head (for models too large to fine-tune fully).
    num_layers > 0 keeps only the first num_layers transformer layers of `base` (a smaller, faster student)."""
    torch = _torch()
    from transformers import AutoConfig, AutoModel

    class TwoHeadRouter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            if config_dir is not None:
                self.encoder = AutoModel.from_config(AutoConfig.from_pretrained(str(config_dir)))
            else:
                # fp32 master weights: transformers 5 otherwise keeps the checkpoint's stored dtype
                # (fp16 for deberta-v3-large, bf16 for Qwen3), and AdamW on fp16 weights diverges.
                # Speed comes from bf16 autocast in train().
                extra = {"num_hidden_layers": num_layers} if num_layers else {}
                self.encoder = AutoModel.from_pretrained(base, dtype=torch.bfloat16 if lora_r else torch.float32, **extra)
                if lora_r:
                    from peft import LoraConfig, get_peft_model
                    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
                    self.encoder = get_peft_model(self.encoder, LoraConfig(
                        r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05, target_modules=targets))
                    for n, p in self.encoder.named_parameters():      # adapters train in fp32
                        if p.requires_grad:
                            p.data = p.data.float()
            h = self.encoder.config.hidden_size
            self.dropout = torch.nn.Dropout(0.1)
            self.head = torch.nn.Linear(h, 2)     # column 0: fail logit, column 1: deficit logit

        def forward(self, input_ids, attention_mask):
            h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            if pooling == "mean":
                m = attention_mask.unsqueeze(-1).to(h.dtype)
                v = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
            elif pooling == "last":                  # decoder LLMs: last real token (right padding)
                last = attention_mask.sum(1).long() - 1
                v = h[torch.arange(h.shape[0], device=h.device), last]
            else:
                v = h[:, 0]
            if lora_r:                               # frozen bf16 encoder, fp32 head
                v = v.float()
            return self.head(self.dropout(v))

    return TwoHeadRouter()


def bucketed_batches(lengths, batch_size, rng=None):
    """Indices grouped into batches of similar length; batch order shuffled when rng is given."""
    order = np.argsort(lengths, kind="stable")
    batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    if rng is not None:
        rng.shuffle(batches)
    return batches


def collate(enc, ix, device):
    torch = _torch()
    L = max(len(enc[i]) for i in ix)
    ids = torch.zeros(len(ix), L, dtype=torch.long)
    att = torch.zeros(len(ix), L, dtype=torch.long)
    for r, i in enumerate(ix):
        n = len(enc[i])
        ids[r, :n] = torch.as_tensor(enc[i])
        att[r, :n] = 1
    return ids.to(device), att.to(device)


def _autocast(device, amp):
    torch = _torch()
    import contextlib
    if amp and str(device).startswith("cuda"):
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def predict_logits(model, enc, device, batch_size=128, amp=False) -> np.ndarray:
    torch = _torch()
    was_training = model.training
    model.eval()
    out = np.zeros((len(enc), 2), dtype=np.float32)
    with torch.no_grad(), _autocast(device, amp):
        for ix in bucketed_batches([len(e) for e in enc], batch_size):
            ids, att = collate(enc, ix, device)
            out[ix] = model(ids, att).float().cpu().numpy()
    model.train(was_training)
    return out


def load_finetuned(directory, device="cpu"):
    """Rebuild a fine-tuned model saved by step 3c (config.json + weights.pt) without downloading."""
    import json
    from pathlib import Path
    torch = _torch()
    meta_path = Path(directory) / "router_meta.json"
    pooling = json.loads(meta_path.read_text()).get("pooling", "cls") if meta_path.exists() else "cls"
    model = build_model(config_dir=directory, pooling=pooling)
    model.load_state_dict(torch.load(f"{directory}/weights.pt", map_location="cpu"))
    return model.to(device).eval()


def train(enc_tr, y_fail, y_deficit, enc_va, val_metric, *, device, epochs=2, batch_size=32, lr=3e-5,
          weight_decay=0.01, warmup=0.06, evals_per_epoch=2, deficit_weight=1.0, seed=42, log=print,
          base=BASE_MODEL, amp=False, eval_batch_size=512, pooling="cls", ema=0.0, grad_ckpt=False, lora_r=0,
          num_layers=0):
    """Fine-tune and keep the weights with the best validation metric.

    val_metric(logits) is evaluated `evals_per_epoch` times per epoch. It returns either a float (the
    selection score) or a dict with the selection score under "select" plus extra values to log.
    amp=True uses bf16 autocast on CUDA (weights and optimiser state stay fp32).
    y_fail / y_deficit may be soft targets in [0, 1] (used for distillation from a teacher).
    ema > 0 keeps an exponential moving average of the weights (decay `ema`); checkpoints are then
    evaluated and kept from the averaged weights.
    Returns (model with best weights, history list).
    """
    torch = _torch()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build_model(base, pooling=pooling, lora_r=lora_r, num_layers=num_layers).to(device)
    if grad_ckpt:                                    # recompute activations in backward: less memory, slower
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if lora_r:
            model.encoder.enable_input_require_grads()
    model.train()
    ema_model = None
    if ema > 0:
        import copy
        ema_model = copy.deepcopy(model).eval()
        ema_p = [p for p in ema_model.parameters()]
        live_p = [p for p in model.parameters()]
    no_decay = ("bias", "LayerNorm.weight", "layer_norm", "norm.weight")
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    params = [
        {"params": [p for n, p in trainable if not any(k in n for k in no_decay)], "weight_decay": weight_decay},
        {"params": [p for n, p in trainable if any(k in n for k in no_decay)], "weight_decay": 0.0},
    ]
    fused = str(device).startswith("cuda")
    opt = torch.optim.AdamW(params, lr=lr, **({"fused": True} if fused else {}))
    lengths = [len(e) for e in enc_tr]
    steps_per_epoch = math.ceil(len(enc_tr) / batch_size)
    total = steps_per_epoch * epochs
    warm = int(total * warmup)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(warm, 1) if s < warm else max(0.0, (total - s) / max(total - warm, 1)))
    yf = torch.as_tensor(y_fail, dtype=torch.float32)
    yd = torch.as_tensor(y_deficit, dtype=torch.float32)
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    eval_every = max(1, steps_per_epoch // evals_per_epoch)

    best, best_state, history, step, t0 = -1.0, None, [], 0, time.time()
    trainable_keys = {n for n, _ in trainable}      # with LoRA, only these change (frozen weights not copied)
    run_loss = 0.0
    for ep in range(epochs):
        for ix in bucketed_batches(lengths, batch_size, rng):
            ids, att = collate(enc_tr, ix, device)
            with _autocast(device, amp):
                logits = model(ids, att).float()
            loss = bce(logits[:, 0], yf[ix].to(device)) + deficit_weight * bce(logits[:, 1], yd[ix].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
            opt.step()
            sched.step()
            if ema_model is not None:
                with torch.no_grad():
                    torch._foreach_mul_(ema_p, ema)
                    torch._foreach_add_(ema_p, [p.detach() for p in live_p], alpha=1 - ema)
            step += 1
            run_loss += loss.detach()       # stays on the GPU; no sync every step
            if step % 200 == 0:
                log(f"step {step}/{total}  loss {float(run_loss) / 200:.4f}  {(time.time() - t0) / 60:.1f} min")
                run_loss = 0.0
            if step % eval_every == 0 or step == total:
                scored = ema_model if ema_model is not None else model
                res = val_metric(predict_logits(scored, enc_va, device, eval_batch_size, amp))
                extra = res if isinstance(res, dict) else {"select": res}
                score = float(extra["select"])
                history.append({"step": step, "epoch": round(step / steps_per_epoch, 2),
                                **{k: round(float(v), 4) for k, v in extra.items()},
                                "minutes": round((time.time() - t0) / 60, 2)})
                log(f"  eval {history[-1]}")
                if score > best:
                    best = score
                    history[-1]["best_so_far"] = True
                    best_state = {k: v.detach().to("cpu").clone() for k, v in scored.state_dict().items()
                                  if not lora_r or k in trainable_keys}
    model.load_state_dict(best_state, strict=not lora_r)
    return model, history


def export_onnx(model, path, quantize=True):
    """Export the two-head model to ONNX (CPU). With quantize=True the linear-layer weights are stored
    as int8 (dynamic quantisation), which makes the file about 4x smaller.
    Returns the path of the file to serve."""
    from pathlib import Path

    torch = _torch()
    path = Path(path)
    model = model.to("cpu").eval()
    ids = torch.randint(1000, 2000, (2, 17))
    att = torch.ones_like(ids)
    att[1, 11:] = 0          # trace with padding, so the attention-mask path is kept in the graph
    fp32 = path.with_name(path.stem + ".export-tmp.onnx")
    torch.onnx.export(model, (ids, att), str(fp32), input_names=["input_ids", "attention_mask"],
                      output_names=["logits"], opset_version=17, dynamo=False,
                      dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                                    "logits": {0: "batch"}})
    if not quantize:
        fp32.rename(path)
        return path
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(str(fp32), str(path), weight_type=QuantType.QInt8)
    fp32.unlink()
    return path
