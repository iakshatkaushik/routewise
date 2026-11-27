# RouteWise: summary of the H100 fine-tuning and distillation work

This file explains what was done to improve
the RouteWise router on an NVIDIA H100, how it was done, what came out, and why. The numbers are filled in by a
script from the pipeline's report files (`reports/*.json`) and run logs. A few come from one-off checks made during
the session (marked *one-off check*), and one comes from the original project notes (marked); nothing is estimated.

## 1. The project in one paragraph

RouteWise (UML501 Machine Learning course project) reads a user prompt *before* any LLM answers it and
predicts whether a cheap LLM (Mixtral-8x7B in the data) will answer it badly. Risky prompts go to the strong
LLM (GPT-4 in the data). Data: the RouteLLM `gpt4_dataset`, 109,101 training and
10,000 held-out test prompts; label = 1 if GPT-4 graded the cheap answer 3 or lower on a 1-5
scale (13.5% of test prompts). Headline metric: **APGR**, the average share of the
cheap-to-strong quality gap the router recovers over all strong-model budgets (random = 0.500, perfect router on
this test set = 0.894). Also reported: ROC-AUC of the fail label, and CPT50 / CPT80,
the share of strong-model calls needed to recover 50% / 80% of the gap.

## 2. Ground rules that were kept

- **The test set decides nothing.** The validation split is fixed (`train_test_split(..., test_size=0.2,
  random_state=42, stratify=label)`, checked against `reports/val_scores.parquet` by every step). Every choice
  (hyperparameters, base model, checkpoint, components, thresholds, int8 vs fp32, teacher, alpha) was made on
  validation. Sweeps never compute test metrics (`03c_finetune.py --tune`).
- The test set was scored only for final configurations, once for each round that produced one (rounds 1-3;
  round 4 produced none), in sequence as the work progressed. Each final configuration was fixed on validation before its test run, and no round was kept
  or dropped because of its test score: the deployed router is always the last round's validation choice.
- Deployment limits: every model file < 95 MB; CPU-only serving with ONNX Runtime, NumPy and scikit-learn (no
  PyTorch); single-prompt latency on 1 thread about p50 ≤ 60 ms and p95 ≤ 250 ms; a light mode for 512 MB servers.
- Reproducible: every experiment is a script in `scripts/`, seed 42.

## 3. Results at a glance (test set, 10,000 prompts)

| Router | Components (full mode) | AUC | APGR | CPT50 | CPT80 |
|---|---|---|---|---|---|
| Before this work | Embedding Neural Network (GPU) + Logistic Regression | 0.7807 | 0.6913 | 25% | 55% |
| Round 1: fine-tuned bge-small | bge-small deficit head + Ridge regression | 0.7919 | 0.7051 | 23% | 53% |
| Round 2: distilled from Qwen3-1.7B | distilled bge-small deficit head | 0.8034 | 0.7149 | 22% | 51% |
| Round 3: Qwen3-4B teacher, searched student | distilled bge-small deficit head | 0.8022 | 0.7141 | 23% | 50% |
| **Deployed now** (the last round above) | Fine-tuned Transformer (GPU), score-deficit head (teacher: Qwen3-4B) | **0.8022** | **0.7141** | **23%** | **50%** |
| Light mode before / now | Logistic Regression / Ridge regression (predicted score) | 0.7680 / 0.7643 | 0.6827 / 0.6847 | 26% / 25% | 57% / 56% |

Validation (5-fold cross-validated stack): before AUC 0.7711 (APGR 0.6828, from the
original project notes); now **APGR 0.7098, AUC 0.7971**.

At a 30% strong-model budget the deployed router recovers 60.5% of the quality gap
(average score 4.77 / 5 vs 4.42 for the cheap model alone); at 50%
it recovers 80.1% (4.88 / 5). Asking for 30% strong traffic sent
29.9% of test prompts there.

The light router's AUC is lower than before (0.768 → 0.764) while its APGR, the selection
criterion, is slightly higher (0.6827 → 0.6847).

## 4. Environment

- One NVIDIA H100 80GB HBM3 (shared machine; GPU 0 via `CUDA_VISIBLE_DEVICES=0`). Python 3.13 venv outside the
  project. scikit-learn 1.9.1, numpy 2.5.5, scipy 1.18.1 exactly as pinned (the saved pickles need them).
- The pinned `torch==2.15.0` wheel is built for CUDA 13.2; this machine's driver supports CUDA 12.5, and no CUDA-12
  build of 2.15.0 exists, so training here used **torch 2.14.1+cu126**. `requirements.txt` still pins 2.15.0
  (with a comment) because serving needs no PyTorch and other machines use the standard wheel.
- Mixed precision: bf16 autocast + TF32 with fp32 master weights. bf16 did not hurt (validation AUC
  0.7820 bf16 vs 0.7818 fp32 for the same bge-small run).

## 5. Round 1: fine-tuning a small transformer

**What.** `src/llm_router/finetune.py` fine-tunes `BAAI/bge-small-en-v1.5` (33M parameters) end to end with two
heads on the [CLS] vector (mean pooling was added later and is used by the deployed model): head 0 = P(cheap model fails) (BCE on the label), head 1 = score deficit
(BCE on the soft target (5 − score) / 4). Length-bucketed batches, AdamW with warmup and linear decay, gradient
clipping, two validation checkpoints per epoch, best kept.

**Why two heads.** APGR rewards sending a score-1 prompt to the strong model more than a score-3 prompt; the binary
label cannot see that difference, the deficit head can. In every final selection the deficit head was chosen
over the fail head by validation APGR.

**Sweep** (`scripts/03c_sweep.py`, validation only, one or two changes at a time from the best so far):

| Run | Params | Best epoch | Val AUC (fail) | Val APGR (fail) | Val APGR (deficit) | Train min |
|---|---|---|---|---|---|---|
| baseline default (bf16) | - | 2.0 | 0.7820 | 0.6944 | 0.6966 | 2.6 |
| baseline default (fp32 check) | - | 2.0 | 0.7818 | 0.6934 | 0.6956 | 3.0 |
| r1: lr 5e-5 | - | 2.0 | 0.7837 | 0.6954 | 0.6982 | 3.7 |
| r1: lr 2e-5 | - | 2.0 | 0.7803 | 0.6923 | 0.6941 | 3.7 |
| r1: max_tokens 512 | - | 2.0 | 0.7830 | 0.6941 | 0.6964 | 3.9 |
| r1: epochs 3 | - | 2.0 | 0.7820 | 0.6939 | 0.6971 | 4.6 |
| r2: lr 5e-5, batch 64 | - | 2.0 | 0.7843 | 0.6959 | 0.6984 | 2.2 |
| r2: lr 5e-5, deficit_weight 2 | - | 2.0 | 0.7833 | 0.6959 | 0.6981 | 3.4 |
| r2: lr 5e-5, deficit_weight 0.5 | - | 2.0 | 0.7838 | 0.6950 | 0.6974 | 3.4 |
| r2: lr 5e-5, max_tokens 512 | - | 2.0 | 0.7847 | 0.6946 | 0.6981 | 3.5 |
| r3: noise check, lr 5e-5, batch 64, seed 43 | - | 2.0 | 0.7814 | 0.6940 | 0.6966 | 2.1 |
| r3: lr 5e-5, batch 64, max_tokens 512 | - | 2.0 | 0.7828 | 0.6955 | 0.6972 | 2.4 |
| r3: bge-base, lr 5e-5, batch 64 (validation-only; int8 file 109.7 MB > 95 MB) | - | 2.0 | 0.7839 | 0.6947 | 0.6976 | 2.5 |

Findings: lr 5e-5 and batch 64 were best; a third epoch overfits plain bge-small; the same configuration with seed 43
moved validation AUC from 0.7843 to 0.7814,
the same size as every difference in the sweep, so the top configurations are not clearly separated.

**ONNX export** (`scripts/07a_export_text_model.py`): int8 dynamic quantisation, served by `text_model.py` with
ONNX Runtime + `tokenizers`. Two problems were found and fixed:
- *The attention mask had to survive the export*: tracing with an all-ones mask can drop the mask path, so the
  export is traced with a padded example; fp32 ONNX matches PyTorch exactly (0.0 at 5 decimals), also in padded batches.
- *int8 outputs depend on batch padding*: dynamic quantisation scales activations per batch tensor. The server scores
  one prompt at a time, so all offline scores (the stacker's training inputs) are computed one prompt at a time too.
  Online and offline risks now agree to 5.3e-08.

**Ridge component** (`scripts/04_regression.py`): Ridge regression of the 1-5 score on TF-IDF features; its stacker
input is minus the predicted score. Rerunning step 4 reproduced the old `04_regression.json` to within 5e-5 on every
metric (*one-off check*).

**Router** (`scripts/07_build_router.py`): candidates carry an input kind (sparse, dense, embed, text) and an output
kind (proba, score, text_logit); every stacker input goes through the exact serving transform
(`Router._component_signal`). Greedy forward selection on 5-fold cross-validated validation APGR (AUC breaks ties),
add only if APGR improves by ≥ 0.001, at most 4 components. Round 1 picked the deficit head + Ridge.

## 6. Round 2: do bigger models route better? Then distillation

**Model-size study** (`scripts/03d_model_study.py`, validation only, one standard configuration per model):

| Run | Params | Best epoch | Val AUC (fail) | Val APGR (fail) | Val APGR (deficit) | Train min |
|---|---|---|---|---|---|---|
| study: bge-base, lr 3e-5, batch 32 | 110M | 1.0 | 0.7855 | 0.6944 | 0.6978 | 3.2 |
| study: bge-large, lr 2e-5, batch 32 | 335M | 1.0 | 0.7886 | 0.6956 | 0.7001 | 5.3 |
| study: ModernBERT-large, lr 2e-5, batch 32 | 395M | 1.0 | 0.7930 | 0.7011 | 0.7038 | 9.7 |
| study: deberta-v3-large, lr 1e-5, batch 32 (FAILED: weights loaded in fp16, no learning; fixed in the next run) | 434M | 0.5 | 0.5000 | 0.5001 | 0.5001 | 12.8 |
| study: deberta-v3-large, lr 1e-5, batch 32 (fp32 master weights) | 434M | 2.0 | 0.7953 | 0.7038 | 0.7056 | 14.4 |
| study: Qwen3-0.6B (decoder, last-token), lr 1e-5, batch 32 | 596M | 1.0 | 0.8015 | 0.7085 | 0.7106 | 16.5 |
| study: Qwen3-1.7B (decoder, last-token), lr 1e-5, batch 32, 1 epoch | 1721M | 1.0 | 0.8149 | 0.7169 | 0.7207 | 11.0 |
| study: Qwen3-4B (decoder, last-token), lr 1e-5, batch 32, 1 epoch, grad-ckpt | 4023M | 1.0 | 0.8207 | 0.7214 | 0.7246 | 23.1 |
| study: Qwen3-8B (decoder, last-token), LoRA r16, lr 1e-4, batch 32, 1 epoch, grad-ckpt | 7612M | 1.0 | 0.8301 | 0.7287 | 0.7310 | 52.1 |

Why bigger models win: the label depends on what the prompt asks (reasoning depth, domain knowledge, ambiguity,
instruction complexity), and larger pretrained models represent that better. Decoder LLMs (Qwen3) were the strongest
family. One bug surfaced here: **transformers 5 keeps a checkpoint's stored dtype by default**, so DeBERTa-v3-large
(stored in fp16) was trained in pure fp16 and collapsed (AUC 0.5000; the run is kept in the log, tagged FAILED), and
the bf16-stored Qwen3 models would have trained with bf16 weights. Fixed by loading fp32 master weights
(`from_pretrained(..., dtype=torch.float32)`); the affected runs were redone. bge and ModernBERT checkpoints are
stored in fp32 and were never affected (including the deployed model).

**Why not deploy a big model?** Size and speed: the int8 ONNX file of bge-base alone is 109.7 MB (*one-off check*:
exported and measured during the session; limit 95 MB), and billion-parameter models are far too slow for a CPU router.

**Knowledge distillation** (`scripts/03e_distill.py`): the big model teaches the small one.
1. Teacher: Qwen3-1.7B, fine-tuned 5 times, each time holding out one fold of the training part, so every
   training prompt gets a prediction from a model that never saw it (fold validation AUC 0.8095–0.8126,
   6.1–6.3 min each). The validation split is never part of the teacher file (asserted).
2. Student: bge-small trained on `alpha × label + (1 − alpha) × teacher probability` (both heads).

The table also lists the later Qwen3-4B and Qwen3-8B teachers and their students (section 7):

| Run | Params | Best epoch | Val AUC (fail) | Val APGR (fail) | Val APGR (deficit) | Train min |
|---|---|---|---|---|---|---|
| teacher: Qwen3-1.7B, lr 1e-5, batch 32, 1 epoch, fold 0 of 5 | 1721M | 1.0 | 0.8125 | 0.7145 | 0.7187 | 6.3 |
| teacher: Qwen3-1.7B, lr 1e-5, batch 32, 1 epoch, fold 1 of 5 | 1721M | 1.0 | 0.8100 | 0.7140 | 0.7176 | 6.1 |
| teacher: Qwen3-1.7B, lr 1e-5, batch 32, 1 epoch, fold 2 of 5 | 1721M | 1.0 | 0.8095 | 0.7130 | 0.7175 | 6.2 |
| teacher: Qwen3-1.7B, lr 1e-5, batch 32, 1 epoch, fold 3 of 5 | 1721M | 1.0 | 0.8118 | 0.7152 | 0.7198 | 6.2 |
| teacher: Qwen3-1.7B, lr 1e-5, batch 32, 1 epoch, fold 4 of 5 | 1721M | 1.0 | 0.8126 | 0.7149 | 0.7197 | 6.1 |
| teacher: Qwen3-4B, lr 1e-5, batch 32, 1 epoch, grad-ckpt, fold 0 of 5 | 4023M | 1.0 | 0.8177 | 0.7192 | 0.7228 | 14.6 |
| teacher: Qwen3-4B, lr 1e-5, batch 32, 1 epoch, grad-ckpt, fold 1 of 5 | 4023M | 1.0 | 0.8176 | 0.7195 | 0.7233 | 14.6 |
| teacher: Qwen3-4B, lr 1e-5, batch 32, 1 epoch, grad-ckpt, fold 2 of 5 | 4023M | 1.0 | 0.8172 | 0.7185 | 0.7231 | 14.4 |
| teacher: Qwen3-4B, lr 1e-5, batch 32, 1 epoch, grad-ckpt, fold 3 of 5 | 4023M | 1.0 | 0.8180 | 0.7200 | 0.7241 | 14.5 |
| teacher: Qwen3-4B, lr 1e-5, batch 32, 1 epoch, grad-ckpt, fold 4 of 5 | 4023M | 1.0 | 0.8181 | 0.7195 | 0.7233 | 14.3 |
| teacher: Qwen3-8B LoRA r16, lr 1e-4, batch 32, 1 epoch, grad-ckpt, fold 2 of 5 | 7612M | 1.0 | 0.8257 | 0.7250 | 0.7286 | 57.6 |
| teacher: Qwen3-8B LoRA r16, lr 1e-4, batch 32, 1 epoch, grad-ckpt, fold 1 of 5 | 7612M | 1.0 | 0.8264 | 0.7268 | 0.7291 | 38.6 |
| teacher: Qwen3-8B LoRA r16, lr 1e-4, batch 32, 1 epoch, grad-ckpt, fold 3 of 5 | 7612M | 1.0 | 0.8271 | 0.7256 | 0.7294 | 57.5 |
| teacher: Qwen3-8B LoRA r16, lr 1e-4, batch 32, 1 epoch, grad-ckpt, fold 0 of 5 | 7612M | 1.0 | 0.8271 | 0.7260 | 0.7290 | 57.5 |
| teacher: Qwen3-8B LoRA r16, lr 1e-4, batch 32, 1 epoch, grad-ckpt, fold 4 of 5 | 7612M | 1.0 | 0.8269 | 0.7254 | 0.7287 | 57.5 |
| distill: bge-small student of Qwen3-1.7B, alpha 0.0 | 33M | 2.0 | 0.7935 | 0.7018 | 0.7039 | 1.9 |
| distill: bge-small student of Qwen3-1.7B, alpha 0.5 | 33M | 2.0 | 0.7929 | 0.7020 | 0.7043 | 2.0 |
| distill: bge-small student of Qwen3-1.7B, alpha 0.2 | 33M | 2.0 | 0.7954 | 0.7033 | 0.7056 | 2.0 |
| distill: bge-small student of Qwen3-1.7B, alpha 0.2, seed 43 (noise check) | 33M | 2.0 | 0.7950 | 0.7031 | 0.7053 | 2.6 |
| distill: bge-small student of Qwen3-1.7B, alpha 0.1 | 33M | 2.0 | 0.7949 | 0.7028 | 0.7051 | 2.6 |
| distill: bge-small student of Qwen3-1.7B, alpha 0.2, max_tokens 512 | 33M | 2.0 | 0.7961 | 0.7037 | 0.7061 | 2.9 |
| distill: bge-small student of Qwen3-1.7B, alpha 0.2, 3 epochs | 33M | 3.0 | 0.7982 | 0.7054 | 0.7080 | 3.2 |
| distill: bge-small student of Qwen3-1.7B, alpha 0.2, 4 epochs | 33M | 4.0 | 0.7998 | 0.7064 | 0.7091 | 4.0 |
| distill: bge-small student of Qwen3-1.7B, alpha 0.2, 4 epochs, max_tokens 512 | 33M | 4.0 | 0.8008 | 0.7074 | 0.7101 | 4.3 |
| distill: bge-small student of Qwen3-1.7B, alpha 0.2, 5 epochs | 33M | 5.0 | 0.8005 | 0.7069 | 0.7094 | 4.5 |
| distill: bge-small student of Qwen3-1.7B + Qwen3-4B (mean logits), alpha 0.2, 4 epochs, max_tokens 512 | 33M | 4.0 | 0.8017 | 0.7083 | 0.7107 | 3.3 |
| distill: bge-small student of Qwen3-4B, alpha 0.2, 4 epochs, max_tokens 512 | 33M | 4.0 | 0.8019 | 0.7087 | 0.7108 | 3.3 |
| distill: bge-small student of Qwen3-8B LoRA, mean pooling, lr 8e-5, alpha 0.1, seed 44 | 33M | 4.0 | 0.8026 | 0.7085 | 0.7105 | 4.6 |
| distill: bge-small student of Qwen3-8B LoRA, mean pooling, lr 8e-5, alpha 0.1, seed 42 | 33M | 4.0 | 0.8021 | 0.7082 | 0.7102 | 4.7 |
| distill: bge-small student of Qwen3-8B LoRA, mean pooling, lr 8e-5, alpha 0.1, seed 43 | 33M | 3.5 | 0.8023 | 0.7077 | 0.7102 | 4.7 |

Why distillation works here: the labels are one noisy judgement per prompt (a single GPT-4 grade). The teacher's
probability is a smoothed estimate of how likely a prompt is to fail, so the student learns from a cleaner and richer
target. Evidence: the student's seed-to-seed change in validation AUC was
0.0004
against 0.0029 for plain bge-small,
and soft targets let it train longer without overfitting (best at 4 epochs instead of 2).

## 7. Rounds 3 and 4: stronger teachers and a wider student search

**Round 3, Qwen3-4B teacher.** Same 5-fold protocol (5 folds, fold validation AUC 0.8172–0.8181, 14.3–14.6 min each,
gradient checkpointing, about 74 GB of GPU memory as seen in `nvidia-smi`, one fold at a time). Its student
(round-2 configuration) beat the Qwen3-1.7B student by only about 0.001 validation AUC: a much better teacher gave a
small gain, so the student had become the bottleneck.

**Student search** (`scripts/03f_student_search.py`, validation only, Qwen3-4B teacher). Stage A changed one setting at a
time from the round-2 student, including other small base models and bigger students that still fit the 95 MB file
limit (bge-base cut to 6 or 8 layers, distilbert, distilroberta: 66.8-82.0 MB as int8, *one-off check*). Stage B
combined the winners. All search runs:

| Run | Params | Best epoch | Val AUC (fail) | Val APGR (fail) | Val APGR (deficit) | Train min |
|---|---|---|---|---|---|---|
| search: student of Qwen3-4B, batch 128 | 33M | 4.0 | 0.7985 | 0.7062 | 0.7079 | 8.9 |
| search: student of Qwen3-4B, lr 3e-5 | 33M | 4.0 | 0.7985 | 0.7065 | 0.7083 | 12.3 |
| search: student of Qwen3-4B, mean pooling | 33M | 4.0 | 0.8028 | 0.7090 | 0.7112 | 12.3 |
| search: student of Qwen3-4B, deficit weight 2 | 33M | 4.0 | 0.8020 | 0.7089 | 0.7111 | 12.4 |
| search: student of Qwen3-4B, deficit weight 0.5 | 33M | 4.0 | 0.8019 | 0.7086 | 0.7105 | 12.6 |
| search: student of Qwen3-4B, lr 8e-5 | 33M | 4.0 | 0.8025 | 0.7089 | 0.7113 | 12.6 |
| search: student of Qwen3-4B, lr 1e-4 | 33M | 4.0 | 0.8023 | 0.7087 | 0.7112 | 12.6 |
| search: student of Qwen3-4B, batch 32 | 33M | 4.0 | 0.8023 | 0.7092 | 0.7113 | 18.1 |
| search: student of Qwen3-4B, EMA 0.999 | 33M | 4.0 | 0.8016 | 0.7083 | 0.7105 | 11.4 |
| search: student of Qwen3-4B, alpha 0.3 | 33M | 4.0 | 0.8001 | 0.7076 | 0.7099 | 11.7 |
| search: student of Qwen3-4B, alpha 0.1 | 33M | 4.0 | 0.8025 | 0.7088 | 0.7107 | 11.7 |
| search: student of Qwen3-4B, base all-MiniLM-L12-v2 | 33M | 4.0 | 0.8015 | 0.7081 | 0.7101 | 11.8 |
| search: student of Qwen3-4B, base e5-small-v2 | 33M | 4.0 | 0.8012 | 0.7075 | 0.7096 | 11.8 |
| search: student of Qwen3-4B, base gte-small | 33M | 4.0 | 0.8033 | 0.7092 | 0.7111 | 11.8 |
| search: student of Qwen3-4B, base bge-base, first 6 layers | 67M | 4.0 | 0.7947 | 0.7034 | 0.7055 | 9.1 |
| search: student of Qwen3-4B, 6 epochs | 33M | 5.5 | 0.8024 | 0.7093 | 0.7113 | 15.8 |
| search: student of Qwen3-4B, base bge-base, first 8 layers | 81M | 4.0 | 0.7984 | 0.7059 | 0.7079 | 9.1 |
| search: student of Qwen3-4B, base distilbert-base-uncased | 66M | 4.0 | 0.7958 | 0.7035 | 0.7059 | 6.0 |
| search: student of Qwen3-4B, base distilroberta-base | 82M | 4.0 | 0.7997 | 0.7076 | 0.7097 | 6.1 |
| search: student of Qwen3-4B, reference (stage-A starting point) | 33M | 4.0 | 0.8019 | 0.7087 | 0.7108 | 6.3 |
| search: student of Qwen3-4B, combined: pooling mean, lr 8e-5, distill-alpha 0.1 | 33M | 4.0 | 0.8038 | 0.7096 | 0.7117 | 11.1 |
| search: student of Qwen3-4B, combined: base thenlper/gte-small, pooling mean | 33M | 4.0 | 0.8028 | 0.7087 | 0.7107 | 11.2 |
| search: student of Qwen3-4B, combined: base thenlper/gte-small, pooling mean, distill-alpha 0.1 | 33M | 4.0 | 0.8032 | 0.7090 | 0.7108 | 11.2 |
| search: student of Qwen3-4B, combined: base thenlper/gte-small, pooling mean, lr 8e-5 | 33M | 4.0 | 0.8026 | 0.7088 | 0.7108 | 11.2 |
| search: student of Qwen3-4B, combined: base thenlper/gte-small, pooling mean, epochs 6 | 33M | 5.0 | 0.8041 | 0.7096 | 0.7117 | 13.7 |
| search: student of Qwen3-4B, combined: base thenlper/gte-small, batch-size 32 | 33M | 4.0 | 0.8030 | 0.7091 | 0.7109 | 14.0 |
| search: student of Qwen3-4B, combined: start point, seed 44 | 33M | 4.0 | 0.8014 | 0.7083 | 0.7104 | 11.9 |
| search: student of Qwen3-4B, combined: pooling mean, lr 8e-5, distill-alpha 0.1, seed 43 | 33M | 4.0 | 0.8029 | 0.7087 | 0.7110 | 11.9 |
| search: student of Qwen3-4B, combined: start point, seed 43 | 33M | 4.0 | 0.8003 | 0.7076 | 0.7097 | 12.0 |
| search: student of Qwen3-4B, combined: pooling mean, lr 8e-5, distill-alpha 0.1, seed 44 | 33M | 4.0 | 0.8039 | 0.7093 | 0.7114 | 12.1 |
| search: student of Qwen3-4B, combined: base thenlper/gte-small, pooling mean, epochs 6, seed 43 | 33M | 4.0 | 0.8026 | 0.7080 | 0.7105 | 16.0 |
| search: student of Qwen3-4B, combined: base thenlper/gte-small, pooling mean, epochs 6, seed 44 | 33M | 5.5 | 0.8036 | 0.7092 | 0.7114 | 16.0 |
| search: student of Qwen3-4B, combined: base thenlper/gte-small, seed 44 | 33M | 4.0 | 0.8027 | 0.7087 | 0.7106 | 6.6 |
| search: student of Qwen3-4B, combined: base thenlper/gte-small, seed 43 | 33M | 4.0 | 0.8018 | 0.7080 | 0.7100 | 6.8 |
| search: student of Qwen3-4B, combined: base BAAI/bge-base-en-v1.5, num-layers 8, pooling mean, lr 3e-5, distill-alpha 0.1, epochs 6 | 81M | 4.5 | 0.7987 | 0.7054 | 0.7076 | 5.0 |
| search: student of Qwen3-8B LoRA, combined: pooling mean, lr 8e-5, distill-alpha 0.0 | 33M | 3.5 | 0.7998 | 0.7057 | 0.7076 | 6.3 |
| search: student of Qwen3-8B LoRA, combined: pooling mean, lr 8e-5, distill-alpha 0.2 | 33M | 4.0 | 0.8028 | 0.7089 | 0.7111 | 6.3 |
| search: student of Qwen3-8B LoRA, combined: pooling mean, lr 8e-5, distill-alpha 0.3 | 33M | 4.0 | 0.8007 | 0.7077 | 0.7099 | 6.4 |
| search: student of Qwen3-8B LoRA, combined: pooling mean, lr 8e-5, distill-alpha 0.5 | 33M | 2.0 | 0.7943 | 0.6988 | 0.7042 | 6.4 |

The bigger students did worse than bge-small at these settings (they were tried at the small model's settings only; a
bge-base-8-layer run at a lower learning rate was started but stopped unfinished to free GPU memory). Because the top differences were of the order of seed noise, the final student was chosen by the **mean of three
seeds**:

| Configuration | Seeds | Mean val AUC (fail) | ± (std) | Mean val APGR (fail) | Mean val APGR (deficit) |
|---|---|---|---|---|---|
| bge-small, mean pooling, lr 8e-5, alpha 0.1 (Qwen3-4B teacher) | 3 | 0.8035 | 0.0004 | 0.7092 | 0.7114 |
| gte-small, mean pooling, 6 epochs (Qwen3-4B teacher) | 3 | 0.8034 | 0.0006 | 0.7089 | 0.7112 |
| gte-small, CLS (Qwen3-4B teacher) | 3 | 0.8026 | 0.0006 | 0.7086 | 0.7106 |
| reference: bge-small, CLS, lr 5e-5, alpha 0.2 (Qwen3-4B teacher) | 3 | 0.8012 | 0.0007 | 0.7082 | 0.7103 |
| bge-small, mean pooling, lr 8e-5, alpha 0.1 (Qwen3-8B LoRA teacher) | 3 | 0.8023 | 0.0002 | 0.7081 | 0.7103 |

**Two students at once?** `scripts/03g_student_ensemble_check.py` stacked the deficit heads of every pair of the top-6
students exactly as step 7 would: the best pair gained +0.0008 cross-validated APGR over the best single
student, for about twice the text-model latency, so no ensemble is served.

**Round 4, Qwen3-8B teacher with LoRA.** A full fine-tune of 8B parameters does not fit in 80 GB, so the weights stay
frozen in bf16 and rank-16 LoRA adapters on every attention and MLP projection are trained (`--lora-r 16`; about 22 GB
per run, so three folds share the GPU). As a single study run it reached validation AUC
0.8301, the best of all models.
Its 5 fold runs reached 0.8257–0.8271 (38.6–57.6 min each, two or three at a time). The first attempts crashed with
out-of-memory errors in the prediction pass after training: `03c_finetune.py` predicted the saved logits with a
hard-coded batch of 512 (fine for bge-small, far too large for an 8B model sharing the GPU). After the fix (it now
uses `--eval-batch-size`) the missing folds were retrained; fold 0 retrained to the same validation AUC (0.8271) as
in the failed attempt. The chosen student configuration was then trained with this teacher for three seeds (last row of the seed table). Despite the much stronger teacher, its students were slightly **worse** on validation: three-seed mean AUC 0.8023 and deficit-head APGR 0.7103, against 0.8035 and 0.7114 with the Qwen3-4B teacher. Re-tuning alpha for the 8B teacher (0.0, 0.2, 0.3, 0.5, one seed each) did not close the gap (best 0.8028 AUC at alpha 0.2). The Qwen3-4B teacher was therefore kept, and round 4 produced no new final configuration, so the test set was not scored again. Interpretation: once the teacher is far ahead, the 33M-parameter student's capacity, not the teacher's quality, limits the result.

**Chosen for deployment (validation only):** teacher Qwen3-4B; BAAI/bge-small-en-v1.5, mean pooling,
alpha 0.1, 4 epochs, 512 tokens, lr 8e-05, batch 64, seed 42.
Final run: validation AUC 0.8037 (fail head), validation APGR
0.7116 (deficit head); 3.93 minutes of training.

## 8. The deployed router

- **Export:** int8 ONNX 33.9 MB (fp32 would be 133.1 MB, over the limit). int8 costs
  0.0021 validation AUC (0.8016 vs 0.8037); the rule was ≤ 0.003.
- **Component selection** (validation, 5-fold CV of the stack):
1. Best candidate **Fine-tuned Transformer (GPU), score-deficit head**: added (CV val APGR 0.7098, AUC 0.7971).
2. Best candidate **Ridge regression (predicted score)**: not added (gain +0.0004, below the 0.001 threshold).
  With the distilled transformer no other component (Ridge, logistic regression, embedding network, …) adds enough,
  so the full router is that one model plus a logistic-regression stacker that keeps the output a calibrated
  probability, P(cheap model fails).
- **Light mode** (512 MB servers): Ridge regression (predicted score) on TF-IDF features only; no ONNX model is loaded.
- **Latency, 1 CPU thread (Linux x86-64):** full p50 8.1 ms, p95 114.8 ms
  (300 test prompts; 188.7 ms p95 over 2,000); text model alone p50
  6.0 ms, p95 115.0 ms; light p50 1.22 ms, p95 2.84 ms.
- **Memory of the web server** (`scripts/10_server_memory.py`, 20 requests): full 368.8 MB (peak
  508.3 MB), light 254.8 MB (peak 262.6 MB). Full mode is too close
  to 512 MB, so `render.yaml` keeps the free plan on light mode.
- **Files:** `models/router.joblib` 14.8 MB, `models/text_router.onnx` 33.9 MB,
  `models/text_tokenizer.json`. The web UI and `/api/metrics` read every number from `reports/summary.json`.

## 9. How to reproduce

```bash
python -m venv venv && venv/bin/pip install -r requirements.txt   # on CUDA 12 drivers: torch from the cu126 index
venv/bin/python scripts/run_all.py            # full pipeline (needs a CUDA GPU for steps 3e and 3c)
venv/bin/python scripts/03c_sweep.py          # the bge-small hyperparameter sweep (validation only)
venv/bin/python scripts/03d_model_study.py    # the larger-model study (validation only)
venv/bin/python scripts/03e_distill.py --teacher-4b   # Qwen3-4B teacher, ensemble teacher, their students
venv/bin/python scripts/03f_student_search.py --teacher models/teacher/teacher_oof_qwen3_4b.npz --teacher-name Qwen3-4B --stage A
venv/bin/python scripts/03g_student_ensemble_check.py  # two-student ensemble check (validation)
venv/bin/python scripts/03e_distill.py --teacher-8b   # Qwen3-8B LoRA teacher (needs `peft`), 3 student seeds
venv/bin/python -m pytest -q
```
`scripts/run_all.py` lists the exact final `03c_finetune.py --final` arguments.

## 10. Lessons and pitfalls

1. With transformers 5, always pass `dtype=torch.float32` for training; checkpoints stored in fp16/bf16 otherwise
   train in that precision.
2. Dynamic int8 quantisation makes a prompt's output depend on what else is in its batch; compute offline scores
   exactly as production does (one prompt at a time here).
3. Export with a padded example so the attention mask is part of the ONNX graph.
4. Four large models at once ran out of GPU memory on 80 GB; two at a time works; Qwen3-1.7B (~62 GB) and
   Qwen3-4B (~74 GB, both as seen in `nvidia-smi`) need the GPU alone.
5. A seed-noise check is cheap and essential: in the plain sweep it showed that most differences were noise.
6. Out-of-fold teacher predictions are required for distillation; in-sample teacher predictions would leak the label.
7. When the remaining differences are about the size of seed noise, choose by the mean over several seeds, not by
   one lucky run. A better teacher helps less and less once the small student is the bottleneck.
8. LoRA makes an 8B-parameter teacher trainable on one 80 GB GPU (about 22 GB per run with gradient checkpointing).
9. Every code path needs the same memory settings: one prediction call with a hard-coded batch of 512 crashed the
   8B teacher folds after almost an hour of training each.

## 11. Limitations and possible next steps

- One standard configuration per model in the study; a per-model search could change the ranking.
- The fine-tuned model reads the first 512 tokens and is trained on the 80% training part (the validation
  part picks checkpoints and trains the stacker), not refitted on all data.
- Labels are single GPT-4 judgements of Mixtral answers; the deployed app routes between newer models, and the
  transfer has not been re-measured. The strong model's answers are treated as perfect (score 5).
- Not verified on this machine: a Docker build (no Docker) and a real-browser screenshot (jsdom rendering only).
- Next steps worth trying: larger students that still fit 95 MB, a per-model hyperparameter search for the teacher,
  and refitting the final student on all training rows with a fixed epoch count.

## 12. Where things are

| Path | What |
|---|---|
| `src/llm_router/finetune.py` | two-head fine-tuning (pooling cls/mean/last, bf16, EMA option, gradient checkpointing, LoRA, layer truncation, soft targets), ONNX export |
| `src/llm_router/text_model.py` | CPU inference with ONNX Runtime + tokenizers |
| `src/llm_router/router.py` | serving router; components can output proba, score or text_logit |
| `scripts/03c_finetune.py` | one fine-tuning run (`--tune` validation only, `--final` scores test and saves the model) |
| `scripts/03c_sweep.py`, `03d_model_study.py`, `03e_distill.py` | sweep, larger-model study, teachers and students |
| `scripts/03f_student_search.py`, `03g_student_ensemble_check.py` | student search (stages A/B, seeds), two-student ensemble check |
| `scripts/07a_export_text_model.py`, `07_build_router.py`, `08_summary.py`, `10_server_memory.py` | export, router, summary, memory |
| `reports/finetune_sweep.json` | every fine-tuning run (validation metrics, settings, time) |
| `reports/student_ensemble_check.json` | the two-student ensemble check |
| `reports/03c_finetune.json`, `07a_text_export.json`, `07_router.json`, `10_server_memory.json`, `summary.json` | final results |
| `tests/test_router_components.py`, `tests/test_api.py` | router component kinds, ONNX reference, full and light modes |
