# RouteWise: complete results summary

UML501 Machine Learning project. This file collects **every result** of the project in one place, with the reason
behind each one, for the report and the presentation.

**Where the numbers come from.** A script fills in all of them from the pipeline's report files
(`reports/*.json`), the run logs of the H100 session, and the saved test-score table. Nothing was typed by hand,
rounded up or estimated. The exceptions are marked:
- *(one-off check)*: a measurement made once during the session that has no report file.
- *(project notes)*: a number from the original project notes.
- *(reporting only)*: section 10's breakdowns of the test set, computed for this summary. They describe the final
  router; no decision was based on them.

Detailed per-run tables of the GPU work, and how to reproduce them, are in `H100_WORK.md`.

---

## 1. Headline results (for the first slide)

| | Original router | **Final router (deployed)** | Change |
|---|---|---|---|
| Test ROC-AUC (fail label) | 0.7807 | **0.8022** | +0.0215 |
| Test APGR (main metric) | 0.6913 | **0.7141** | +0.0228 |
| CPT50: strong calls needed for 50% of the gap | 25% | **23%** | -2 pts |
| CPT80: strong calls needed for 80% of the gap | 55% | **50%** | -5 pts |
| Gap recovered at a 30% strong budget | 56.9% | **60.5%** | +3.6 pts |
| Gap recovered at a 50% strong budget | 77.1% | **80.1%** | +3.0 pts |
| Single-prompt latency, 1 CPU thread, p50 / p95 | 16.3 / 159.7 ms | 8.1 / 114.8 ms | faster |

The three numbers for the slide:
1. **50% of strong-model calls recover 80% of the quality gap** (the original router needed
   55%, random routing needs 80%).
2. **APGR 0.714** against random 0.500 and a perfect oracle 0.894. The final router
   closes 11% of the distance between the original router and the oracle.
3. **One 33.9 MB int8 model on a CPU**, distilled from a 4-billion-parameter teacher trained on an H100.

**What the deployed router is.** A 33M-parameter transformer, `BAAI/bge-small-en-v1.5`, fine-tuned on an H100. It
learns from the labels and from a Qwen3-4B teacher (knowledge distillation). It is exported to int8 ONNX and runs on
the CPU with ONNX Runtime, followed by a logistic-regression stacker that outputs P(cheap model fails). A light mode
for 512 MB servers uses only Ridge regression on TF-IDF features (test AUC 0.7643, APGR 0.6847).

---

## 2. The problem, the data and the metrics

**Task.** Read a user prompt *before* any LLM answers it, and predict whether the cheap model will answer it badly.
Prompts predicted as risky go to the strong model; the rest go to the cheap one. The aim is the strong model's
quality at a fraction of its cost.

**Data.** The RouteLLM `gpt4_dataset`, with prompts from lmsys-chat, anthropic-hh, FLAN, evol-instruct and more. Each
prompt has an answer from Mixtral-8x7B (the cheap model), graded 1-5 by GPT-4 (the strong model).
- 109,101 training prompts and 10,000 held-out test prompts.
- After cleaning, no rows were dropped: 0 empty, 0 duplicates,
  0 overlapping with test.
- **Label = 1 ("fail")** if the grade is 3 or lower: 13.52% of training prompts and
  13.53% of test prompts. The classes are imbalanced, which is why accuracy is not the headline metric.
- Test grade distribution: 1: 492, 2: 432, 3: 429, 4: 1,716, 5: 6,931.

**Metrics.**
- **ROC-AUC**: how well the risk score ranks failing prompts above passing ones (0.5 = random, 1 = perfect).
- **APGR** (average performance-gap recovered), the headline metric. At each strong-model budget from 0% to 100%,
  send the riskiest prompts to the strong model. PGR is the share of the quality gap between "always cheap"
  (4.4162 / 5 on test) and "always strong" (5.0, treated as perfect) that this recovers.
  APGR is the average over all budgets.
  - Random routing scores 0.500.
  - A perfect oracle scores 0.894 on this test set; it needs only 20% strong calls
    for 80% of the gap. This is the ceiling.
- **CPT50 / CPT80**: the share of prompts that must go to the strong model to recover 50% / 80% of the gap. Lower
  is better; they translate directly into cost.

**Why APGR, not accuracy.** Routing is a ranking problem under a budget. A router is only useful if the prompts it
sends first are the ones that would have failed worst. APGR measures exactly that, and it rewards sending a grade-1
prompt before a grade-3 one, which a binary label can't see.

---

## 3. Evaluation protocol (why the numbers can be trusted)

- **Fixed validation split:** `train_test_split(test_size=0.2, random_state=42, stratify=label)` of the training set.
  Every step checks it against `reports/val_scores.parquet`.
- **Every choice was made on validation only:** hyperparameters, features, base model, checkpoint, which components
  to stack, int8 vs fp32, teacher, distillation weight. Hyperparameter sweeps never compute test metrics.
- **The test set was scored once per final configuration.** Three final configurations were built in sequence
  (rounds 1-3). Each was fixed on validation before its test run, and none was kept or dropped because of its test
  score. The deployed router is the last round's validation choice, **even though its test score is slightly below
  round 2's** (section 8).
- **Seed 42 everywhere.** Seed-noise checks (seeds 43, 44) were used to tell real gains from luck.
- **Online equals offline.** The served router reproduces the offline scores: maximum difference
  5.3e-08 over 2,000 prompts, with
  100% routing agreement at the 30% budget.

---

## 4. Classical machine-learning pipeline (the syllabus part)

### 4.1 Features (step 2)

TF-IDF word and character n-grams plus 24 prompt statistics (length, digits, punctuation, question words, code and
math markers and so on). Ablation with logistic regression, validation AUC:

| Feature set | Features | Val AUC |
|---|---|---|
| word_only | 60,025 | 0.7544 |
| char_only | 60,025 | 0.7466 |
| word+char | 120,024 | 0.7527 |
| word+char+stats | 120,024 | 0.7552 |
| small_vocab_20k | 40,024 | 0.7522 |
| large_vocab_150k | 300,024 | 0.7572 |

- **Chosen: `large_vocab_150k`** (300,024 features).
- Truncated SVD to 256 dimensions keeps 91.5% of the variance;
  it feeds the models that need dense input.
- **Reason:** words carry more signal than characters (topic and task type matter), and a larger vocabulary helps
  because the failure-prone tasks have specific vocabularies (translation, constrained answers, math).

**Sentence embeddings (step 2b).** BAAI/bge-small-en-v1.5 gives 384-dimensional embeddings, computed on the GPU
at 7.33 ms per prompt. The CPU ONNX copy used online matches it with a minimum cosine
similarity of 0.999998 over 1,000 prompts (step 9).

### 4.2 Classifiers (step 3): test results, sorted by APGR

All models were tuned on validation and scored once on test. The two GPU transformer rows are the later H100 work,
listed for comparison.

| Model | Best params (validation) | Val AUC | Test AUC | Test F1 | Test APGR | CPT50 | CPT80 | Fit s | Predict ms / 1k |
|---|---|---|---|---|---|---|---|---|---|
| Fine-tuned Transformer (GPU), score-deficit head | epochs=4, batch_size=64, lr=8e-05 | 0.7971 | 0.8022 | 0.426 | 0.7141 | 23% | 50% | 235.7 | 119.5 |
| Fine-tuned Transformer (GPU) | epochs=4, batch_size=64, lr=8e-05 | 0.8016 | 0.8054 | 0.426 | 0.7110 | 23% | 51% | 235.7 | 119.5 |
| Embedding Neural Network (GPU) | hidden=(512,), dropout=0.4, lr=0.0003, epochs=8 | 0.7600 | 0.7689 | 0.388 | 0.6838 | 26% | 55% | 1.1 | 1.2 |
| Linear SVM | C=0.03 | 0.7603 | 0.7686 | 0.395 | 0.6829 | 26% | 57% | 111.1 | 12.9 |
| Logistic Regression | C=0.25 | 0.7598 | 0.7680 | 0.397 | 0.6827 | 26% | 57% | 53.0 | 2.5 |
| Neural Network (MLP) | hidden_layer_sizes=(256,), alpha=0.001, activation=relu | 0.7514 | 0.7617 | 0.389 | 0.6771 | 27% | 58% | 10.7 | 1.2 |
| Gradient Boosting (beyond syllabus) | learning_rate=0.05, max_leaf_nodes=63 | 0.7462 | 0.7520 | 0.369 | 0.6699 | 28% | 59% | 55.0 | 15.4 |
| Random Forest | max_depth=None, min_samples_leaf=8, max_features=sqrt | 0.7472 | 0.7518 | 0.375 | 0.6691 | 27% | 59% | 139.5 | 13.0 |
| K-Nearest Neighbours | n_neighbors=150, weights=distance | 0.7257 | 0.7247 | 0.347 | 0.6518 | 30% | 62% | 0.0 | 365.2 |
| RBF-kernel SVM (20k sample) | C=0.5, gamma=scale | 0.7245 | 0.7176 | 0.359 | 0.6481 | 29% | 61% | 72.6 | 2806.2 |
| Decision Tree (Entropy / information gain) | max_depth=8, min_samples_leaf=200 | 0.7104 | 0.7153 | 0.348 | 0.6450 | 31% | 62% | 45.6 | 0.8 |
| Decision Tree (Gini) | max_depth=8, min_samples_leaf=200 | 0.7049 | 0.7100 | 0.343 | 0.6420 | 31% | 63% | 33.0 | 0.6 |
| Multinomial Naive Bayes | alpha=0.1 | 0.7161 | 0.7140 | 0.362 | 0.6419 | 30% | 67% | 0.9 | 12.6 |
| Complement Naive Bayes | alpha=0.1 | 0.7161 | 0.7140 | 0.362 | 0.6419 | 30% | 67% | 0.4 | 4.9 |

F1 uses the threshold chosen on validation to send 30% of prompts to the strong model. Rows are sorted by test APGR, but every choice above was made on
validation.

**Why the ranking looks like this:**
- **Linear models (logistic regression, linear SVM) are the best classical models.** With 300,024 sparse
  TF-IDF features the problem is high-dimensional and nearly linearly separable by vocabulary. L2-regularised linear
  models are made for this.
- **Tree, neighbour and neural models** get the 256-dimensional SVD representation (the full sparse matrix is too
  large for them), so they lose part of the fine vocabulary signal. Trees also split on one feature at a time, which
  suits dense, low-noise features better than many weak lexical cues.
- **Naive Bayes** assumes the words are independent given the label, which is wrong for phrases like "translate the
  sentence".
- **KNN and the RBF-SVM** suffer from distance concentration in high dimensions, and the RBF-SVM had to be trained on
  a 20k sample because kernel SVMs scale quadratically. KNN is also slow to predict
  (365 ms per 1k prompts).
- **The embedding neural network (GPU)** on bge-small embeddings is level with the best linear model
  (test AUC 0.7689). A *frozen* general-purpose
  embedding knows meaning but was not trained for this question.
- **The fine-tuned transformer** trains the whole network on the routing task. That is the single biggest jump in the
  project (section 6).

**Activation functions** (validation AUC):
- scikit-learn MLP: logistic 0.7401, tanh 0.7384, relu 0.7514.
- GPU embedding network: logistic 0.7397, tanh 0.7611, relu 0.7600, gelu 0.7594.

ReLU is best for the MLP and tanh for the GPU network (ReLU a close second). The logistic (sigmoid) activation needs
the most epochs (36 and 35) because it saturates, and it is the lowest on the GPU network. On the MLP it scores
just above tanh.

**GPU embedding-network grid** (validation; best: hidden (512,), dropout 0.4, lr 0.0003; test AUC 0.7689, APGR 0.6838):

| Hidden layers | Dropout | LR | Val AUC | Best epoch |
|---|---|---|---|---|
| (512,) | 0.2 | 0.001 | 0.7526 | 2 |
| (512,) | 0.2 | 0.0003 | 0.7573 | 4 |
| (512,) | 0.4 | 0.001 | 0.7536 | 3 |
| (512,) | 0.4 | 0.0003 | 0.7600 | 8 |
| (1024, 256) | 0.2 | 0.001 | 0.7542 | 4 |
| (1024, 256) | 0.2 | 0.0003 | 0.7519 | 3 |
| (1024, 256) | 0.4 | 0.001 | 0.7567 | 6 |
| (1024, 256) | 0.4 | 0.0003 | 0.7576 | 7 |

Dropout 0.4 beat 0.2 in all four pairs, and the lower learning rate won in three of four. With fixed embeddings the network overfits quickly, so
regularisation matters more than size.

### 4.3 Regression of the 1-5 grade (step 4)

| Model | RMSE | MAE | R² | Pearson r | Test APGR (route by predicted grade) | CPT50 | CPT80 |
|---|---|---|---|---|---|---|---|
| Simple Linear (log prompt length) | 1.069 | 0.789 | 0.030 | 0.174 | 0.5954 | 36% | 68% |
| Multiple Linear (24 prompt stats) | 1.051 | 0.774 | 0.063 | 0.252 | 0.6264 | 33% | 64% |
| Polynomial deg-2 (24 prompt stats) | 1.039 | 0.761 | 0.083 | 0.290 | 0.6405 | 30% | 63% |
| Ridge (L2), full features | 0.989 | 0.707 | 0.171 | 0.413 | 0.6847 | 25% | 56% |
| Lasso (L1), full features | 1.029 | 0.743 | 0.101 | 0.325 | 0.6496 | 29% | 63% |
| Elastic Net (L1+L2), full features | 1.026 | 0.746 | 0.107 | 0.328 | 0.6536 | 29% | 62% |

The baseline that always predicts the mean grade has RMSE 1.086.
- Simple linear regression: `score = 5.191 + -0.151 * log(1 + chars)`, so longer prompts get slightly
  lower grades.
- **Ridge is the best regressor and the best light-mode router** (APGR 0.6847,
  slightly above logistic regression's 0.6827).
- **Reason:** regressing the grade directly keeps the difference between a grade-1 and a grade-3 failure, which is
  what APGR rewards. Lasso and Elastic Net keep only 604 of
  300,024 features and lose too much vocabulary signal.
- **Why R² is low (0.171) but routing still works:** most prompts get
  grade 5, and the exact grade is noisy. Routing needs only the *ranking* to be right, not the exact grade.

### 4.4 Clustering (step 5)

K-means on the SVD features. k = 16 was chosen by silhouette over k = 6..30
(best silhouette 0.223 at k = 16). Silhouette values around 0.2 mean the prompt
topics overlap rather than forming clean groups. Clusters, sorted by training fail rate:

| Cluster | Top words | Train size | Train fail rate | Test fail rate | Main source |
|---|---|---|---|---|---|
| 15 | translate, sentence, translation, english | 2,243 | 37% | 35% | flan_v2_niv2 |
| 8 | say something, words, 50 words, answer when | 6,266 | 27% | 28% | lmsys-chat-1m |
| 4 | input, output, given, question | 5,410 | 22% | 18% | flan_v2_niv2 |
| 13 | step, it step, please eliminate, eliminate two | 2,236 | 22% | 20% | lmsys-chat-1m |
| 10 | question, answer, yes, options | 4,224 | 20% | 23% | flan_v2_cot |
| 11 | story, character, the story, essay | 3,455 | 19% | 20% | lmsys-chat-1m |
| 0 | assistant, system instructions, the assistant, completion | 7,661 | 18% | 18% | lmsys-chat-1m |
| 7 | code, return, https, int | 3,936 | 18% | 19% | lmsys-chat-1m |
| 9 | code, provide, python, program | 3,357 | 17% | 17% | evol_instruct |
| 14 | analyze, step, compare, analyze the | 3,027 | 17% | 19% | evol_instruct |
| 6 | task, behavior, sentence, option | 7,117 | 16% | 19% | lmsys-chat-1m |
| 12 | name_1, you know, is name_1, name_2 | 5,483 | 11% | 8% | lmsys-chat-1m |
| 2 | write, create, tell me, explain | 12,226 | 11% | 11% | lmsys-chat-1m |
| 3 | search for, hello, suggest, search | 8,112 | 9% | 9% | lmsys-chat-1m |
| 5 | want, want to, like to, trying | 8,390 | 8% | 8% | anthropic-hh |
| 1 | does, you tell, the best, how does | 25,958 | 6% | 6% | anthropic-hh |

- **Finding:** failure is concentrated by task type. Translation prompts fail
  37% of the time, and "say something … in 50 words" constrained prompts
  27%. General chat and advice prompts (anthropic-hh) fail only
  6%.
- Test fail rates track the training rates, so the clusters generalise.
- **Hierarchical clustering** (sample of 4,000): silhouette
  0.165 vs K-means 0.162, with an
  adjusted Rand index of 0.472 between them. They partly agree.
- **DBSCAN** labels 81.7% of prompts as noise. Density-based clustering does not
  suit this data, which has no dense, well-separated groups.

### 4.5 Association rules (step 6)

Apriori on 109,101 prompts described by 42 items (cluster, source, length bucket,
flags), with minimum support 0.003 and itemsets of up to 3 items.
- **Apriori found the same 1,288 frequent itemsets as a naive search** while counting
  2,557 candidates instead of 12,383
  (1.07 s vs 5.96 s). That is the downward-closure property at work.
- Base fail rate 13.5%. Top rules that predict failure:

| If | Support | Confidence | Lift |
|---|---|---|---|
| cluster=8 AND source=flan_v2_niv2 | 0.0040 | 0.517 | 3.82 |
| cluster=15 AND source=flan_v2_niv2 | 0.0061 | 0.448 | 3.31 |
| length=medium AND source=flan_v2_niv2 | 0.0045 | 0.416 | 3.08 |
| cluster=8 AND length=long | 0.0068 | 0.382 | 2.83 |
| cluster=15 | 0.0076 | 0.370 | 2.73 |
| cluster=15 AND length=long | 0.0039 | 0.366 | 2.70 |
| cluster=8 AND has_math | 0.0063 | 0.333 | 2.46 |
| has_math AND source=flan_v2_niv2 | 0.0085 | 0.332 | 2.45 |
| many_constraints AND source=flan_v2_niv2 | 0.0043 | 0.318 | 2.35 |
| source=flan_v2_niv2 | 0.0299 | 0.314 | 2.32 |

Rules that predict a good answer:

| If | Support | Confidence | Lift |
|---|---|---|---|
| cluster=1 AND source=anthropic-hh | 0.0938 | 0.967 | 1.12 |
| length=short AND source=anthropic-hh | 0.1159 | 0.965 | 1.12 |
| is_question AND source=anthropic-hh | 0.1380 | 0.965 | 1.12 |

**Reason:** FLAN (`flan_v2_niv2`) tasks are strict-format NLP tasks (translation, extraction, fixed options). The
cheap model often misses the exact format, and the grader penalises that. Open chat and advice prompts have no
single right answer, so the cheap model rarely fails them. These rules are human-readable evidence that the risk is
predictable from the prompt alone.

### 4.6 The original router (before the H100 work)

- **Components:** Embedding Neural Network (GPU) + Logistic Regression, stacked by logistic regression.
- **Test:** AUC 0.7807, APGR 0.6913, CPT50 25%, CPT80 55%.
- **Validation:** AUC 0.7711, APGR 0.6828 *(project notes)*.
- **Size and speed:** 16.87 MB; p50 16.33 ms and p95 159.68 ms.
- **Light mode:** Logistic Regression, test AUC 0.7680, APGR 0.6827.

---

## 5. Hardware and setup for the deep-learning part

- **GPU:** one NVIDIA H100 80GB HBM3 on a shared machine, using only GPU 0.
- **Software:** PyTorch 2.14.1+cu126 and transformers 5.24.1. bf16 autocast and TF32 with fp32 master weights; fused
  AdamW; batches bucketed by length.
- **bf16 cost no accuracy:** validation AUC 0.7820 in bf16 vs
  0.7818 in fp32 for the same run.
- **Logged training time:** 91 fine-tuning runs, 1032 GPU-minutes in total
  (17.2 h; some runs shared the GPU in parallel).
- **Serving needs no GPU or PyTorch:** ONNX Runtime, NumPy and scikit-learn on a CPU only.

---

## 6. Deep-learning results, round by round

### 6.1 Round 1: fine-tuning a small transformer

`bge-small-en-v1.5` (33M parameters) is fine-tuned end to end with **two output heads**:
- head 0: P(fail), trained with BCE on the label;
- head 1: the grade deficit, trained with BCE on the soft target (5 − grade) / 4.

A 13-run sweep (validation only) was run.
- **Best:** lr 5e-5, batch 64, with validation AUC 0.7843, against
  0.7600 for the frozen-embedding network.
- **Seed noise:** changing only the seed moved validation AUC by 0.0029, the same size as most sweep
  differences. Most "improvements" in a plain sweep are noise.
- **Router:** greedy selection picked the deficit head + Ridge. Test AUC 0.7919, APGR 0.7051.
- **Why fine-tuning beats frozen embeddings:** the network adapts every layer to the routing question instead of
  reusing a general-purpose sentence meaning.
- **Why the deficit head:** it ranks grade-1 prompts above grade-3 prompts, which is exactly what APGR rewards. In
  every final selection it beat the fail head on validation APGR.

### 6.2 Do bigger models route better? (model-size study, validation only)

| Run | Params | Val AUC | Val APGR (deficit head) | Train min |
|---|---|---|---|---|
| bge-small, round-1 best: lr 5e-5, batch 64 | 33M | 0.7843 | 0.6984 | 2.2 |
| study: bge-base, lr 3e-5, batch 32 | 110M | 0.7855 | 0.6978 | 3.2 |
| study: bge-large, lr 2e-5, batch 32 | 335M | 0.7886 | 0.7001 | 5.3 |
| study: ModernBERT-large, lr 2e-5, batch 32 | 395M | 0.7930 | 0.7038 | 9.7 |
| study: deberta-v3-large, lr 1e-5, batch 32 (fp32 master weights) | 434M | 0.7953 | 0.7056 | 14.4 |
| study: Qwen3-0.6B (decoder, last-token), lr 1e-5, batch 32 | 596M | 0.8015 | 0.7106 | 16.5 |
| study: Qwen3-1.7B (decoder, last-token), lr 1e-5, batch 32, 1 epoch | 1721M | 0.8149 | 0.7207 | 11.0 |
| study: Qwen3-4B (decoder, last-token), lr 1e-5, batch 32, 1 epoch, grad-ckpt | 4023M | 0.8207 | 0.7246 | 23.1 |
| study: Qwen3-8B (decoder, last-token), LoRA r16, lr 1e-4, batch 32, 1 epoch, grad-ckpt | 7612M | 0.8301 | 0.7310 | 52.1 |

- **Finding:** routing quality rises steadily with model size, from AUC 0.784 at 33M to
  0.830 at 8B.
- **Reason:** whether a cheap model fails depends on what the prompt really asks (reasoning depth, domain knowledge,
  hidden constraints, ambiguity). Larger pretrained models, especially decoder LLMs like Qwen3, which have seen
  instruction-style text, represent this much better.
- **Why not deploy them:** the 95 MB file limit and CPU latency. Even bge-base as int8 ONNX is 109.7 MB
  *(one-off check)*, and billion-parameter models are far too slow on one CPU thread.
- **A bug found along the way:** transformers 5 keeps a checkpoint's stored dtype, so DeBERTa-v3-large (stored in fp16)
  first trained in pure fp16 and collapsed to AUC 0.5000. Fixed by loading fp32 master weights and rerunning; the
  failed run is kept in the log, tagged FAILED.

### 6.3 Round 2: knowledge distillation (big teacher → small student)

1. **Teacher:** Qwen3-1.7B, trained 5 times. Each time one fold of the training part is held out, so every training
   prompt gets an *out-of-fold* prediction from a model that never saw it. Fold validation AUC was
   0.8095–0.8126, at 6.1–6.3 min per fold.
2. **Student:** bge-small trained on `alpha × label + (1 − alpha) × teacher probability` (both heads).

| Student of Qwen3-1.7B | Val AUC | Val APGR (deficit) | Best epoch |
|---|---|---|---|
| alpha 0.0 (teacher only) | 0.7935 | 0.7039 | 2.0 |
| alpha 0.1 | 0.7949 | 0.7051 | 2.0 |
| alpha 0.2 | 0.7954 | 0.7056 | 2.0 |
| alpha 0.5 | 0.7929 | 0.7043 | 2.0 |
| alpha 0.2, 3 epochs | 0.7982 | 0.7080 | 3.0 |
| alpha 0.2, 4 epochs | 0.7998 | 0.7091 | 4.0 |
| alpha 0.2, 5 epochs | 0.8005 | 0.7094 | 5.0 |
| **alpha 0.2, 4 epochs, 512 tokens (round 2 final)** | 0.8008 | 0.7101 | 4.0 |

- **Result:** the small model rose from 0.7843 to
  0.8008 validation AUC, at the same size.
- **Router test:** AUC 0.8034, APGR 0.7149, CPT50 22%, CPT80 51%.
- **Why distillation works here:** each label is a *single* GPT-4 grade of a *single* Mixtral answer, which makes it a
  noisy target. The teacher's probability is a smoothed estimate of how likely a prompt is to fail, so the student
  learns from a cleaner and richer signal. Evidence from the logs:
  - in one seed pair each, seed noise fell from 0.0029 (plain) to 0.0004 (distilled);
  - the student's best checkpoint is its last epoch, with up to 5 epochs, while plain bge-small trained for 3 epochs
    peaked at epoch 2 (it overfits);
  - alpha 0 (teacher only) is worse than a small amount of the real label, because the teacher is not perfect either.
- **Why out-of-fold:** in-sample teacher predictions would have memorised the labels and leaked them to the student.

### 6.4 Round 3: a stronger teacher and a wide student search

- **Teacher:** Qwen3-4B, same 5-fold protocol. Fold validation AUC 0.8172–0.8181, at
  14.3–14.6 min per fold, using about 74 GB of the GPU *(one-off check, nvidia-smi)*.
- **The student gained only about 0.001 AUC from the better teacher**, so the student had become the bottleneck. That
  led to a search over the student itself.
- **Stage A** changed one setting at a time from the reference (validation AUC 0.8019):

| Change (one at a time) | Val AUC | vs reference | Val APGR (deficit) |
|---|---|---|---|
| base gte-small | 0.8033 | +0.0014 | 0.7111 |
| mean pooling | 0.8028 | +0.0009 | 0.7112 |
| lr 8e-5 | 0.8025 | +0.0006 | 0.7113 |
| alpha 0.1 | 0.8025 | +0.0006 | 0.7107 |
| 6 epochs | 0.8024 | +0.0005 | 0.7113 |
| lr 1e-4 | 0.8023 | +0.0004 | 0.7112 |
| batch 32 | 0.8023 | +0.0004 | 0.7113 |
| deficit weight 2 | 0.8020 | +0.0001 | 0.7111 |
| deficit weight 0.5 | 0.8019 | +0.0000 | 0.7105 |
| EMA 0.999 | 0.8016 | -0.0003 | 0.7105 |
| base all-MiniLM-L12-v2 | 0.8015 | -0.0004 | 0.7101 |
| base e5-small-v2 | 0.8012 | -0.0007 | 0.7096 |
| alpha 0.3 | 0.8001 | -0.0018 | 0.7099 |
| base distilroberta-base | 0.7997 | -0.0022 | 0.7097 |
| batch 128 | 0.7985 | -0.0034 | 0.7079 |
| lr 3e-5 | 0.7985 | -0.0034 | 0.7083 |
| base bge-base, first 8 layers | 0.7984 | -0.0035 | 0.7079 |
| base distilbert-base-uncased | 0.7958 | -0.0061 | 0.7059 |
| base bge-base, first 6 layers | 0.7947 | -0.0072 | 0.7055 |

- **Stage B** combined the winners, then reran the top candidates with seeds 43 and 44. **The final choice is the
  3-seed mean**, because the top differences were the size of seed noise:

| Configuration | Seeds | Mean val AUC ± std | Mean val APGR (deficit) |
|---|---|---|---|
| **bge-small, mean pooling, lr 8e-5, alpha 0.1, Qwen3-4B teacher (chosen)** | 3 | 0.8035 ± 0.0004 | 0.7114 |
| gte-small, mean pooling, 6 epochs, Qwen3-4B teacher | 3 | 0.8034 ± 0.0006 | 0.7112 |
| gte-small, CLS pooling, Qwen3-4B teacher | 3 | 0.8026 ± 0.0006 | 0.7106 |
| reference: bge-small, CLS, lr 5e-5, alpha 0.2, Qwen3-4B teacher | 3 | 0.8012 ± 0.0007 | 0.7103 |
| bge-small, mean pooling, lr 8e-5, alpha 0.1, **Qwen3-8B LoRA teacher** | 3 | 0.8023 ± 0.0002 | 0.7103 |

- **Likely reasons mean pooling and a higher lr helped (not tested separately):** averaging all token vectors uses the whole prompt, not just the [CLS]
  summary. A higher learning rate suits the softer, less noisy distillation target.
- **Why bigger students did worse:** they were tried at the small model's settings only, and cutting bge-base to 6 or 8
  layers throws away pretrained structure. Either reason may explain it; this was not tested further.
- **Two students at once** (`03g`): stacking the best pair of the top-6 students gained only
  +0.0008 cross-validated APGR, for about twice the latency, so it is not served.
- **Router test:** AUC 0.8022, APGR 0.7141, CPT50 23%, CPT80 50%.
  This is the deployed router.

### 6.5 Round 4: the strongest teacher (Qwen3-8B with LoRA)

- A full 8B fine-tune does not fit in 80 GB. The base weights stay frozen in bf16, and rank-16 LoRA adapters on every
  attention and MLP projection are trained, using about 22 GB per run.
- **Best model of the whole project:** validation AUC
  0.8301.
  Fold models 0.8257–0.8271.
- **Its student was slightly worse:** 3-seed mean 0.8023 AUC / 0.7103 APGR, against 0.8035 /
  0.7114 with the Qwen3-4B teacher.
- Retuning alpha for this teacher (0.0, 0.2, 0.3, 0.5) did not close the gap (best 0.8028).
- **Decision (validation):** keep the Qwen3-4B teacher. Round 4 produced no new final configuration, so the test set was
  not scored again.
- **Reason:** a 33M-parameter student can absorb only so much. Once the teacher is far ahead, the student's capacity,
  not the teacher's quality, limits the result. A possible further reason is a gap in *how* they see the text: the decoder LLM reads
  the prompt left to right, while the student is a bidirectional encoder with a 512-token limit.

---

## 7. The final router in detail

### 7.1 Model and export

- **Base and settings:** BAAI/bge-small-en-v1.5, mean pooling, 4 epochs, 512 tokens,
  lr 8e-05, batch 64, alpha 0.1, Qwen3-4B teacher.
- **Training:** 87,280 rows, 33.4M parameters, 3.93 min on the H100.
- **Final run (PyTorch):** validation AUC 0.8037 (fail head) and APGR
  0.7116 (deficit head).
- **Export:**

| Variant | File | Val AUC (fail head) | Val APGR (deficit head) |
|---|---|---|---|
| fp32 ONNX | 133.12 MB (over the 95 MB limit) | 0.8037 | 0.7116 |
| **int8 ONNX (served)** | **33.87 MB** | 0.8016 | 0.7099 |

- **int8 loses 0.0021 AUC**, within the validation rule of at most 0.003, for a 4× smaller
  file. fp32 ONNX matches PyTorch exactly.
- **Two export pitfalls, both fixed:**
  - tracing with an all-ones mask can drop the attention mask from the graph, so the model is traced with a padded
    example;
  - int8 outputs depend on batch padding, so all offline scores are computed one prompt at a time, exactly as the
    server does.

### 7.2 Component selection (validation, 5-fold cross-validated stack)

Rule: greedy forward selection on 5-fold cross-validated stacked validation APGR (AUC breaks ties); add only if APGR improves by >= 0.001; at most 4. Step 1 candidates:

| Candidate | CV val APGR | CV val AUC |
|---|---|---|
| Fine-tuned Transformer (GPU), score-deficit head | 0.7098 | 0.7971 |
| Fine-tuned Transformer (GPU) | 0.7068 | 0.8015 |
| Ridge regression (predicted score) | 0.6804 | 0.7589 |
| Embedding Neural Network (GPU) | 0.6753 | 0.7600 |
| Logistic Regression | 0.6750 | 0.7597 |
| Neural Network (MLP) | 0.6684 | 0.7513 |
| Gradient Boosting (beyond syllabus) | 0.6642 | 0.7462 |
| Decision Tree (Entropy / information gain) | 0.6383 | 0.7098 |

- **Step 2:** the best addition, Ridge regression (predicted score), gained only +0.0004, below the 0.001 threshold.
  **The final router is one model + the stacker.**
- **Reason:** the distilled transformer already contains what the TF-IDF models know, and more. In round 1 Ridge was
  still worth adding; after distillation it is not.
- **Light mode:** Ridge regression (predicted score). Adding logistic regression lowered CV APGR by
  0.0008.

### 7.3 Test results (scored once)

| Metric | Original router | Round 1 | Round 2 | **Final (round 3)** | Light mode (Ridge) | Oracle |
|---|---|---|---|---|---|---|
| Test AUC | 0.7807 | 0.7919 | 0.8034 | **0.8022** | 0.7643 | - |
| Test APGR | 0.6913 | 0.7051 | 0.7149 | **0.7141** | 0.6847 | 0.8942 |
| CPT50 | 25% | 23% | 22% | **23%** | 25% | 9% |
| CPT80 | 55% | 53% | 51% | **50%** | 56% | 20% |
| Validation CV APGR | 0.6828 *(project notes)* | 0.6991 | 0.7085 | **0.7098** | 0.6804 | - |
| Validation CV AUC | 0.7711 *(project notes)* | 0.7838 | 0.7950 | **0.7971** | 0.7589 | - |

**Quality at fixed strong-model budgets** (test; average grade out of 5, share of the gap recovered):

| Strong budget | Original router | **Final router** | Oracle |
|---|---|---|---|
| 0% (always cheap) | 4.42 | 4.42 | 4.42 |
| 20% | 4.67 (43%) | **4.69 (47%)** | 4.89 (82%) |
| 30% | 4.75 (57%) | **4.77 (60%)** | 4.99 (99%) |
| 50% | 4.87 (77%) | **4.88 (80%)** | 5.00 (100%) |
| 100% (always strong) | 5.00 | 5.00 | 5.00 |

**Classification view at the 30% budget threshold** (test, 13.53% positives):

| | Original router | **Final router** |
|---|---|---|
| Accuracy | 0.7455 | 0.7508 |
| Precision | 0.2967 | 0.3093 |
| Recall (sensitivity) | 0.6430 | 0.6829 |
| Specificity | 0.7615 | 0.7614 |
| F1 | 0.4061 | 0.4258 |
| Confusion (TN / FP / FN / TP) | 6585 / 2062 / 483 / 870 | 6584 / 2063 / 429 / 924 |

- At the same budget the final router **catches 54 more failing prompts** (924 vs 870)
  and misses 54 fewer.
- Precision looks low (0.31) only because 30% of prompts are sent while just 13.5% fail.
  That is by design: the budget, not the threshold, decides how many go to the strong model.

**The budget is honoured:** asking for 30% strong traffic sent 29.9% of test prompts,
and asking for 50% sent 49.9%.

### 7.4 Deployment numbers

| | Full mode | Light mode | Limit |
|---|---|---|---|
| Model files | router.joblib 14.83 MB + text_router.onnx 33.87 MB + tokenizer | router.joblib 14.83 MB | each < 95 MB |
| Latency p50 / p95, 1 thread (300 prompts) | 8.07 / 114.81 ms | 1.22 / 2.84 ms | ≤ 60 / ≤ 250 ms |
| Latency p50 / p95 (2,000 prompts) | 8.63 / 188.7 ms | 1.3 / 3.41 ms | |
| Web-server memory (RSS / peak) | 368.8 / 508.3 MB | 254.8 / 262.6 MB | 512 MB servers |

- **Latency:** the gap between p50 and p95 is expected from prompt length, because the transformer's cost grows with
  the number of tokens (up to 512). This was not measured per length.
- **Memory:** full mode peaks close to 512 MB, so the free Render plan runs light mode (`render.yaml`). Full mode is
  for servers with more memory.
- **Measurement conditions:** latency was measured on a shared Linux machine while a GPU job was running.

---

## 8. The honest caveat: round 3 vs round 2 on test

- **Validation:** round 3 is better than round 2 (CV APGR 0.7098 vs 0.7085; AUC
  0.7971 vs 0.7950).
- **Test:** round 3 is slightly lower (APGR 0.7141 vs 0.7149; AUC 0.8022 vs 0.8034).
- **The gap is within noise:** 0.0008 APGR, against seed-to-seed differences of about 0.001 on
  validation.
- **Why round 3 is still deployed:** the rule is that the test set decides nothing. Switching back to round 2 because
  it looks better on test would mean selecting on test, and the reported number would no longer be an unbiased
  estimate.
- For the report: *"Each round was chosen on validation; the final router is the last validation choice. Its test
  score (APGR 0.7141) is slightly below the previous round's (0.7149), a difference within seed
  noise."*

---

## 9. Why the results came out this way (summary of reasons)

1. **The task is predictable from the prompt alone, but only partly.** Clusters and association rules show failure
   concentrated in strict-format tasks (translation, constrained answers, math, FLAN tasks). Yet even the best model
   (Qwen3-8B) reaches only validation AUC 0.830. Most likely, the rest depends on the cheap model's random sampling and on one noisy grade per
   prompt.
2. **Linear models on TF-IDF are a strong classical baseline** because the signal is largely lexical (task words).
3. **Fine-tuning beats frozen embeddings** because every layer adapts to the routing question.
4. **Bigger pretrained models understand prompts better** (AUC 0.784 → 0.830 from 33M to
   8B), but they can't be served on a CPU within 95 MB.
5. **Distillation transfers most of that gain to a small model**, because the teacher's soft probabilities are less
   noisy than single grades.
6. **Returns diminish:** a better teacher stops helping once the 33M student is the bottleneck (the 8B teacher's
   student was worse than the 4B teacher's).
7. **Grade-aware training (the deficit head, Ridge on the grade) consistently helps APGR**, because APGR rewards
   catching the worst failures first.
8. **Stacking stops paying off** once one strong model dominates. Every extra component gained less than 0.001 APGR.

---

## 10. Where the final router is strong and weak (test breakdown, reporting only)

*Computed for this summary from the saved test scores of the deployed router (which reproduce its test AUC
0.8022 and APGR 0.7141 exactly). Logistic regression, the best classical model, is shown for
comparison. Nothing was chosen on these numbers.*

**By data source:**

| Source | Test prompts | Fail rate | AUC, logistic regression | AUC, final router | Gain |
|---|---|---|---|---|---|
| lmsys-chat-1m | 3,144 | 15% | 0.761 | 0.788 | +0.027 |
| sharegpt | 2,027 | 10% | 0.706 | 0.790 | +0.083 |
| anthropic-hh | 1,594 | 3% | 0.720 | 0.733 | +0.013 |
| flan_v2_niv2 | 991 | 30% | 0.726 | 0.772 | +0.047 |
| evol_instruct | 809 | 16% | 0.682 | 0.762 | +0.081 |
| ultrachat | 803 | 11% | 0.746 | 0.782 | +0.036 |
| flan_v2_cot | 231 | 26% | 0.671 | 0.691 | +0.020 |
| false_qa | 193 | 15% | 0.467 | 0.475 | +0.008 |
| flan_v2_p3 | 111 | 23% | 0.550 | 0.592 | +0.042 |
| truthful_qa | 62 | 18% | 0.554 | 0.836 | +0.282 |
| flan_v2_flan2021 | 35 | 11% | 0.694 | 0.702 | +0.008 |

**By prompt length:**

| Prompt length | Test prompts | Fail rate | AUC, logistic regression | AUC, final router | Gain |
|---|---|---|---|---|---|
| < 100 chars | 4,106 | 8% | 0.741 | 0.794 | +0.052 |
| 100-299 | 2,323 | 13% | 0.780 | 0.799 | +0.019 |
| 300-999 | 2,385 | 21% | 0.707 | 0.756 | +0.049 |
| 1,000-2,999 | 1,167 | 20% | 0.718 | 0.767 | +0.049 |
| 3,000+ | 19 | 11% | 0.441 | 0.382 | -0.059 |

**Predicted risk by the true grade** (a good router gives the worst grades the highest risk):

| True grade | Test prompts | Mean predicted risk | Median risk | Sent to strong at the 30% budget |
|---|---|---|---|---|
| 1 | 492 | 0.345 | 0.276 | 80% |
| 2 | 432 | 0.268 | 0.214 | 69% |
| 3 | 429 | 0.198 | 0.168 | 59% |
| 4 | 1,716 | 0.147 | 0.118 | 37% |
| 5 | 6,931 | 0.105 | 0.076 | 22% |

Within a single source, AUC is usually lower than overall, because part of the overall signal is *which kind of
prompt* it is (for example, FLAN vs chat). That kind is easy to tell apart, and the breakdown removes it.
- **Strong:** the final router beats logistic regression in every source with more than 200 prompts. The biggest gains
  are on sharegpt and evol_instruct (about +0.08), likely because there the risk depends on meaning rather than keywords.
- **Weak:** `false_qa` (questions with a false premise) is near random for both models. Whether the cheap model handles
  a false premise well is probably not visible from the wording.
- **Read small groups with care:** the groups with fewer than 200 prompts (flan_v2_p3, truthful_qa, flan_v2_flan2021,
  3,000+ characters) are too small for stable AUCs.
- **The grade table shows the ranking works:** the mean risk falls steadily from grade 1 to grade 5. At the 30% budget,
  80% of grade-1 prompts go to the strong model, but only 22% of grade-5 prompts.

---

## 11. Limitations

- Labels are single GPT-4 judgements of Mixtral answers. The deployed app routes between newer models, and that
  transfer was not measured. The strong model is treated as perfect (grade 5).
- The final student reads only the first 512 tokens. It is trained on the 80% training part, because the validation
  part picks checkpoints and trains the stacker; it was not refitted on all data.
- The model study used one standard configuration per model. Bigger students were tried only at the small model's
  settings.
- The test set was scored for three final configurations in sequence (each fixed on validation first).
- Latency was measured on a shared Linux server, not on the deployment host. A Docker build and a real-browser check
  were not run on the training machine.

---

## 12. Slide-by-slide outline

1. **Problem:** strong LLMs are expensive; most prompts don't need them. Route each prompt before answering.
2. **Data:** 109,101 + 10,000 prompts, GPT-4 grades of Mixtral answers,
   13.5% fail.
3. **Metric:** APGR and the routing curve (random 0.50, oracle 0.89). Figure:
   `reports/figures/routing_curve.png`.
4. **Classical ML:** 11 classical models compared; linear models best (test APGR
   0.683). Figure: `reports/figures/model_comparison.png`.
5. **Unsupervised insight:** clusters and association rules show where the cheap model fails (translation
   37%, chat 6%). Figures:
   `cluster_failure_rates.png`, `kmeans_selection.png`.
6. **Fine-tuning on an H100:** bge-small end to end, two heads.
7. **Scaling study:** 33M → 8B parameters, AUC 0.784 → 0.830.
8. **Distillation:** a big teacher with out-of-fold predictions and a small student; why soft targets help.
9. **Student search and seed-mean selection;** the 8B teacher didn't help (diminishing returns).
10. **Final router:** int8 ONNX 33.9 MB, CPU only; the selection log shows one model is enough.
11. **Results:** test APGR 0.691 → 0.714, CPT80 55% → 50%, ROC
    figure `reports/figures/roc_curves.png`.
12. **Honesty:** validation-only choices; round 3 vs round 2 on test; limitations.

Figures in `reports/figures/`: `routing_curve.png`, `roc_curves.png`, `model_comparison.png`,
`cluster_failure_rates.png` and `kmeans_selection.png`. The web app's results page (`/results`) has an interactive
"Do bigger models route better?" chart.

---

## 13. Reproducing the results

```bash
python scripts/run_all.py                    # the full pipeline (steps 3e and 3c need a CUDA GPU)
python scripts/03c_sweep.py                  # round-1 sweep (validation only)
python scripts/03d_model_study.py            # model-size study (validation only)
python scripts/03e_distill.py --teacher-4b   # Qwen3-4B teacher and its students
python scripts/03f_student_search.py --teacher models/teacher/teacher_oof_qwen3_4b.npz --teacher-name Qwen3-4B --stage A
python scripts/03g_student_ensemble_check.py # two-student ensemble check
python scripts/03e_distill.py --teacher-8b   # Qwen3-8B LoRA teacher (needs peft)
python -m pytest -q
```

Every fine-tuning run is recorded in `reports/finetune_sweep.json`. The final results are in `reports/07_router.json`,
`reports/03c_finetune.json`, `reports/07a_text_export.json`, `reports/10_server_memory.json` and
`reports/summary.json`.
