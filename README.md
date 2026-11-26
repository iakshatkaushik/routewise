# RouteWise

**Send each prompt to the cheapest LLM that can answer it well.**

![Python 3.13](https://img.shields.io/badge/python-3.13-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-orange)
![PyTorch MPS](https://img.shields.io/badge/PyTorch-Apple%20GPU%20(MPS)-red)
![FastAPI](https://img.shields.io/badge/FastAPI-web%20app-009688)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Strong LLMs cost far more than small ones, yet most everyday prompts don't need them.
RouteWise trains a classical machine-learning **router** that reads a prompt *before any model
answers it*, predicts whether a cheap model will give a poor answer, and only then pays for
the strong model. It comes with a web app where you can chat and watch every routing decision.

![Routing curve: share of the quality gap recovered at each strong-model budget](reports/figures/routing_curve.png)

The idea follows the RouteLLM paper (Ong et al., 2024, LMSYS / UC Berkeley); every model and
algorithm here is implemented and tuned for this project, and Apriori is written from scratch.

## Results (10,000 held-out test prompts, never used for tuning)

| | Final router (`ROUTER_MODE=full`) | Light router (`ROUTER_MODE=light`) | Random routing | Perfect router |
|---|---|---|---|---|
| **APGR** (share of quality gap recovered, averaged over all budgets) | **0.691** | 0.683 | 0.500 | 0.894 |
| ROC-AUC for "cheap model will fail" | **0.781** | 0.768 | 0.500 | 1.000 |
| Strong-model calls needed for 50% of the quality gap | **25%** | 26% | 50% | - |
| Strong-model calls needed for 80% of the quality gap | **55%** | 57% | 80% | - |

At a 30% strong-model budget the final router recovers 56.9% of the quality gap (average score 4.75 / 5,
against 4.42 for the cheap model alone); at 50% it recovers 77.1% (4.87 / 5).

- **Final router**: a PyTorch neural network on sentence embeddings (trained on the Mac's GPU) plus
  logistic regression on TF-IDF features, combined by a logistic-regression stacker. Selected by greedy
  forward selection on validation data.
- **Speed and size**: 16 ms median to route one prompt on one CPU core (p95 160 ms); model file 16.9 MB.
- **Memory** (measured locally on macOS): about 500 MB in full mode and 270 MB in light mode.
- **Budget control holds on unseen data**: asking for 30% strong-model traffic sent 29.3% of test prompts there.

All numbers come from `reports/*.json`, produced by the pipeline; the web app's Results page shows them all.

## GPU use (Apple Silicon)

Two steps run on the Mac's GPU through PyTorch's `mps` backend (CUDA or CPU is used automatically elsewhere):

- `scripts/02b_embeddings.py` encodes all 119,101 prompts with `BAAI/bge-small-en-v1.5`
  (7.3 ms per prompt on an Apple M3).
- `scripts/03b_gpu_network.py` trains the embedding neural network (grid search plus an activation-function
  study: logistic, tanh, ReLU, GELU). Each training run takes a few seconds.

The scikit-learn models run on the CPU; scikit-learn has no GPU support. The deployed server needs no GPU or
PyTorch: the network is exported to NumPy, and embeddings are computed with ONNX Runtime (fastembed).
`scripts/09_check_onnx_parity.py` confirms that the CPU (ONNX) and GPU (PyTorch) embeddings agree
(cosine similarity of at least 0.999998 on 1,000 test prompts).


## Syllabus coverage (UML501)

| Unit | Where it is used | File |
|---|---|---|
| Data pre-processing: cleaning, integration, transformation, reduction, scaling, train/test split | Unicode/whitespace cleaning, de-duplication, leakage guard, TF-IDF, standardised prompt statistics, 256-dim SVD, validation + held-out test split | `src/llm_router/data.py`, `src/llm_router/features.py`, `scripts/01_preprocess.py`, `scripts/02_features.py` |
| Regression: simple, multiple, polynomial, regularisation; RMSE, MAE, correlation, R² | Predicting the cheap model's 1-5 score; Ridge, Lasso, Elastic Net | `scripts/04_regression.py` |
| Classification: logistic regression, decision tree (Gini and information gain), random forest, Naive Bayes, K-NN, SVM; sensitivity, specificity, precision, recall | Predicting "cheap model fails"; all tuned and compared | `scripts/03_classifiers.py`, `src/llm_router/metrics.py` |
| Clustering: partitioned, hierarchical, density-based | K-Means (silhouette-chosen k), Ward agglomerative, DBSCAN; failure rate per prompt type | `src/llm_router/clustering.py`, `scripts/05_clustering.py` |
| Association rules: naive algorithm, Apriori | Both written from scratch; rules "properties → cheap model fails" ranked by lift | `src/llm_router/association.py`, `scripts/06_association.py` |
| Artificial neural networks, activation functions | scikit-learn MLP, and a PyTorch network trained on the GPU (part of the final router); logistic vs tanh vs ReLU vs GELU comparison | `scripts/03_classifiers.py`, `scripts/03b_gpu_network.py`, `src/llm_router/nn.py` |

## How it works

1. **Data**: the public [RouteLLM `gpt4_dataset`](https://huggingface.co/datasets/routellm/gpt4_dataset):
   about 119,000 real user prompts (Chatbot Arena, ShareGPT and others), each answered by GPT-4 (strong)
   and Mixtral-8x7B (cheap), with GPT-4's 1-5 score of the Mixtral answer. A score of 3 or less counts as
   a failure (13.5% of prompts).
2. **Features**: word TF-IDF (1-2 grams), character TF-IDF (3-5 grams) and 24 hand-made prompt statistics
   (length, code and maths markers, constraints, question type, language …). Chosen by an ablation study.
3. **Models**: every syllabus classifier is tuned on a validation split and scored once on 10,000 held-out
   test prompts. The final router stacks the best complementary models with a logistic-regression stacker
   into one calibrated probability, P(cheap model fails). Models that would be too large to serve are
   left out of the final router (K-NN keeps all training vectors; the random forest is hundreds of MB).
4. **Budget control**: you choose what share of traffic may use the strong model (for example 30%); the
   router converts that share into a probability threshold.
5. **Evaluation**: the RouteLLM metrics. **APGR** is the average share of the quality gap between "cheap
   only" and "strong only" that the router recovers (random routing = 0.5). **CPT(80%)** is the share of
   strong-model calls needed to recover 80% of that gap.

## Run it locally

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

```bash
.venv/bin/python scripts/run_all.py
```

The full pipeline takes about 1.5 hours on an M3 MacBook (step 3's model grid and step 2b's embeddings
are the slow parts). `scripts/run_all.py 07` resumes from a given step.

```bash
.venv/bin/uvicorn app.main:app --reload
```

Then open http://127.0.0.1:8000. Without an API key the app runs in routing-only mode. To get real answers,
create a free [Groq](https://console.groq.com) API key, copy `.env.example` to `.env`, fill in `LLM_API_KEY`
and start the server with `--env-file .env`. Any OpenAI-compatible provider works through `LLM_BASE_URL`.
Model IDs change: check the provider's model list before deploying.

Run the tests:

```bash
.venv/bin/python -m pytest -q
```

## Deploy

The image only contains the web service and the final router (`models/router.joblib`).

- **Render (free)**: push this folder to GitHub, then in Render choose *New → Blueprint* and pick the repo;
  `render.yaml` sets everything up. Add `LLM_API_KEY` in the Render dashboard (or leave it empty for the
  routing-only demo). The free plan has 512 MB of memory, so `render.yaml` uses `ROUTER_MODE=light`; on a
  plan with 1 GB or more, set `ROUTER_MODE=full`. Free instances sleep when idle, so the first visit takes
  up to a minute.
- **Any Docker host**: `docker build -t routewise .` then `docker run -p 8000:8000 --env-file .env routewise`.

## API

`POST /api/route` with `{"prompt": "...", "strong_share": 0.3}` returns the decision, the probability,
the threshold, the expected quality at that budget and the words that pushed the decision each way.
`POST /api/chat` does the same and then calls the chosen model. Interactive docs are at `/api/docs`.

## Project layout

```
src/llm_router/   library code: data cleaning, features, metrics, clustering, Apriori, neural network,
                  CPU embeddings, inference router
scripts/          numbered pipeline steps 01-08 and run_all.py
app/              FastAPI service and the web UI (chat + results dashboard)
reports/          JSON results and figures produced by the pipeline
models/           router.joblib (the deployable model); other models are rebuilt by the pipeline
tests/            unit and API tests
```

## Limitations

- The strong model's answers are treated as perfect (score 5), because the GPT-4 judge used them as the
  reference. Real strong-model quality is a little lower, so the savings are slightly optimistic.
- The labels describe Mixtral-8x7B vs GPT-4. The deployed app routes between newer models; prompt
  difficulty transfers reasonably across model pairs (the RouteLLM paper reports this), but it has not
  been re-measured for the new pair.
- The judge is itself an LLM, so the labels contain some noise; this caps how well any router can do.

## Author and licence

Built by Akshat Kaushik for UML501 (Machine Learning), Thapar Institute of Engineering and Technology.
Released under the [MIT licence](LICENSE). The training data is the public RouteLLM `gpt4_dataset`;
it is downloaded by `scripts/01_preprocess.py` and not stored in this repository.
