"""Project-wide settings: paths, label definition and random seed."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

SEED = 42

# The dataset scores the cheap model's (Mixtral-8x7B) answer from 1 to 5, judged by GPT-4.
# A score of 3 or less counts as a failed answer: the prompt should go to the strong model.
FAIL_SCORE = 3

# The strong model (GPT-4) answer is the reference the judge compared against,
# so a prompt sent to the strong model is treated as getting the top score.
STRONG_SCORE = 5.0

# Prompts longer than this are truncated (99th percentile is about 2,550 characters).
MAX_PROMPT_CHARS = 4000

for _p in (DATA_PROCESSED, MODELS, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)
