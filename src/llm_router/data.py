"""Data loading and cleaning (syllabus unit: Data Pre-processing).

Raw data: the public RouteLLM `gpt4_dataset` on Hugging Face. Each row is a user prompt,
the answers of GPT-4 (strong, expensive) and Mixtral-8x7B (cheap), and GPT-4's 1-5 score
of the Mixtral answer.

Cleaning steps
1. Unicode normalisation (NFKC) and removal of control characters.
2. Whitespace normalisation and truncation of extreme-length prompts.
3. Removal of empty prompts and invalid scores.
4. De-duplication on a normalised key (case and whitespace folded), inside each split.
5. Leakage guard: training prompts that also appear in the test split are dropped.
"""
from __future__ import annotations

import re
import unicodedata
from urllib.request import urlretrieve

import pandas as pd

from . import config

HF_BASE = "https://huggingface.co/datasets/routellm/gpt4_dataset/resolve/main"
RAW_FILES = {"train": "train.jsonl", "test": "valid.jsonl"}

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SPACES = re.compile(r"[ \t]+")
_BLANK_LINES = re.compile(r"\n{3,}")


def download(force: bool = False) -> None:
    """Download the raw JSONL files if they are not already on disk."""
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    for name in RAW_FILES.values():
        target = config.DATA_RAW / name
        if force or not target.exists():
            print(f"downloading {name} ...")
            urlretrieve(f"{HF_BASE}/{name}", target)


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    text = _CONTROL.sub("", text).replace("\r\n", "\n").replace("\r", "\n")
    text = _SPACES.sub(" ", text)
    text = _BLANK_LINES.sub("\n\n", text).strip()
    return text[: config.MAX_PROMPT_CHARS]


def _dedupe_key(s: pd.Series) -> pd.Series:
    return s.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report = {"rows_in": len(df)}
    out = pd.DataFrame(
        {
            "prompt": df["prompt"].map(clean_text),
            "source": df["source"].map(lambda s: s[0] if isinstance(s, list) and s else str(s)),
            "score": pd.to_numeric(df["mixtral_score"], errors="coerce"),
            "strong_len": df["gpt4_response"].str.len(),
            "cheap_len": df["mixtral_response"].str.len(),
        }
    )
    bad = (out["prompt"] == "") | ~out["score"].between(1, 5)
    report["dropped_empty_or_invalid"] = int(bad.sum())
    out = out[~bad]
    key = _dedupe_key(out["prompt"])
    dup = key.duplicated()
    report["dropped_duplicates"] = int(dup.sum())
    out = out[~dup].copy()
    out["score"] = out["score"].astype(int)
    out["label"] = (out["score"] <= config.FAIL_SCORE).astype(int)  # 1 = needs strong model
    report["rows_out"] = len(out)
    return out.reset_index(drop=True), report


def build_processed() -> dict:
    """Clean both splits, apply the leakage guard and save them as Parquet."""
    download()
    train, r_train = clean(pd.read_json(config.DATA_RAW / RAW_FILES["train"], lines=True))
    test, r_test = clean(pd.read_json(config.DATA_RAW / RAW_FILES["test"], lines=True))
    leak = _dedupe_key(train["prompt"]).isin(set(_dedupe_key(test["prompt"])))
    r_train["dropped_overlap_with_test"] = int(leak.sum())
    train = train[~leak].reset_index(drop=True)
    r_train["rows_out"] = len(train)
    train.to_parquet(config.DATA_PROCESSED / "train.parquet", index=False)
    test.to_parquet(config.DATA_PROCESSED / "test.parquet", index=False)
    return {"train": r_train, "test": r_test,
            "fail_rate_train": float(train["label"].mean()),
            "fail_rate_test": float(test["label"].mean())}


def load(split: str) -> pd.DataFrame:
    path = config.DATA_PROCESSED / f"{split}.parquet"
    if not path.exists():
        build_processed()
    return pd.read_parquet(path)
