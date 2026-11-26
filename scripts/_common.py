"""Shared helpers for the numbered pipeline scripts."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_router import config  # noqa: E402


def _finite(o):
    """Replace NaN/inf with None so the JSON is valid (browsers and Starlette reject NaN)."""
    import math
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_finite(v) for v in o]
    return o


def save_json(name, obj):
    path = config.REPORTS / name
    path.write_text(json.dumps(_finite(obj), indent=2, allow_nan=False))
    print(f"saved {path.relative_to(ROOT)}")


def load_json(name):
    return json.loads((config.REPORTS / name).read_text())


class Timer:
    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.s = time.perf_counter() - self.t


def slug(name):
    import re
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
