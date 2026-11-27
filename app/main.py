"""FastAPI web service for RouteWise, the LLM router.

Endpoints
  GET  /                 chat UI that shows every routing decision
  GET  /results          project results dashboard
  GET  /api/health       liveness + model metadata
  POST /api/route        route a prompt (no LLM call)
  POST /api/chat         route a prompt, then call the chosen LLM (needs LLM_API_KEY)
  GET  /api/metrics      evaluation summary used by the dashboard

LLM provider: any OpenAI-compatible chat-completions API. Defaults target Groq's free tier.
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_router.router import get_router  # noqa: E402

STATIC = Path(__file__).parent / "static"
SUMMARY = ROOT / "reports" / "summary.json"
FIGURES = ROOT / "reports" / "figures"

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
CHEAP_MODEL = os.getenv("CHEAP_MODEL", "openai/gpt-oss-20b")
STRONG_MODEL = os.getenv("STRONG_MODEL", "openai/gpt-oss-120b")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))
# gpt-oss models think before they answer; that hidden reasoning counts against MAX_TOKENS, so keep it short
REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "low")
CHAT_LIMIT = int(os.getenv("CHAT_LIMIT_PER_10MIN", "15"))
ROUTE_LIMIT = int(os.getenv("ROUTE_LIMIT_PER_MIN", "120"))

@asynccontextmanager
async def lifespan(_app):
    get_router().route("warm up", explain=False)  # load the model before the first request
    yield


app = FastAPI(title="RouteWise", version="1.0", docs_url="/api/docs", redoc_url=None, lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")
if FIGURES.exists():
    app.mount("/figures", StaticFiles(directory=FIGURES), name="figures")

_hits: dict[tuple[str, str], deque] = defaultdict(deque)


def _rate_limit(request: Request, bucket: str, limit: int, window_s: int) -> None:
    ip = request.headers.get("x-forwarded-for", request.client.host if request.client else "?").split(",")[0].strip()
    q, now = _hits[(bucket, ip)], time.monotonic()
    while q and now - q[0] > window_s:
        q.popleft()
    if len(q) >= limit:
        raise HTTPException(429, f"Too many requests: limit is {limit} per {window_s // 60 or 1} min. Try again shortly.")
    q.append(now)


class RouteIn(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    strong_share: float = Field(0.3, ge=0.0, le=1.0, description="Share of traffic the strong model may take")


class ChatIn(RouteIn):
    compare: bool = Field(False, description="Also ask the other model, to compare answers")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/results", include_in_schema=False)
def results():
    return FileResponse(STATIC / "results.html")


@app.get("/api/health")
def health():
    r = get_router()
    return {"status": "ok", "llm_configured": bool(LLM_API_KEY), "cheap_model": CHEAP_MODEL,
            "strong_model": STRONG_MODEL, "router": r.meta}


@app.post("/api/route")
def route(body: RouteIn, request: Request):
    _rate_limit(request, "route", ROUTE_LIMIT, 60)
    return get_router().route(body.prompt, body.strong_share)


async def _complete(client: httpx.AsyncClient, model: str, prompt: str) -> dict:
    t = time.perf_counter()
    r = await client.post(f"{LLM_BASE_URL}/chat/completions",
                          headers={"Authorization": f"Bearer {LLM_API_KEY}"},
                          json={"model": model, "max_tokens": MAX_TOKENS,
                                "messages": [{"role": "user", "content": prompt}],
                                **({"reasoning_effort": REASONING_EFFORT} if "gpt-oss" in model and REASONING_EFFORT else {})})
    if r.status_code != 200:
        detail = r.text[:300]
        raise HTTPException(502, f"LLM provider error ({r.status_code}) for {model}: {detail}")
    j = r.json()
    return {"model": model, "answer": j["choices"][0]["message"].get("content") or "",
            "usage": j.get("usage", {}), "latency_ms": round((time.perf_counter() - t) * 1000)}


@app.post("/api/chat")
async def chat(body: ChatIn, request: Request):
    _rate_limit(request, "chat", CHAT_LIMIT, 600)
    decision = get_router().route(body.prompt, body.strong_share)
    if not LLM_API_KEY:
        return {"decision": decision, "answer": None,
                "note": "Routing-only demo: no LLM_API_KEY is set on the server, so no model was called."}
    chosen = STRONG_MODEL if decision["route"] == "strong" else CHEAP_MODEL
    other = CHEAP_MODEL if chosen == STRONG_MODEL else STRONG_MODEL
    async with httpx.AsyncClient(timeout=60) as client:
        main = await _complete(client, chosen, body.prompt)
        alt = await _complete(client, other, body.prompt) if body.compare else None
    return {"decision": decision, "answer": main, "comparison": alt}


@app.get("/api/metrics")
def metrics():
    if not SUMMARY.exists():
        raise HTTPException(404, "reports/summary.json not found: run scripts/run_all.py first")
    return JSONResponse(json.loads(SUMMARY.read_text()))
