"""Step 10 - memory of the real web service (uvicorn + FastAPI + router) in light and full mode.

Starts `uvicorn app.main:app` with one worker and one CPU thread, sends 20 test prompts to /api/route,
then reads the server's current (VmRSS) and peak (VmHWM) resident memory from /proc (Linux only).
Writes reports/10_server_memory.json. Used to choose ROUTER_MODE for 512 MB servers (render.yaml).
"""
import json
import os
import platform
import subprocess
import sys
import time
import urllib.request

from _common import ROOT, save_json
from llm_router import config, data

PORT = 8765


def status(pid):
    s = dict(line.split(":", 1) for line in open(f"/proc/{pid}/status"))
    return {k: round(int(s[k].split()[0]) / 1024, 1) for k in ("VmRSS", "VmHWM")}


def measure(mode, prompts):
    env = {**os.environ, "ROUTER_MODE": mode, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(PORT), "--workers", "1"],
                            cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(240):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/health", timeout=1)
                break
            except OSError:
                time.sleep(0.5)
        after_start = status(proc.pid)
        for p in prompts:
            req = urllib.request.Request(f"http://127.0.0.1:{PORT}/api/route",
                                         json.dumps({"prompt": p, "strong_share": 0.3}).encode(),
                                         {"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=30).read()
        after = status(proc.pid)
    finally:
        proc.terminate()
        proc.wait()
    return {"rss_mb_after_start": after_start["VmRSS"], "rss_mb": after["VmRSS"], "peak_rss_mb": after["VmHWM"],
            "requests": len(prompts)}


if __name__ == "__main__":
    prompts = [p[: config.MAX_PROMPT_CHARS] for p in data.load("test").prompt.values[:20]]
    out = {mode: measure(mode, prompts) for mode in ("light", "full")}
    out["platform"] = f"{platform.system()} {platform.machine()}, Python {platform.python_version()}"
    print(out)
    save_json("10_server_memory.json", out)
