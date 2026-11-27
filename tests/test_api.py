import pytest

from llm_router import config

pytestmark = pytest.mark.skipif(not (config.MODELS / "router.joblib").exists(),
                                reason="trained router not built yet (run scripts/run_all.py)")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from app.main import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    j = client.get("/api/health").json()
    assert j["status"] == "ok" and "components" in j["router"]


def test_route_decision(client):
    j = client.post("/api/route", json={"prompt": "Prove that there are infinitely many primes.", "strong_share": 0.3}).json()
    assert j["route"] in {"cheap", "strong"}
    assert 0 <= j["p_cheap_fails"] <= 1
    assert j["router_latency_ms"] < 200


def test_budget_extremes(client):
    p = "hello there"
    assert client.post("/api/route", json={"prompt": p, "strong_share": 0.0}).json()["route"] == "cheap"
    assert client.post("/api/route", json={"prompt": p, "strong_share": 1.0}).json()["route"] == "strong"


def test_validation(client):
    assert client.post("/api/route", json={"prompt": ""}).status_code == 422
    assert client.post("/api/route", json={"prompt": "x" * 4001}).status_code == 422
    assert client.post("/api/route", json={"prompt": "ok", "strong_share": 2}).status_code == 422


def test_chat_without_key_is_route_only(client, monkeypatch):
    import app.main as m
    monkeypatch.setattr(m, "LLM_API_KEY", "")
    j = client.post("/api/chat", json={"prompt": "hi"}).json()
    assert j["answer"] is None and "decision" in j


def test_pages_and_metrics(client):
    assert client.get("/").status_code == 200
    assert client.get("/results").status_code == 200
    assert client.get("/api/metrics").status_code == 200


def test_light_mode_router():
    """Light mode needs no embedding runtime and no ONNX text model (for 512 MB servers)."""
    from llm_router.router import Router
    r = Router.load(mode="light")
    assert r.mode == "light" and r.components
    assert all(c["input"] == "sparse" for c in r.components)
    feats = r._features(["hello"])
    assert set(feats) == {"sparse"}
    out = r.route("Translate this sentence into German: good night.", 0.3)
    assert out["route"] in {"cheap", "strong"}


def test_full_mode_router():
    from llm_router.router import Router
    r = Router.load(mode="full")
    assert r.mode == "full" and r.components
    spec = r.a.get("text_model")
    if spec and not (config.MODELS / spec["onnx"]).exists():
        pytest.skip("text model file missing")
    if any(c["input"] == "embed" for c in r.components):
        pytest.importorskip("fastembed")
    for p in ["hi", "Prove that there are infinitely many primes.", "x" * 4000]:
        out = r.route(p, 0.3)
        assert out["route"] in {"cheap", "strong"} and 0 <= out["p_cheap_fails"] <= 1
