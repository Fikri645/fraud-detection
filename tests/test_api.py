"""Smoke test for the FastAPI scoring service. Skips if deps/model are absent."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from src import config  # noqa: E402

if not config.LGBM_MODEL.exists():
    pytest.skip("trained model not available", allow_module_level=True)

from fastapi.testclient import TestClient  # noqa: E402
from api.main import app  # noqa: E402

client = TestClient(app)

_TXN = {
    "cc_num": 1234567890123456, "amt": 842.15, "unix_time": 1593561600,
    "merchant": "demo", "category": "shopping_net", "gender": "F", "state": "NY",
    "lat": 40.71, "long": -74.0, "merch_lat": 36.0, "merch_long": -90.0,
    "city_pop": 1000000, "dob": "1988-03-09",
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_score_returns_valid_response():
    r = client.post("/score", json=_TXN)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["decision"] in {"approve", "review", "decline"}
    assert body["latency_ms"] >= 0
    assert isinstance(body["top_factors"], list)


def test_score_high_amount_night_is_riskier():
    """A large late-night out-of-pattern txn should score >= a small daytime one."""
    low = dict(_TXN, amt=5.0, unix_time=1593585600)   # small, midday
    high = dict(_TXN, amt=4000.0, unix_time=1593561600)
    s_low = client.post("/score", json=low).json()["fraud_probability"]
    s_high = client.post("/score", json=high).json()["fraud_probability"]
    assert s_high >= s_low
