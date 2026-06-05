"""
FastAPI real-time fraud-scoring service.

POST a raw transaction → get a fraud probability, an approve/review/decline
decision at the cost-optimal threshold, a per-transaction SHAP explanation
(for compliance), and the end-to-end latency. The online feature store keeps
per-card rolling state in memory so velocity features are computed incrementally
in single-digit milliseconds.

    uvicorn api.main:app --port 8000
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.online import OnlineFeatureStore  # noqa: E402
from api.schemas import Transaction, ScoreResponse, FeatureContribution  # noqa: E402

app = FastAPI(title="Fraud Detection API", version="1.0")

# Lazy-loaded singletons
_STATE = {"model": None, "store": None, "threshold": 0.5, "explainer": None}


def _load():
    if _STATE["model"] is None:
        if not config.LGBM_MODEL.exists():
            raise RuntimeError("Model missing — run scripts/run_training.py first.")
        _STATE["model"] = joblib.load(config.LGBM_MODEL)
        _STATE["store"] = OnlineFeatureStore()
        if config.MODEL_META.exists():
            meta = json.loads(config.MODEL_META.read_text(encoding="utf-8"))
            _STATE["threshold"] = meta["test_metrics"].get("best_threshold", 0.5)
        try:
            import shap
            _STATE["explainer"] = shap.TreeExplainer(_STATE["model"])
        except Exception:
            _STATE["explainer"] = None


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _STATE["model"] is not None}


@app.post("/score", response_model=ScoreResponse)
def score(txn: Transaction):
    _load()
    t0 = time.perf_counter()

    raw = txn.model_dump()
    feats = _STATE["store"].transform(raw)
    X = pd.DataFrame([feats])[config.ALL_FEATURES]
    for c in config.CATEGORICAL_FEATURES:
        X[c] = X[c].astype("category")

    prob = float(_STATE["model"].predict_proba(X)[:, 1][0])
    thr = _STATE["threshold"]

    # Three-way decision band around the cost-optimal threshold
    if prob >= thr:
        decision = "decline"
    elif prob >= thr * 0.5:
        decision = "review"
    else:
        decision = "approve"

    # Per-transaction explanation
    top = []
    if _STATE["explainer"] is not None:
        sv = _STATE["explainer"].shap_values(X)
        if isinstance(sv, list):
            sv = sv[1]
        row = np.asarray(sv)[0]
        order = np.argsort(np.abs(row))[::-1][:5]
        top = [FeatureContribution(feature=config.ALL_FEATURES[i],
                                   contribution=round(float(row[i]), 4)) for i in order]

    _STATE["store"].update(raw)
    latency = (time.perf_counter() - t0) * 1000
    return ScoreResponse(
        fraud_probability=round(prob, 5),
        decision=decision,
        threshold=round(thr, 5),
        latency_ms=round(latency, 2),
        top_factors=top,
    )
