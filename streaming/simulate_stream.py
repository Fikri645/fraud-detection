"""
Replay the test period as a live transaction stream and benchmark scoring.

This proves the model is deployable, not just a notebook artifact: it feeds raw
transactions one-by-one (in true chronological order) through the online feature
store + model, exactly as a production consumer would, and reports per-event
latency percentiles plus the running fraud-catch rate.

    python streaming/simulate_stream.py --limit 20000
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config, data  # noqa: E402
from src.online import OnlineFeatureStore  # noqa: E402


def run(limit: int = 20000, save: bool = True):
    if not config.LGBM_MODEL.exists():
        raise SystemExit("Model missing — run scripts/run_training.py first.")
    model = joblib.load(config.LGBM_MODEL)
    threshold = 0.5
    if config.MODEL_META.exists():
        meta = json.loads(config.MODEL_META.read_text(encoding="utf-8"))
        threshold = meta["test_metrics"].get("best_threshold", 0.5)

    # Stream the test period in true time order
    df = data.load_raw("test").head(limit)
    store = OnlineFeatureStore()

    latencies, scores, labels = [], [], []
    for _, row in df.iterrows():
        txn = row.to_dict()
        txn["dob"] = str(txn["dob"])
        t0 = time.perf_counter()
        feats = store.transform(txn)
        X = pd.DataFrame([feats])[config.ALL_FEATURES]
        for c in config.CATEGORICAL_FEATURES:
            X[c] = X[c].astype("category")
        prob = float(model.predict_proba(X)[:, 1][0])
        store.update(txn)
        latencies.append((time.perf_counter() - t0) * 1000)
        scores.append(prob)
        labels.append(int(row[config.TARGET]))

    lat = np.array(latencies)
    scores = np.array(scores)
    labels = np.array(labels)
    flagged = scores >= threshold
    caught = int(((flagged) & (labels == 1)).sum())
    total_fraud = int(labels.sum())

    result = {
        "transactions": int(len(df)),
        "throughput_per_sec": round(len(df) / (lat.sum() / 1000), 1),
        "latency_ms": {
            "p50": round(float(np.percentile(lat, 50)), 2),
            "p95": round(float(np.percentile(lat, 95)), 2),
            "p99": round(float(np.percentile(lat, 99)), 2),
            "max": round(float(lat.max()), 2),
        },
        "fraud_in_stream": total_fraud,
        "fraud_caught": caught,
        "catch_rate": round(caught / total_fraud, 3) if total_fraud else None,
        "decline_rate": round(float(flagged.mean()), 5),
    }
    print(json.dumps(result, indent=2))
    if save:
        out = config.MODELS_DIR / "stream_benchmark.json"
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"[stream] saved {out.name}")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=20000)
    args = ap.parse_args()
    run(args.limit)
