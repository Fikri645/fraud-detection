"""
Concept-drift report: compare the TEST period against the TRAIN period via PSI,
and also score-drift using the trained model. Saves drift_report.json.

    python scripts/run_drift.py
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402

from src import config, preprocess, drift  # noqa: E402


def main():
    df_tr = preprocess.load_features("train")
    df_te = preprocess.load_features("test")

    # Feature-level PSI (train = reference, test = current)
    report = drift.feature_drift_report(df_tr, df_te, config.NUMERIC_FEATURES)
    print("[drift] Feature PSI (train -> test):")
    for r in report[:10]:
        print(f"  {r['feature']:28s} PSI={r['psi']:.4f}  {r['status']}")

    # Score-level PSI (the single most useful production monitor)
    score_psi = None
    if config.LGBM_MODEL.exists():
        model = joblib.load(config.LGBM_MODEL)
        X_tr, _ = preprocess.split_xy(df_tr, categorical=True)
        X_te, _ = preprocess.split_xy(df_te, categorical=True)
        s_tr = model.predict_proba(X_tr)[:, 1]
        s_te = model.predict_proba(X_te)[:, 1]
        score_psi = drift.psi(s_tr, s_te)
        print(f"\n[drift] MODEL SCORE PSI = {score_psi:.4f}  ({drift.classify_psi(score_psi)})")

    out = {
        "reference": "train period (2019-01 .. 2020-06)",
        "current": "test period (2020-06 .. 2020-12)",
        "feature_psi": report,
        "score_psi": round(score_psi, 4) if score_psi is not None else None,
        "score_psi_status": drift.classify_psi(score_psi) if score_psi is not None else None,
    }
    with open(config.MODELS_DIR / "drift_report.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("[drift] saved drift_report.json")


if __name__ == "__main__":
    main()
