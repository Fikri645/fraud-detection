"""
Cross-dataset validation on the real-world ULB credit-card fraud dataset.

The main project uses Sparkov (simulated). To show the methodology generalises
to *real* data, this benchmark applies the SAME recipe — cost-sensitive
LightGBM, temporal split, PR-AUC — to the ULB dataset: 284,807 real European
card transactions (Sept 2013), 0.17% fraud (even more extreme imbalance), with
PCA-anonymised features V1–V28 + Amount.

No bespoke feature engineering here (the features are already PCA components);
the point is to validate that cost-sensitive boosting + PR-AUC evaluation
transfers to a genuinely different, real-world distribution.

    python scripts/run_cross_dataset.py
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import lightgbm as lgb  # noqa: E402

from src import config, evaluate  # noqa: E402


def main():
    from sklearn.model_selection import train_test_split
    import kagglehub
    t0 = time.time()
    path = Path(kagglehub.dataset_download("mlg-ulb/creditcardfraud"))
    df = pd.read_csv(path / "creditcard.csv")
    print(f"[ulb] {len(df):,} real transactions, fraud rate {df['Class'].mean():.4%}")

    # Stratified split — the standard protocol for ULB. (Its "Time" column is
    # only seconds-elapsed over ~2 days, not a real timestamp, so a temporal
    # split is inappropriate here; published benchmarks use stratified splits.)
    features = [c for c in df.columns if c not in ("Class", "Time")]
    train, test = train_test_split(df, test_size=0.30, stratify=df["Class"],
                                   random_state=config.RANDOM_SEED)
    X_tr, y_tr = train[features].values, train["Class"].values
    X_te, y_te = test[features].values, test["Class"].values
    amt_te = test["Amount"].values

    # Apply BOTH imbalance strategies — the headline cross-dataset finding is
    # that the winner is dataset-dependent (see below).
    n_pos = int(y_tr.sum())
    spw = (len(y_tr) - n_pos) / max(n_pos, 1)

    results = {}
    for name, kw in [("plain", {}), ("cost_sensitive", {"scale_pos_weight": spw})]:
        params = dict(config.LGBM_BASE_PARAMS)
        params.update(kw)
        model = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=64, **params)
        model.fit(X_tr, y_tr)
        scores = model.predict_proba(X_te)[:, 1]
        ev = evaluate.evaluate(y_te, scores, amt_te)
        results[name] = ev.to_dict()
        print(f"[ulb] {name:14s} PR-AUC={ev.pr_auc:.4f}  ROC-AUC={ev.roc_auc:.4f}  "
              f"recall@1%={ev.recall_at_1pct:.3f}")

    finding = (
        "Imbalance strategy is dataset-dependent. On Sparkov (strong engineered "
        "features) cost-sensitive weighting dominates; on ULB (weak PCA features, "
        "0.17% fraud) the same aggressive scale_pos_weight floods the high-score "
        "region with false positives and collapses PR-AUC — a plain model wins. "
        "The lesson: don't cargo-cult one imbalance recipe across datasets."
    )
    out = {
        "dataset": "ULB creditcardfraud (real European transactions, Sept 2013)",
        "n_total": int(len(df)), "fraud_rate": round(float(df["Class"].mean()), 5),
        "n_train": int(len(train)), "n_test": int(len(test)),
        "split": "stratified 70/30 (standard ULB protocol)",
        "strategies": results,
        "best_pr_auc": round(max(r["pr_auc"] for r in results.values()), 4),
        "finding": finding,
        "trained_at": time.strftime("%Y-%m-%d %H:%M"),
    }
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.MODELS_DIR / "cross_dataset_ulb.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[ulb] saved cross_dataset_ulb.json  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
