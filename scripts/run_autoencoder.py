"""
Train the unsupervised autoencoder and compare it to the supervised model.

    python scripts/run_autoencoder.py
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src import config, preprocess, autoencoder, evaluate  # noqa: E402


def main():
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    df_tr = preprocess.load_features("train")
    df_te = preprocess.load_features("test")

    # Numeric matrix; fit scaler on train, apply to test
    Xn_tr, Xn_te, _ = preprocess.encode_numeric_matrix(df_tr, df_te)
    y_tr = df_tr[config.TARGET].astype(int).values
    y_te = df_te[config.TARGET].astype(int).values

    # Train on LEGIT-only transactions
    X_legit = Xn_tr.values[y_tr == 0]
    print(f"[ae] training on {len(X_legit):,} legit transactions "
          f"({'GPU' if torch.cuda.is_available() else 'CPU'})")
    model = autoencoder.train_autoencoder(X_legit)

    # Anomaly score = reconstruction error on test
    scores = autoencoder.reconstruction_error(model, Xn_te.values)
    ev = evaluate.evaluate(y_te, scores, df_te["amt"].values)
    print(f"\n[ae] TEST  PR-AUC={ev.pr_auc:.4f}  ROC-AUC={ev.roc_auc:.4f}  "
          f"recall@1%={ev.recall_at_1pct:.3f}")

    torch.save(model.state_dict(), config.AE_MODEL)
    out = {
        "model": "Autoencoder (unsupervised, legit-only training)",
        "trained_at": time.strftime("%Y-%m-%d %H:%M"),
        "test_metrics": ev.to_dict(),
        "n_features_in": int(Xn_tr.shape[1]),
    }
    with open(config.MODELS_DIR / "autoencoder_meta.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[ae] saved {config.AE_MODEL.name}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
