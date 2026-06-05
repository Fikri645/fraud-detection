"""
Train the supervised fraud stack and persist artifacts for the app.

Pipeline:
  1. Load engineered features (train/test), time-based validation split
  2. Imbalance study: none / cost-sensitive / SMOTE / undersample (fair compare)
  3. Optuna HPO on the production LightGBM (native categoricals)
  4. Fit production model, evaluate on the held-out TEST period
  5. SHAP global importance for explainability
  6. Save model + model_meta.json (consumed by the Gradio app) + MLflow run

    python scripts/run_training.py
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import mlflow  # noqa: E402

from src import config, preprocess, train, evaluate, explain  # noqa: E402


def main():
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # ── 1. Load + split ──────────────────────────────────────────────────────
    df_train_full = preprocess.load_features("train")
    df_test = preprocess.load_features("test")
    df_tr, df_val = preprocess.time_validation_split(df_train_full)
    print(f"[train] train={len(df_tr):,}  valid={len(df_val):,}  test={len(df_test):,}")

    X_tr, y_tr = preprocess.split_xy(df_tr, categorical=True)
    X_val, y_val = preprocess.split_xy(df_val, categorical=True)
    X_te, y_te = preprocess.split_xy(df_test, categorical=True)
    amt_te = df_test["amt"].values

    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name="lgbm-production"):

        # ── 2. Imbalance study (numeric matrix, fair compare) ────────────────
        print("\n[train] Imbalance strategy comparison …")
        Xn_tr, Xn_val, _ = preprocess.encode_numeric_matrix(df_tr, df_val)
        imbalance = train.compare_imbalance_strategies(
            Xn_tr.values, y_tr, Xn_val.values, y_val, amounts_valid=df_val["amt"].values
        )

        # ── 3. Optuna HPO (native categoricals) ──────────────────────────────
        print("\n[train] Optuna HPO …")
        cat_idx = [X_tr.columns.get_loc(c) for c in config.CATEGORICAL_FEATURES
                   if c in X_tr.columns]
        best_params, best_val = train.tune_lgbm(
            X_tr, y_tr, X_val, y_val, categorical_feature=cat_idx
        )
        print(f"[train] best valid PR-AUC={best_val:.4f}")

        # ── 4. Fit production model + evaluate on TEST ───────────────────────
        model = train.train_production_model(
            X_tr, y_tr, X_val, y_val, best_params, categorical_feature=cat_idx
        )
        test_scores = model.predict_proba(X_te)[:, 1]
        ev = evaluate.evaluate(y_te, test_scores, amt_te)
        print(f"\n[train] TEST  PR-AUC={ev.pr_auc:.4f}  ROC-AUC={ev.roc_auc:.4f}  "
              f"recall@1%={ev.recall_at_1pct:.3f}")

        # ── 5. SHAP global importance ────────────────────────────────────────
        print("[train] SHAP …")
        sample = X_te.sample(min(5000, len(X_te)), random_state=config.RANDOM_SEED)
        shap_vals, _ = explain.compute_shap(model, sample)
        importance = explain.global_importance(shap_vals, list(X_te.columns), top_k=20)

        # ── 6. Persist ───────────────────────────────────────────────────────
        joblib.dump(model, config.LGBM_MODEL)
        prec, rec = evaluate.pr_curve_points(y_te, test_scores)

        meta = {
            "model": "LightGBM (cost-sensitive, Optuna-tuned)",
            "trained_at": time.strftime("%Y-%m-%d %H:%M"),
            "n_train": int(len(df_tr)), "n_valid": int(len(df_val)),
            "n_test": int(len(df_test)), "n_features": len(config.ALL_FEATURES),
            "test_metrics": ev.to_dict(),
            "imbalance_study": imbalance,
            "shap_importance": [{"feature": f, "value": v} for f, v in importance],
            "pr_curve": {"precision": prec, "recall": rec},
            "best_params": {k: (round(v, 5) if isinstance(v, float) else v)
                            for k, v in best_params.items()},
            "best_valid_pr_auc": round(best_val, 4),
        }
        with open(config.MODEL_META, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"[train] saved {config.LGBM_MODEL.name} + {config.MODEL_META.name}")

        # MLflow logging
        mlflow.log_params({k: v for k, v in best_params.items()
                           if isinstance(v, (int, float, str))})
        mlflow.log_metrics({
            "test_pr_auc": ev.pr_auc, "test_roc_auc": ev.roc_auc,
            "test_recall_at_1pct": ev.recall_at_1pct,
            "valid_pr_auc": best_val,
        })

    print(f"\n[train] total {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
