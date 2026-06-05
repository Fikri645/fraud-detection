"""
Central configuration — paths, column groups, model defaults, business costs.
Import this everywhere instead of scattering magic strings.
"""
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[1]
DATA_RAW    = ROOT / "data" / "raw"
DATA_PROC   = ROOT / "data" / "processed"
MODELS_DIR  = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Sparkov dataset files (Kaggle: kartik2112/fraud-detection)
RAW_TRAIN_CSV = DATA_RAW / "fraudTrain.csv"
RAW_TEST_CSV  = DATA_RAW / "fraudTest.csv"
KAGGLE_DATASET = "kartik2112/fraud-detection"

# Processed feature tables
FEATURES_TRAIN = DATA_PROC / "features_train.parquet"
FEATURES_TEST  = DATA_PROC / "features_test.parquet"

# Model artifacts
LGBM_MODEL   = MODELS_DIR / "lgbm_fraud.joblib"
XGB_MODEL    = MODELS_DIR / "xgb_fraud.joblib"
AE_MODEL     = MODELS_DIR / "autoencoder.pt"
GNN_MODEL    = MODELS_DIR / "gnn_fraud.pt"
MODEL_META   = MODELS_DIR / "model_meta.json"
SHAP_VALUES  = MODELS_DIR / "shap_values.npy"
FEATURE_PIPE = MODELS_DIR / "feature_pipeline.joblib"

# ── Target / identifiers ───────────────────────────────────────────────────
TARGET    = "is_fraud"
CARD_COL  = "cc_num"
TIME_COL  = "trans_date_trans_time"
MERCHANT_COL = "merchant"

# Raw columns we drop (PII / identifiers not used as features directly)
DROP_RAW = ["first", "last", "street", "trans_num", "unix_time"]

# ── Engineered feature groups (populated by features.py) ───────────────────
# These names are produced by src/features.py::engineer_features
NUMERIC_FEATURES = [
    "amt", "amt_log",
    "hour", "day_of_week", "is_night", "is_weekend",
    "age",
    "city_pop_log",
    # Geo
    "dist_home_merchant_km", "dist_from_prev_txn_km",
    # Velocity (per card)
    "txn_count_1h", "txn_count_24h", "txn_count_7d",
    "amt_sum_1h", "amt_sum_24h", "amt_sum_7d",
    "amt_mean_24h",
    "secs_since_prev_txn",
    # Behavioral
    "amt_dev_from_card_mean", "amt_ratio_to_card_mean",
    "distinct_merchants_24h",
]

CATEGORICAL_FEATURES = [
    "category", "gender", "state",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# ── Train / test split ──────────────────────────────────────────────────────
# Sparkov ships pre-split temporally (fraudTrain = earlier, fraudTest = later).
# We respect that temporal ordering — no random shuffling (avoids leakage).
RANDOM_SEED = 42
VALID_FRACTION = 0.15  # last 15% of train (by time) held out for early stopping

# ── Business cost model ─────────────────────────────────────────────────────
# Used for threshold optimization. Tunable to a company's economics.
COST_FALSE_NEGATIVE = 1.0   # missed fraud → lose the transaction amount (avg modelled per-txn)
COST_FALSE_POSITIVE = 5.0   # blocking a legit txn → customer friction / churn (flat cost units)
# Interpretation: blocking one good customer costs ~5× a single missed-fraud unit of goodwill.

# ── Gradient boosting defaults (overridden by Optuna) ──────────────────────
LGBM_BASE_PARAMS = {
    "objective": "binary",
    "metric": "average_precision",   # PR-AUC — correct metric for imbalance
    "boosting_type": "gbdt",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbose": -1,
}

OPTUNA_N_TRIALS = 40

# ── Autoencoder ─────────────────────────────────────────────────────────────
AE_HIDDEN = [32, 16, 8]
AE_EPOCHS = 30
AE_BATCH  = 2048
AE_LR     = 1e-3

# ── GNN ─────────────────────────────────────────────────────────────────────
GNN_HIDDEN     = 64
GNN_LAYERS     = 2
GNN_EPOCHS     = 50
GNN_BATCH      = 4096
GNN_LR         = 5e-3
GNN_NEIGHBORS  = [15, 10]   # neighbor sampling fan-out per layer

# ── MLflow ──────────────────────────────────────────────────────────────────
MLFLOW_EXPERIMENT = "fraud-detection"

# ── Drift (PSI) ─────────────────────────────────────────────────────────────
PSI_BINS = 10
PSI_THRESHOLD_WARN = 0.10   # 0.1–0.25 = moderate shift
PSI_THRESHOLD_ALERT = 0.25  # > 0.25 = significant shift
