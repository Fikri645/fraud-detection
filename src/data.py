"""
Data acquisition + loading for the Sparkov credit-card transaction dataset.

Sparkov is a *simulated* but realistic transaction stream (Kaggle:
kartik2112/fraud-detection). Unlike the over-used ULB `creditcard.csv`
(PCA-anonymised V1–V28), Sparkov keeps human-readable columns — merchant,
category, geo-coordinates, timestamps — which is exactly what makes rich
feature engineering (velocity, geo-distance, graph) possible.

Files:
  fraudTrain.csv  ~1.30M rows (earlier period)
  fraudTest.csv   ~0.56M rows (later period)
Combined ~1.85M transactions, fraud rate ~0.58%.
"""
from __future__ import annotations

import shutil

import pandas as pd

from src import config

# Columns present in the raw Sparkov CSVs
_PARSE_DATES = [config.TIME_COL, "dob"]


def download_sparkov() -> None:
    """Download the Sparkov dataset via kagglehub and copy CSVs into data/raw."""
    if config.RAW_TRAIN_CSV.exists() and config.RAW_TEST_CSV.exists():
        print(f"[data] Using cached CSVs in {config.DATA_RAW}")
        return

    import kagglehub

    print(f"[data] Downloading {config.KAGGLE_DATASET} via kagglehub …")
    path = kagglehub.dataset_download(config.KAGGLE_DATASET)
    src_dir = __import__("pathlib").Path(path)

    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    for name in ("fraudTrain.csv", "fraudTest.csv"):
        matches = list(src_dir.rglob(name))
        if not matches:
            raise FileNotFoundError(f"{name} not found in downloaded dataset at {src_dir}")
        shutil.copy(matches[0], config.DATA_RAW / name)
        print(f"[data]   copied {name}")
    print("[data] Done.")


def _load_csv(path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=_PARSE_DATES)
    # Drop the unnamed index column Sparkov ships with
    drop = [c for c in df.columns if c.startswith("Unnamed")]
    if drop:
        df = df.drop(columns=drop)
    return df


def load_raw(which: str = "train") -> pd.DataFrame:
    """Load one raw split ('train' or 'test') as a DataFrame, time-sorted."""
    path = config.RAW_TRAIN_CSV if which == "train" else config.RAW_TEST_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — run `python scripts/download_data.py` first."
        )
    df = _load_csv(path)
    return df.sort_values(config.TIME_COL).reset_index(drop=True)


def dataset_summary(df: pd.DataFrame) -> dict:
    """Quick descriptive stats for logging / sanity checks."""
    n = len(df)
    n_fraud = int(df[config.TARGET].sum())
    return {
        "rows": n,
        "fraud": n_fraud,
        "fraud_rate": round(n_fraud / n, 5) if n else 0.0,
        "cards": int(df[config.CARD_COL].nunique()),
        "merchants": int(df[config.MERCHANT_COL].nunique()),
        "date_min": str(df[config.TIME_COL].min()),
        "date_max": str(df[config.TIME_COL].max()),
    }
