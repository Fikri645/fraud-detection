"""
Turn engineered feature tables into model-ready matrices.

Gradient boosting (LightGBM) handles categoricals natively, so the default path
keeps `category`/`gender`/`state` as pandas categoricals. For models that need
purely numeric input (autoencoder, GNN, SMOTE), we one-hot / ordinal encode.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


def load_features(which: str = "train") -> pd.DataFrame:
    path = config.FEATURES_TRAIN if which == "train" else config.FEATURES_TEST
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run scripts/run_features.py first.")
    return pd.read_parquet(path)


def split_xy(df: pd.DataFrame, categorical: bool = True):
    """Return (X, y) with categoricals typed for LightGBM if requested."""
    X = df[config.ALL_FEATURES].copy()
    y = df[config.TARGET].astype(int).values
    if categorical:
        for c in config.CATEGORICAL_FEATURES:
            if c in X.columns:
                X[c] = X[c].astype("category")
    return X, y


def time_validation_split(df: pd.DataFrame, valid_fraction: float = config.VALID_FRACTION):
    """
    Hold out the last `valid_fraction` of the (already time-sorted) training
    frame for early stopping. No shuffling — mirrors real deployment where you
    train on the past and validate on the more-recent past.
    """
    n = len(df)
    cut = int(n * (1 - valid_fraction))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def encode_numeric_matrix(df_train: pd.DataFrame, df_other: pd.DataFrame = None):
    """
    Fully-numeric encoding for AE / SMOTE / GNN node features:
    numeric columns passed through (standardised), categoricals one-hot encoded.
    Returns (X_train, [X_other], encoder_state) where encoder_state lets you
    re-apply the exact same transform at inference time.
    """
    from sklearn.preprocessing import StandardScaler

    num = config.NUMERIC_FEATURES
    cat = [c for c in config.CATEGORICAL_FEATURES if c in df_train.columns]

    # One-hot via pandas, aligned columns across splits
    tr_oh = pd.get_dummies(df_train[cat].astype(str), prefix=cat)
    scaler = StandardScaler()
    tr_num = pd.DataFrame(scaler.fit_transform(df_train[num]), columns=num, index=df_train.index)
    X_train = pd.concat([tr_num, tr_oh.set_index(df_train.index)], axis=1)

    state = {"scaler": scaler, "columns": list(X_train.columns), "num": num, "cat": cat}

    if df_other is not None:
        ot_oh = pd.get_dummies(df_other[cat].astype(str), prefix=cat)
        ot_num = pd.DataFrame(scaler.transform(df_other[num]), columns=num, index=df_other.index)
        X_other = pd.concat([ot_num, ot_oh.set_index(df_other.index)], axis=1)
        X_other = X_other.reindex(columns=X_train.columns, fill_value=0.0)
        return X_train.astype(np.float32), X_other.astype(np.float32), state

    return X_train.astype(np.float32), state
