"""
Feature engineering — the analytical core of this project.

Raw Sparkov transactions are turned into signals a fraud model can learn from.
Every per-card feature is computed in strict time order and looks **only at the
past** (closed='left' rolling windows, shifted expanding stats). This prevents
target leakage: at scoring time you never know the current/future transactions.

Feature families
----------------
1. Transaction      — amount, log-amount
2. Temporal         — hour, day-of-week, night flag, weekend flag
3. Demographic      — cardholder age, city population
4. Geo              — haversine distance home→merchant, and from previous txn
5. Velocity         — rolling count / sum / mean of txns per card (1h/24h/7d)
6. Behavioral       — deviation of amount from the card's own past average,
                      time since previous txn, distinct merchants in 24h

The velocity + behavioral families are what catch real fraud: a stolen card
shows a burst of transactions, in new locations, deviating from normal spend.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

EARTH_RADIUS_KM = 6371.0088


# ── Geo ─────────────────────────────────────────────────────────────────────

def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorised great-circle distance in kilometres."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ── Feature builders (each returns the df with new columns) ─────────────────

def _add_temporal(df: pd.DataFrame) -> pd.DataFrame:
    t = df[config.TIME_COL].dt
    df["hour"] = t.hour
    df["day_of_week"] = t.dayofweek
    df["is_night"] = ((t.hour < 6) | (t.hour >= 22)).astype("int8")
    df["is_weekend"] = (t.dayofweek >= 5).astype("int8")
    return df


def _add_demographic(df: pd.DataFrame) -> pd.DataFrame:
    # Age at transaction time (years)
    age = (df[config.TIME_COL] - df["dob"]).dt.days / 365.25
    df["age"] = age.clip(lower=0, upper=120)
    df["city_pop_log"] = np.log1p(df["city_pop"].clip(lower=0))
    return df


def _add_amount(df: pd.DataFrame) -> pd.DataFrame:
    df["amt_log"] = np.log1p(df["amt"].clip(lower=0))
    return df


def _add_geo(df: pd.DataFrame) -> pd.DataFrame:
    # Distance between cardholder home and merchant location
    df["dist_home_merchant_km"] = haversine_km(
        df["lat"], df["long"], df["merch_lat"], df["merch_long"]
    )
    # Distance from the card's previous transaction (movement speed proxy)
    df = df.sort_values([config.CARD_COL, config.TIME_COL])
    prev_lat = df.groupby(config.CARD_COL)["merch_lat"].shift(1)
    prev_lon = df.groupby(config.CARD_COL)["merch_long"].shift(1)
    dist_prev = haversine_km(df["merch_lat"], df["merch_long"], prev_lat, prev_lon)
    df["dist_from_prev_txn_km"] = dist_prev.fillna(0.0)
    return df


def _add_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling per-card counts and sums over 1h / 24h / 7d, past-only."""
    df = df.sort_values([config.CARD_COL, config.TIME_COL]).reset_index(drop=True)

    for window, suffix in [("1h", "1h"), ("24h", "24h"), ("7d", "7d")]:
        roll = df.groupby(config.CARD_COL).rolling(
            window, on=config.TIME_COL, closed="left"
        )["amt"]
        cnt = roll.count().reset_index(level=0, drop=True)
        s = roll.sum().reset_index(level=0, drop=True)
        df[f"txn_count_{suffix}"] = cnt.fillna(0).astype("float32").values
        df[f"amt_sum_{suffix}"] = s.fillna(0).astype("float32").values

    # 24h mean amount (past)
    df["amt_mean_24h"] = (
        df["amt_sum_24h"] / df["txn_count_24h"].replace(0, np.nan)
    ).fillna(0.0).astype("float32")

    # Seconds since previous transaction
    secs = df.groupby(config.CARD_COL)[config.TIME_COL].diff().dt.total_seconds()
    df["secs_since_prev_txn"] = secs.fillna(-1.0).astype("float32")

    return df


def _add_behavioral(df: pd.DataFrame) -> pd.DataFrame:
    """Deviation of the current amount from the card's own past behaviour."""
    df = df.sort_values([config.CARD_COL, config.TIME_COL]).reset_index(drop=True)

    g = df.groupby(config.CARD_COL)["amt"]
    # Past mean via cumulative sums (vectorised, excludes current row)
    cumsum_prev = g.cumsum() - df["amt"]
    cumcount_prev = g.cumcount()  # number of strictly-previous txns
    past_mean = cumsum_prev / cumcount_prev.replace(0, np.nan)
    past_mean = past_mean.fillna(df["amt"])  # first txn: no history → neutral

    df["amt_dev_from_card_mean"] = (df["amt"] - past_mean).astype("float32")
    df["amt_ratio_to_card_mean"] = (
        df["amt"] / past_mean.replace(0, np.nan)
    ).fillna(1.0).clip(upper=1000).astype("float32")

    # Distinct merchants in the past 24h (rolling unique count)
    df["_merch_code"] = df[config.MERCHANT_COL].astype("category").cat.codes
    distinct = (
        df.groupby(config.CARD_COL)
        .rolling("24h", on=config.TIME_COL, closed="left")["_merch_code"]
        .apply(lambda s: s.nunique(), raw=False)
        .reset_index(level=0, drop=True)
    )
    df["distinct_merchants_24h"] = distinct.fillna(0).astype("float32").values
    df = df.drop(columns=["_merch_code"])
    return df


def engineer_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Full feature pipeline. Input: raw Sparkov rows. Output: a frame containing
    all engineered features in config.ALL_FEATURES plus identifiers + target.
    """
    df = df.copy()
    steps = [
        ("amount", _add_amount),
        ("temporal", _add_temporal),
        ("demographic", _add_demographic),
        ("geo", _add_geo),
        ("velocity", _add_velocity),
        ("behavioral", _add_behavioral),
    ]
    for name, fn in steps:
        df = fn(df)
        if verbose:
            print(f"[features]   {name} done")

    # Restore chronological order (important for downstream temporal split)
    df = df.sort_values(config.TIME_COL).reset_index(drop=True)

    keep = (
        config.ALL_FEATURES
        + [config.TARGET, config.CARD_COL, config.MERCHANT_COL, config.TIME_COL]
    )
    keep = [c for c in keep if c in df.columns]
    return df[keep]
