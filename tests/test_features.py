"""Tests for feature engineering — correctness and no-leakage guarantees."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, features


def _toy_df():
    """Two cards, a handful of transactions in known time order."""
    rows = [
        # card 1
        ("2020-01-01 10:00:00", 1, "m_a", "grocery_pos", 50.0, "F", "NY", 40.0, -74.0, 40.1, -74.1, 1000, "1990-01-01"),
        ("2020-01-01 10:30:00", 1, "m_b", "shopping_net", 500.0, "F", "NY", 40.0, -74.0, 41.0, -75.0, 1000, "1990-01-01"),
        ("2020-01-02 12:00:00", 1, "m_a", "grocery_pos", 60.0, "F", "NY", 40.0, -74.0, 40.1, -74.1, 1000, "1990-01-01"),
        # card 2
        ("2020-01-01 09:00:00", 2, "m_c", "gas_transport", 30.0, "M", "CA", 34.0, -118.0, 34.1, -118.1, 5000, "1985-06-15"),
        ("2020-01-01 23:00:00", 2, "m_d", "misc_net", 900.0, "M", "CA", 34.0, -118.0, 36.0, -120.0, 5000, "1985-06-15"),
    ]
    cols = ["trans_date_trans_time", "cc_num", "merchant", "category", "amt",
            "gender", "state", "lat", "long", "merch_lat", "merch_long", "city_pop", "dob"]
    df = pd.DataFrame(rows, columns=cols)
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["dob"] = pd.to_datetime(df["dob"])
    df["is_fraud"] = [0, 1, 0, 0, 1]
    return df


def test_haversine_zero_distance():
    assert features.haversine_km(40.0, -74.0, 40.0, -74.0) == 0.0


def test_haversine_known_distance():
    # NYC to LA ~ 3935 km
    d = features.haversine_km(40.71, -74.0, 34.05, -118.24)
    assert 3800 < d < 4000


def test_output_has_all_features():
    feat = features.engineer_features(_toy_df(), verbose=False)
    for col in config.ALL_FEATURES:
        assert col in feat.columns, f"missing {col}"


def test_no_nan_or_inf():
    feat = features.engineer_features(_toy_df(), verbose=False)
    num = feat[config.NUMERIC_FEATURES]
    assert num.isna().sum().sum() == 0
    assert np.isinf(num.values).sum() == 0


def test_velocity_is_past_only():
    """First transaction of each card must have zero prior velocity (no leakage)."""
    feat = features.engineer_features(_toy_df(), verbose=False)
    feat = feat.sort_values(["cc_num", "trans_date_trans_time"])
    first_per_card = feat.groupby("cc_num").head(1)
    assert (first_per_card["txn_count_1h"] == 0).all()
    assert (first_per_card["txn_count_24h"] == 0).all()
    assert (first_per_card["amt_sum_24h"] == 0).all()


def test_velocity_counts_accumulate():
    """Card 1's 2nd txn (30 min later) should see exactly 1 prior txn in 1h."""
    feat = features.engineer_features(_toy_df(), verbose=False)
    feat = feat.sort_values(["cc_num", "trans_date_trans_time"]).reset_index(drop=True)
    card1 = feat[feat["cc_num"] == 1].reset_index(drop=True)
    assert card1.loc[1, "txn_count_1h"] == 1
    assert card1.loc[1, "amt_sum_1h"] == 50.0


def test_amount_ratio_neutral_on_first_txn():
    feat = features.engineer_features(_toy_df(), verbose=False)
    feat = feat.sort_values(["cc_num", "trans_date_trans_time"])
    first = feat.groupby("cc_num").head(1)
    # First txn has no history -> ratio defaults to 1.0
    assert np.allclose(first["amt_ratio_to_card_mean"].values, 1.0)
