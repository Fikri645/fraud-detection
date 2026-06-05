"""Tests for the online feature store — must match batch semantics, no leakage."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.online import OnlineFeatureStore


def _txn(cc, amt, unix_time, merch="m", merch_lat=40.0, merch_long=-74.0):
    return {
        "cc_num": cc, "amt": amt, "unix_time": unix_time, "merchant": merch,
        "category": "shopping_net", "gender": "F", "state": "NY",
        "lat": 40.0, "long": -74.0, "merch_lat": merch_lat, "merch_long": merch_long,
        "city_pop": 1000, "dob": "1990-01-01",
    }


def test_first_txn_has_zero_velocity():
    store = OnlineFeatureStore()
    feats = store.transform(_txn(1, 100.0, 1_600_000_000))
    assert feats["txn_count_1h"] == 0
    assert feats["amt_sum_24h"] == 0
    assert feats["secs_since_prev_txn"] == -1.0
    assert feats["amt_ratio_to_card_mean"] == 1.0


def test_velocity_accumulates_after_update():
    store = OnlineFeatureStore()
    t0 = 1_600_000_000
    store.update(_txn(1, 100.0, t0))           # commit first txn
    feats = store.transform(_txn(1, 200.0, t0 + 600))  # 10 min later
    assert feats["txn_count_1h"] == 1
    assert feats["amt_sum_1h"] == 100.0
    assert feats["secs_since_prev_txn"] == 600.0


def test_window_expiry():
    store = OnlineFeatureStore()
    t0 = 1_600_000_000
    store.update(_txn(1, 100.0, t0))
    # 2 hours later — the prior txn is outside the 1h window but inside 24h
    feats = store.transform(_txn(1, 50.0, t0 + 7200))
    assert feats["txn_count_1h"] == 0
    assert feats["txn_count_24h"] == 1


def test_emits_all_model_features():
    store = OnlineFeatureStore()
    feats = store.transform(_txn(1, 100.0, 1_600_000_000))
    for col in config.ALL_FEATURES:
        assert col in feats, f"online store missing {col}"


def test_distinct_merchants():
    store = OnlineFeatureStore()
    t0 = 1_600_000_000
    store.update(_txn(1, 10.0, t0, merch="A"))
    store.update(_txn(1, 10.0, t0 + 100, merch="B"))
    feats = store.transform(_txn(1, 10.0, t0 + 200, merch="C"))
    assert feats["distinct_merchants_24h"] == 2
