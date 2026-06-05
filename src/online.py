"""
Online feature store for real-time scoring.

Batch feature engineering (src/features.py) recomputes rolling windows over a
whole DataFrame. That's impossible at serving time — when a single transaction
arrives you have milliseconds and only the card's recent history. This module
maintains a compact in-memory state per card (recent timestamps, amounts, last
location, running mean) and derives the SAME engineered features incrementally.

This is the piece that turns an offline notebook model into a deployable
fraud service. The features it emits are column-compatible with the batch
pipeline, so the exact same trained model scores them.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np

from src import config
from src.features import haversine_km

_WINDOWS = {"1h": 3600, "24h": 86400, "7d": 604800}


@dataclass
class CardState:
    times: deque = field(default_factory=lambda: deque())          # unix seconds
    amts: deque = field(default_factory=lambda: deque())           # aligned amounts
    merch: deque = field(default_factory=lambda: deque())          # aligned merchant ids
    last_time: float = None
    last_merch_lat: float = None
    last_merch_long: float = None
    sum_amt: float = 0.0
    count: int = 0

    def prune(self, now: float, horizon: int = 604800):
        """Drop events older than the largest window (7d)."""
        while self.times and now - self.times[0] > horizon:
            self.times.popleft()
            self.amts.popleft()
            self.merch.popleft()


class OnlineFeatureStore:
    """Incremental per-card feature computation for single transactions."""

    def __init__(self):
        self._state: dict = defaultdict(CardState)

    def transform(self, txn: dict) -> dict:
        """
        Given a raw transaction dict, return the engineered feature row
        (looking only at the card's PAST). Does NOT mutate state — call
        `update` after you've scored, to mirror production ordering.
        """
        cc = txn[config.CARD_COL]
        st = self._state[cc]
        now = float(txn["unix_time"])
        st.prune(now)

        amt = float(txn["amt"])
        feats = {}

        # Transaction + temporal
        import datetime as _dt
        ts = _dt.datetime.fromtimestamp(now)
        feats["amt"] = amt
        feats["amt_log"] = float(np.log1p(max(amt, 0)))
        feats["hour"] = ts.hour
        feats["day_of_week"] = ts.weekday()
        feats["is_night"] = int(ts.hour < 6 or ts.hour >= 22)
        feats["is_weekend"] = int(ts.weekday() >= 5)

        # Demographic
        age = (now - _to_unix(txn["dob"])) / (365.25 * 86400)
        feats["age"] = float(np.clip(age, 0, 120))
        feats["city_pop_log"] = float(np.log1p(max(float(txn.get("city_pop", 0)), 0)))

        # Geo
        feats["dist_home_merchant_km"] = float(haversine_km(
            txn["lat"], txn["long"], txn["merch_lat"], txn["merch_long"]))
        if st.last_merch_lat is not None:
            feats["dist_from_prev_txn_km"] = float(haversine_km(
                txn["merch_lat"], txn["merch_long"], st.last_merch_lat, st.last_merch_long))
        else:
            feats["dist_from_prev_txn_km"] = 0.0

        # Velocity (past only)
        t_arr = np.array(st.times)
        a_arr = np.array(st.amts)
        for suffix, secs in _WINDOWS.items():
            if len(t_arr):
                mask = (now - t_arr) <= secs
                feats[f"txn_count_{suffix}"] = float(mask.sum())
                feats[f"amt_sum_{suffix}"] = float(a_arr[mask].sum())
            else:
                feats[f"txn_count_{suffix}"] = 0.0
                feats[f"amt_sum_{suffix}"] = 0.0
        feats["amt_mean_24h"] = (
            feats["amt_sum_24h"] / feats["txn_count_24h"] if feats["txn_count_24h"] else 0.0)
        feats["secs_since_prev_txn"] = float(now - st.last_time) if st.last_time else -1.0

        # Behavioral
        past_mean = st.sum_amt / st.count if st.count else amt
        feats["amt_dev_from_card_mean"] = float(amt - past_mean)
        feats["amt_ratio_to_card_mean"] = float(min(amt / past_mean, 1000) if past_mean else 1.0)
        if len(t_arr):
            mask24 = (now - t_arr) <= 86400
            feats["distinct_merchants_24h"] = float(len(set(np.array(st.merch)[mask24])))
        else:
            feats["distinct_merchants_24h"] = 0.0

        # Categoricals (passed through)
        feats["category"] = txn.get("category", "")
        feats["gender"] = txn.get("gender", "")
        feats["state"] = txn.get("state", "")
        return feats

    def update(self, txn: dict):
        """Commit this transaction to the card's state (after scoring)."""
        cc = txn[config.CARD_COL]
        st = self._state[cc]
        now = float(txn["unix_time"])
        amt = float(txn["amt"])
        st.times.append(now)
        st.amts.append(amt)
        st.merch.append(txn.get(config.MERCHANT_COL, ""))
        st.last_time = now
        st.last_merch_lat = txn["merch_lat"]
        st.last_merch_long = txn["merch_long"]
        st.sum_amt += amt
        st.count += 1


import datetime as _dt
_EPOCH = _dt.datetime(1970, 1, 1)


def _to_unix(value) -> float:
    """
    Accept a unix float, ISO string, or date string for dob.
    Uses an explicit epoch difference (not .timestamp()) so pre-1970 dates —
    common for cardholder dob — work on Windows, where .timestamp() raises
    OSError for negative values.
    """
    if isinstance(value, (int, float)):
        return float(value)
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S"):
        try:
            return (_dt.datetime.strptime(str(value), fmt) - _EPOCH).total_seconds()
        except ValueError:
            continue
    import pandas as pd
    return (pd.Timestamp(value).to_pydatetime() - _EPOCH).total_seconds()
