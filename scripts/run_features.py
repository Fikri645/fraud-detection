"""
Build engineered feature tables for both splits and save as parquet.

    python scripts/run_features.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, data, features  # noqa: E402


def main():
    config.DATA_PROC.mkdir(parents=True, exist_ok=True)
    for which, out in [("train", config.FEATURES_TRAIN), ("test", config.FEATURES_TEST)]:
        print(f"\n=== {which} ===")
        df = data.load_raw(which)
        print(f"[features] {len(df):,} raw rows")
        t0 = time.time()
        feat = features.engineer_features(df)
        print(f"[features] done in {time.time() - t0:.1f}s -> {feat.shape}")
        feat.to_parquet(out, index=False)
        print(f"[features] saved {out}")


if __name__ == "__main__":
    main()
