"""Tests for the imbalance-handling strategies in src/train.py."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train import _apply_strategy


def _imbalanced_data(n=2000, pos=40, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    y = np.zeros(n, dtype=int)
    y[:pos] = 1
    rng.shuffle(y)
    return X, y


def test_none_leaves_data_untouched():
    X, y = _imbalanced_data()
    Xs, ys, spw = _apply_strategy(X, y, "none")
    assert len(ys) == len(y)
    assert spw == 1.0


def test_cost_sensitive_sets_pos_weight():
    X, y = _imbalanced_data(n=2000, pos=40)
    Xs, ys, spw = _apply_strategy(X, y, "cost_sensitive")
    assert len(ys) == len(y)            # no resampling
    assert spw == (2000 - 40) / 40      # n_neg / n_pos


def test_undersample_shrinks_majority():
    X, y = _imbalanced_data(n=2000, pos=40)
    Xs, ys, spw = _apply_strategy(X, y, "undersample")
    assert len(ys) < len(y)             # data was reduced
    # minority preserved, majority cut toward the 0.1 ratio
    assert ys.sum() == 40
    assert spw == 1.0


def test_smote_grows_minority():
    X, y = _imbalanced_data(n=2000, pos=40)
    Xs, ys, spw = _apply_strategy(X, y, "smote")
    assert ys.sum() > 40                # synthetic minority added
    assert len(ys) > len(y)


def test_unknown_strategy_raises():
    X, y = _imbalanced_data()
    try:
        _apply_strategy(X, y, "bogus")
        assert False, "should have raised"
    except ValueError:
        pass
