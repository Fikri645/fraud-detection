"""Tests for PSI concept-drift detection."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import drift


def test_psi_identical_is_zero():
    rng = np.random.default_rng(0)
    x = rng.normal(size=10000)
    assert drift.psi(x, x.copy()) < 0.01


def test_psi_shifted_is_large():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 10000)
    cur = rng.normal(3, 1, 10000)  # big mean shift
    assert drift.psi(ref, cur) > 0.25


def test_classify_psi_bands():
    assert drift.classify_psi(0.05) == "stable"
    assert drift.classify_psi(0.15) == "moderate"
    assert drift.classify_psi(0.40) == "significant"


def test_psi_constant_feature_safe():
    ref = np.ones(100)
    cur = np.ones(100)
    # Near-constant feature must not crash or return nan
    val = drift.psi(ref, cur)
    assert np.isfinite(val)
