"""Tests for evaluation metrics + business-cost threshold optimization."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import evaluate


def test_perfect_separation_pr_auc():
    y = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.9, 0.95])
    ev = evaluate.evaluate(y, scores)
    assert ev.pr_auc > 0.99
    assert ev.roc_auc == 1.0


def test_precision_at_k():
    y = np.array([1, 1, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    assert evaluate.precision_at_k(y, scores, 2) == 1.0


def test_recall_at_fraction():
    y = np.array([0] * 90 + [1] * 10)
    scores = np.concatenate([np.zeros(90), np.ones(10)])
    # Reviewing top 10% should catch all 10 frauds
    assert evaluate.recall_at_fraction(y, scores, 0.10) == 1.0


def test_expected_cost_asymmetry():
    y = np.array([1, 0])
    # Missing the fraud (pred 0 for true 1) vs blocking legit (pred 1 for true 0)
    miss = evaluate.expected_cost(y, np.array([0, 0]), c_fn=1.0, c_fp=5.0)
    block = evaluate.expected_cost(y, np.array([1, 1]), c_fn=1.0, c_fp=5.0)
    assert miss == 1.0       # one FN
    assert block == 5.0      # one FP


def test_optimal_threshold_beats_half():
    rng = np.random.default_rng(0)
    y = (rng.random(1000) < 0.05).astype(int)
    scores = np.clip(y * 0.5 + rng.random(1000) * 0.5, 0, 1)
    ev = evaluate.evaluate(y, scores)
    # Cost at optimal threshold should not exceed cost at naive 0.5
    assert ev.total_cost <= ev.cost_at_half + 1e-6
