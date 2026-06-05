"""
Evaluation utilities centred on the metrics that matter for imbalanced fraud.

Why not accuracy / ROC-AUC?
  At a 0.5% fraud rate, a model that predicts "never fraud" scores 99.5%
  accuracy and a deceptively high ROC-AUC. **PR-AUC (average precision)** is the
  honest summary: it focuses on the positive (fraud) class and collapses when
  the model can't separate the rare class.

Business-cost framing
  A fraud system is a cost-minimiser, not an accuracy-maximiser. Each decision
  carries an asymmetric cost: a missed fraud (false negative) loses money; a
  blocked legit customer (false positive) creates friction. We pick the decision
  threshold that minimises total expected cost, not the default 0.5.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score,
    precision_recall_curve, confusion_matrix,
)

from src import config


@dataclass
class EvalResult:
    pr_auc: float            # average precision — primary metric
    roc_auc: float
    f1_at_best: float
    best_threshold: float    # cost-optimal threshold
    precision_at_best: float
    recall_at_best: float
    precision_at_100: float  # precision in the 100 highest-risk txns
    recall_at_1pct: float    # recall if we review the riskiest 1% of txns
    total_cost: float        # expected cost at the cost-optimal threshold
    cost_at_half: float      # expected cost at naive threshold 0.5
    n: int
    n_fraud: int

    def to_dict(self) -> dict:
        return {k: (round(v, 5) if isinstance(v, float) else v)
                for k, v in asdict(self).items()}


def precision_at_k(y_true, y_score, k: int) -> float:
    """Precision among the k highest-scored transactions."""
    k = min(k, len(y_score))
    idx = np.argsort(y_score)[::-1][:k]
    return float(np.mean(np.asarray(y_true)[idx])) if k else 0.0


def recall_at_fraction(y_true, y_score, frac: float) -> float:
    """Recall achieved if analysts review the top `frac` of transactions."""
    y_true = np.asarray(y_true)
    k = max(1, int(len(y_score) * frac))
    idx = np.argsort(y_score)[::-1][:k]
    caught = y_true[idx].sum()
    total = y_true.sum()
    return float(caught / total) if total else 0.0


def expected_cost(y_true, y_pred, amounts=None,
                  c_fn: float = config.COST_FALSE_NEGATIVE,
                  c_fp: float = config.COST_FALSE_POSITIVE) -> float:
    """
    Total cost of a hard 0/1 decision.

    False negative (missed fraud): costs c_fn (optionally scaled by txn amount).
    False positive (blocked legit): costs c_fp per event (friction / goodwill).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    fn_mask = (y_true == 1) & (y_pred == 0)
    fp_mask = (y_true == 0) & (y_pred == 1)
    if amounts is not None:
        amounts = np.asarray(amounts)
        # Missed fraud loses the transaction value (normalised to cost units)
        fn_cost = (amounts[fn_mask].sum() / max(amounts.mean(), 1e-9)) * c_fn
    else:
        fn_cost = fn_mask.sum() * c_fn
    fp_cost = fp_mask.sum() * c_fp
    return float(fn_cost + fp_cost)


def optimal_threshold(y_true, y_score, amounts=None,
                      c_fn: float = config.COST_FALSE_NEGATIVE,
                      c_fp: float = config.COST_FALSE_POSITIVE):
    """Scan thresholds and return the one minimising expected cost."""
    # Candidate thresholds: a quantile grid of the scores, plus the naive 0.5
    # (guarantees the cost-optimal choice can never do worse than 0.5).
    thresholds = np.unique(np.concatenate([
        np.quantile(y_score, np.linspace(0.50, 0.9995, 200)),
        [0.5],
    ]))
    best_t, best_cost = 0.5, np.inf
    for t in thresholds:
        cost = expected_cost(y_true, (y_score >= t).astype(int), amounts, c_fn, c_fp)
        if cost < best_cost:
            best_cost, best_t = cost, t
    return float(best_t), float(best_cost)


def evaluate(y_true, y_score, amounts=None) -> EvalResult:
    """Full evaluation bundle at the cost-optimal decision threshold."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    pr_auc = average_precision_score(y_true, y_score)
    roc = roc_auc_score(y_true, y_score)

    best_t, best_cost = optimal_threshold(y_true, y_score, amounts)
    y_pred = (y_score >= best_t).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    cost_half = expected_cost(y_true, (y_score >= 0.5).astype(int), amounts)

    return EvalResult(
        pr_auc=float(pr_auc),
        roc_auc=float(roc),
        f1_at_best=float(f1_score(y_true, y_pred, zero_division=0)),
        best_threshold=best_t,
        precision_at_best=float(precision),
        recall_at_best=float(recall),
        precision_at_100=precision_at_k(y_true, y_score, 100),
        recall_at_1pct=recall_at_fraction(y_true, y_score, 0.01),
        total_cost=best_cost,
        cost_at_half=cost_half,
        n=int(len(y_true)),
        n_fraud=int(y_true.sum()),
    )


def pr_curve_points(y_true, y_score, max_points: int = 300):
    """Downsampled precision-recall curve for plotting."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    if len(precision) > max_points:
        idx = np.linspace(0, len(precision) - 1, max_points).astype(int)
        precision, recall = precision[idx], recall[idx]
    return precision.tolist(), recall.tolist()
