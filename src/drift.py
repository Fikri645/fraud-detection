"""
Concept-drift monitoring via the Population Stability Index (PSI).

Fraud is adversarial: attack patterns evolve, so a model trained on last
quarter's data silently decays. PSI is the industry-standard, **label-free**
early-warning signal — it compares the distribution of a feature (or the model
score) between a reference window and a recent window, with no need to wait for
fraud labels to arrive.

    PSI < 0.10  → stable
    0.10–0.25   → moderate shift, investigate
    PSI > 0.25  → significant shift, retrain

Computing PSI on the model's *output score* is the single most useful monitor:
it catches both data drift and concept drift in one number.
"""
from __future__ import annotations

import numpy as np

from src import config


def psi(reference, current, bins: int = config.PSI_BINS) -> float:
    """
    Population Stability Index between two 1-D samples.
    Bin edges are quantiles of the reference distribution.
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    # Quantile bin edges from reference; widen the outer edges to catch tails
    edges = np.quantile(reference, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:  # near-constant feature
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    ref_pct = np.histogram(reference, bins=edges)[0] / len(reference)
    cur_pct = np.histogram(current, bins=edges)[0] / len(current)

    # Laplace smoothing to avoid log(0)
    eps = 1e-6
    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def classify_psi(value: float) -> str:
    if value < config.PSI_THRESHOLD_WARN:
        return "stable"
    if value < config.PSI_THRESHOLD_ALERT:
        return "moderate"
    return "significant"


def feature_drift_report(ref_df, cur_df, features) -> list[dict]:
    """PSI per feature between two periods, sorted by severity."""
    rows = []
    for f in features:
        if f not in ref_df.columns or f not in cur_df.columns:
            continue
        val = psi(ref_df[f].values, cur_df[f].values)
        rows.append({"feature": f, "psi": round(val, 4), "status": classify_psi(val)})
    rows.sort(key=lambda r: r["psi"], reverse=True)
    return rows
