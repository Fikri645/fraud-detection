"""
Explainability via SHAP — a compliance requirement, not a nice-to-have.

Fraud models in finance fall under model-risk-management regimes (e.g. SR 11-7)
that demand every automated decline be explainable. SHAP gives both:
  - global feature importance (which signals drive fraud overall)
  - per-transaction attributions (why THIS txn was flagged) for adverse-action
    notices and analyst review.

TreeExplainer is exact and fast for LightGBM/XGBoost.
"""
from __future__ import annotations

import numpy as np


def compute_shap(model, X_sample):
    """Return (shap_values_for_positive_class, explainer)."""
    import shap
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_sample)
    # Newer SHAP returns a single array for binary LightGBM; older returns a list
    if isinstance(sv, list):
        sv = sv[1]
    return np.asarray(sv), explainer


def global_importance(shap_values, feature_names, top_k: int = 20):
    """Mean absolute SHAP per feature, sorted descending."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:top_k]
    return [(feature_names[i], float(mean_abs[i])) for i in order]


def explain_transaction(shap_values, feature_names, row_idx: int, top_k: int = 6):
    """Top contributing features (signed) for a single transaction."""
    row = shap_values[row_idx]
    order = np.argsort(np.abs(row))[::-1][:top_k]
    return [(feature_names[i], float(row[i])) for i in order]
