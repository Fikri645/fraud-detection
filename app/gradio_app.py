"""
Fraud Detection — interactive Gradio dashboard.

Tabs:
  1. Model Performance   — headline metrics, PR curve, imbalance study
  2. Live Scoring        — score a transaction in real time + SHAP explanation
  3. Explainability      — global SHAP feature importance
  4. Model Comparison    — LightGBM vs Autoencoder vs GNN
  5. Drift Monitoring    — PSI report (train period -> test period)
  6. Real-time Benchmark — streaming latency percentiles

Heavy results are pre-computed by the scripts and read from JSON; live scoring
loads the LightGBM model + online feature store on demand.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import gradio as gr

ROOT = Path(__file__).resolve().parents[1]  # NOTE: HF Space copies app to root -> use .parent there
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402

# ── Theme colors ────────────────────────────────────────────────────────────
DARK_BG, PANEL_BG = "#0f172a", "#1e293b"
TEAL, AMBER, RED, GREEN = "#2dd4bf", "#fbbf24", "#f87171", "#34d399"
TEXT_WHITE, GRID = "#f1f5f9", "#334155"


def _load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


META = _load_json(config.MODEL_META)
AE_META = _load_json(config.MODELS_DIR / "autoencoder_meta.json")
GNN_META = _load_json(config.MODELS_DIR / "gnn_meta.json")
DRIFT = _load_json(config.MODELS_DIR / "drift_report.json")
STREAM = _load_json(config.MODELS_DIR / "stream_benchmark.json")

# Lazy live-scoring singletons
_LIVE = {"model": None, "store": None, "explainer": None}


def _style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_WHITE, labelsize=9)
    for s in ax.spines.values():
        s.set_edgecolor(GRID)
    ax.grid(color=GRID, linewidth=0.5, alpha=0.5)
    if title:
        ax.set_title(title, color=TEXT_WHITE, fontsize=12, fontweight="bold", pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, color=TEXT_WHITE, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT_WHITE, fontsize=10)


def _fig(w=9, h=5):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=DARK_BG)
    return fig, ax


# ── Tab 1: Model Performance ────────────────────────────────────────────────

def performance_view():
    plt.close("all")
    if META is None:
        return None, None, "Run `python scripts/run_training.py` first."

    m = META["test_metrics"]

    # PR curve
    fig1, ax1 = _fig()
    pr = META["pr_curve"]
    ax1.plot(pr["recall"], pr["precision"], color=TEAL, linewidth=2.5)
    ax1.fill_between(pr["recall"], pr["precision"], alpha=0.15, color=TEAL)
    ax1.axhline(META["n_test"] and m["n_fraud"] / META["n_test"], color=RED,
                linestyle="--", alpha=0.7, label="Random baseline")
    _style(ax1, f"Precision-Recall Curve (PR-AUC = {m['pr_auc']:.3f})",
           "Recall", "Precision")
    ax1.legend(facecolor=PANEL_BG, edgecolor=GRID, labelcolor=TEXT_WHITE)
    ax1.set_ylim(0, 1.02)
    plt.tight_layout()

    # Imbalance study bar
    fig2, ax2 = _fig()
    study = META["imbalance_study"]
    names = [s["strategy"] for s in study]
    praucs = [s["pr_auc"] for s in study]
    times = [s["fit_seconds"] for s in study]
    colors = [GREEN if n == "cost_sensitive" else TEAL for n in names]
    bars = ax2.bar(names, praucs, color=colors, alpha=0.85, width=0.55)
    for b, pa, t in zip(bars, praucs, times):
        ax2.text(b.get_x() + b.get_width() / 2, pa + 0.005,
                 f"{pa:.3f}\n{t:.0f}s", ha="center", color=TEXT_WHITE,
                 fontsize=9, fontweight="bold")
    _style(ax2, "Imbalance Strategy Study (PR-AUC + fit time)",
           "", "Validation PR-AUC")
    ax2.set_ylim(0, max(praucs) * 1.15)
    plt.tight_layout()

    md = f"""
## Production Model — {META['model']}

| Metric | Value | What it means |
|:---|:---|:---|
| **PR-AUC** | **{m['pr_auc']:.4f}** | Primary metric for imbalanced fraud (avg precision) |
| ROC-AUC | {m['roc_auc']:.4f} | Ranking quality |
| Recall @ top 1% | {m['recall_at_1pct']:.1%} | Fraud caught if analysts review riskiest 1% |
| Precision @ top 100 | {m['precision_at_100']:.1%} | Hit rate in the 100 highest-risk txns |
| Cost-optimal threshold | {m['best_threshold']:.4f} | Minimises expected business cost (not 0.5) |
| Cost vs naive (0.5) | {m['total_cost']:.0f} vs {m['cost_at_half']:.0f} | Threshold tuning saves cost |

**Trained on** {META['n_train']:,} transactions · **tested on** {META['n_test']:,} (later period) ·
{META['n_features']} engineered features.

### Key finding — imbalance handling
The bar chart reproduces the 2025 industry finding: **cost-sensitive weighting matches
SMOTE on PR-AUC while training far faster**. SMOTE's extra compute buys nothing here —
which is why production fraud systems have largely abandoned it.
"""
    return fig1, fig2, md


# ── Tab 2: Live Scoring ─────────────────────────────────────────────────────

def _load_live():
    if _LIVE["model"] is None:
        import joblib
        from src.online import OnlineFeatureStore
        _LIVE["model"] = joblib.load(config.LGBM_MODEL)
        _LIVE["store"] = OnlineFeatureStore()
        try:
            import shap
            _LIVE["explainer"] = shap.TreeExplainer(_LIVE["model"])
        except Exception:
            _LIVE["explainer"] = None


def score_transaction(amt, category, hour, gender, state, home_lat, home_long,
                      merch_lat, merch_long, city_pop, age_years):
    if not config.LGBM_MODEL.exists():
        return "Run training first.", None
    _load_live()

    # Synthesize a transaction at the requested hour today
    import datetime as dt
    base = dt.datetime.now().replace(hour=int(hour), minute=0, second=0)
    unix_t = base.timestamp()
    dob = (base - dt.timedelta(days=int(age_years * 365.25))).strftime("%Y-%m-%d")

    txn = {
        "cc_num": 9999999999999999, "amt": float(amt), "unix_time": unix_t,
        "merchant": "demo_merchant", "category": category, "gender": gender,
        "state": state, "lat": float(home_lat), "long": float(home_long),
        "merch_lat": float(merch_lat), "merch_long": float(merch_long),
        "city_pop": float(city_pop), "dob": dob,
    }

    t0 = time.perf_counter()
    feats = _LIVE["store"].transform(txn)
    X = pd.DataFrame([feats])[config.ALL_FEATURES]
    for c in config.CATEGORICAL_FEATURES:
        X[c] = X[c].astype("category")
    prob = float(_LIVE["model"].predict_proba(X)[:, 1][0])
    latency = (time.perf_counter() - t0) * 1000

    thr = META["test_metrics"]["best_threshold"] if META else 0.5
    if prob >= thr:
        decision = "🔴 DECLINE"
    elif prob >= thr * 0.5:
        decision = "🟡 REVIEW"
    else:
        decision = "🟢 APPROVE"

    # SHAP explanation
    fig, ax = _fig(9, 4)
    if _LIVE["explainer"] is not None:
        sv = _LIVE["explainer"].shap_values(X)
        if isinstance(sv, list):
            sv = sv[1]
        row = np.asarray(sv)[0]
        order = np.argsort(np.abs(row))[::-1][:8]
        feats_n = [config.ALL_FEATURES[i] for i in order][::-1]
        vals = [row[i] for i in order][::-1]
        bcol = [RED if v > 0 else GREEN for v in vals]
        ax.barh(feats_n, vals, color=bcol, alpha=0.85)
        ax.axvline(0, color=TEXT_WHITE, alpha=0.3)
        _style(ax, "Why this decision? (SHAP — red pushes toward fraud)",
               "SHAP contribution", "")
    plt.tight_layout()

    md = f"""
### {decision}

| | |
|:---|:---|
| **Fraud probability** | **{prob:.2%}** |
| Decision threshold | {thr:.2%} |
| Scoring latency | {latency:.1f} ms |

Velocity features start at zero (fresh demo card). To see velocity signal,
score several transactions in a row — the store accumulates history.
"""
    return md, fig


# ── Tab 3: Explainability ───────────────────────────────────────────────────

def explainability_view():
    plt.close("all")
    if META is None:
        return None, "Run training first."
    imp = META["shap_importance"]
    feats = [d["feature"] for d in imp][::-1]
    vals = [d["value"] for d in imp][::-1]
    fig, ax = _fig(9, 7)
    ax.barh(feats, vals, color=TEAL, alpha=0.85)
    _style(ax, "Global Feature Importance (mean |SHAP|)", "mean |SHAP value|", "")
    plt.tight_layout()
    md = """
## What drives the model

These are the features the model relies on most, averaged over many transactions.
Velocity (`txn_count_*`, `amt_sum_*`), amount-deviation (`amt_ratio_to_card_mean`),
and amount itself dominate — exactly the signals a fraud analyst looks for: a
sudden burst of spending that deviates from the card's normal pattern.
"""
    return fig, md


# ── Tab 4: Model Comparison ─────────────────────────────────────────────────

def comparison_view():
    plt.close("all")
    rows = []
    if META:
        rows.append(("LightGBM (supervised)", META["test_metrics"]["pr_auc"],
                     META["test_metrics"]["roc_auc"]))
    if GNN_META:
        rows.append(("GraphSAGE (GNN)", GNN_META["test_metrics"]["pr_auc"],
                     GNN_META["test_metrics"]["roc_auc"]))
    if AE_META:
        rows.append(("Autoencoder (unsupervised)", AE_META["test_metrics"]["pr_auc"],
                     AE_META["test_metrics"]["roc_auc"]))
    if not rows:
        return None, "Train the models first."

    fig, ax = _fig(9, 5)
    names = [r[0] for r in rows]
    prs = [r[1] for r in rows]
    bars = ax.bar(names, prs, color=[GREEN, TEAL, AMBER][:len(rows)], alpha=0.85, width=0.5)
    for b, p in zip(bars, prs):
        ax.text(b.get_x() + b.get_width() / 2, p + 0.01, f"{p:.3f}",
                ha="center", color=TEXT_WHITE, fontweight="bold")
    _style(ax, "Model Comparison — Test PR-AUC", "", "PR-AUC")
    ax.set_ylim(0, 1.0)
    plt.setp(ax.get_xticklabels(), rotation=10)
    plt.tight_layout()

    md = "## Three approaches, three trade-offs\n\n| Model | PR-AUC | ROC-AUC | When to use |\n|:---|:---|:---|:---|\n"
    notes = {
        "LightGBM (supervised)": "Best accuracy when fraud labels exist. The workhorse.",
        "GraphSAGE (GNN)": "Adds relational signal from the card's transaction chain.",
        "Autoencoder (unsupervised)": "No labels needed — catches novel fraud patterns.",
    }
    for name, pr, roc in rows:
        md += f"| {name} | {pr:.4f} | {roc:.4f} | {notes.get(name, '')} |\n"
    md += ("\n> Supervised gradient boosting typically wins on labelled benchmarks; "
           "the autoencoder is the safety net for fraud the labels haven't caught up to yet.")
    return fig, md


# ── Tab 5: Drift ────────────────────────────────────────────────────────────

def drift_view():
    plt.close("all")
    if DRIFT is None:
        return None, "Run `python scripts/run_drift.py` first."
    rep = DRIFT["feature_psi"][:12]
    feats = [r["feature"] for r in rep][::-1]
    vals = [r["psi"] for r in rep][::-1]
    cols = [RED if v >= 0.25 else (AMBER if v >= 0.1 else TEAL) for v in vals]
    fig, ax = _fig(9, 6)
    ax.barh(feats, vals, color=cols, alpha=0.85)
    ax.axvline(0.1, color=AMBER, linestyle="--", alpha=0.6, label="moderate (0.10)")
    ax.axvline(0.25, color=RED, linestyle="--", alpha=0.6, label="significant (0.25)")
    _style(ax, "Feature Drift: train period -> test period (PSI)", "PSI", "")
    ax.legend(facecolor=PANEL_BG, edgecolor=GRID, labelcolor=TEXT_WHITE)
    plt.tight_layout()

    score_line = ""
    if DRIFT.get("score_psi") is not None:
        score_line = (f"\n**Model-score PSI = {DRIFT['score_psi']:.4f} "
                      f"({DRIFT['score_psi_status']})** — the single most useful production monitor.")
    md = f"""
## Concept Drift Monitoring (PSI)

Fraud is adversarial — patterns shift, and a stale model decays silently.
PSI is a **label-free** early warning: it compares feature distributions between
the training period and the live period, no fraud labels required.
{score_line}

| PSI | Interpretation |
|:---|:---|
| < 0.10 | Stable |
| 0.10 – 0.25 | Moderate shift — investigate |
| > 0.25 | Significant shift — retrain |
"""
    return fig, md


# ── Tab 6: Real-time benchmark ──────────────────────────────────────────────

def stream_view():
    if STREAM is None:
        return "Run `python streaming/simulate_stream.py` first."
    lat = STREAM["latency_ms"]
    return f"""
## Real-Time Streaming Benchmark

The test period was replayed transaction-by-transaction through the online
feature store + model, exactly as a production consumer would process a live
stream.

| Metric | Value |
|:---|:---|
| Transactions streamed | {STREAM['transactions']:,} |
| Throughput | {STREAM['throughput_per_sec']:,.0f} txn/sec |
| **Latency P50** | **{lat['p50']} ms** |
| Latency P95 | {lat['p95']} ms |
| Latency P99 | {lat['p99']} ms |
| Latency max | {lat['max']} ms |
| Fraud caught | {STREAM['fraud_caught']} / {STREAM['fraud_in_stream']} ({STREAM['catch_rate']:.0%}) |
| Decline rate | {STREAM['decline_rate']:.2%} |

Velocity features are maintained incrementally in memory — no batch recompute —
which is what keeps per-transaction latency in the single-digit-millisecond range,
well inside the sub-100ms budget real payment systems require.
"""


# ── Layout ──────────────────────────────────────────────────────────────────

_DESC = """
# 🛡️ Real-Time Credit Card Fraud Detection

End-to-end fraud system on the **Sparkov** dataset (1.85M transactions, ~0.5% fraud):
rich feature engineering, an honest imbalance study, three modelling approaches
(**LightGBM · GraphSAGE GNN · Autoencoder**), SHAP explainability, concept-drift
monitoring, and a real-time scoring service.
"""

with gr.Blocks(title="Fraud Detection",
               theme=gr.themes.Base(primary_hue="teal", secondary_hue="cyan",
                                    neutral_hue="slate")) as demo:
    gr.Markdown(_DESC)

    with gr.Tab("1. Model Performance"):
        b1 = gr.Button("Load results", variant="primary")
        with gr.Row():
            p1 = gr.Plot()
            p2 = gr.Plot()
        md1 = gr.Markdown()
        b1.click(performance_view, outputs=[p1, p2, md1], scroll_to_output=False)
        demo.load(performance_view, outputs=[p1, p2, md1])

    with gr.Tab("2. Live Scoring"):
        gr.Markdown("### Score a transaction in real time\nAdjust the inputs and click Score. "
                    "Score several in a row to build up the card's velocity history.")
        with gr.Row():
            with gr.Column():
                amt = gr.Slider(1, 5000, value=850, label="Amount ($)")
                category = gr.Dropdown(
                    ["shopping_net", "grocery_pos", "gas_transport", "misc_net",
                     "shopping_pos", "entertainment", "food_dining", "health_fitness",
                     "travel", "kids_pets", "home", "personal_care"],
                    value="shopping_net", label="Category")
                hour = gr.Slider(0, 23, value=2, step=1, label="Hour of day (0-23)")
                gender = gr.Dropdown(["F", "M"], value="F", label="Gender")
                state = gr.Dropdown(["NY", "CA", "TX", "FL", "PA", "OH", "IL"],
                                    value="NY", label="State")
            with gr.Column():
                home_lat = gr.Number(value=40.71, label="Cardholder home lat")
                home_long = gr.Number(value=-74.0, label="Cardholder home long")
                merch_lat = gr.Number(value=36.0, label="Merchant lat")
                merch_long = gr.Number(value=-90.0, label="Merchant long")
                city_pop = gr.Number(value=1000000, label="City population")
                age_years = gr.Slider(18, 90, value=35, step=1, label="Cardholder age")
        sbtn = gr.Button("Score Transaction", variant="primary")
        smd = gr.Markdown()
        splot = gr.Plot()
        sbtn.click(score_transaction,
                   inputs=[amt, category, hour, gender, state, home_lat, home_long,
                           merch_lat, merch_long, city_pop, age_years],
                   outputs=[smd, splot], scroll_to_output=False)

    with gr.Tab("3. Explainability"):
        b3 = gr.Button("Load SHAP", variant="primary")
        p3 = gr.Plot()
        md3 = gr.Markdown()
        b3.click(explainability_view, outputs=[p3, md3], scroll_to_output=False)
        demo.load(explainability_view, outputs=[p3, md3])

    with gr.Tab("4. Model Comparison"):
        b4 = gr.Button("Load comparison", variant="primary")
        p4 = gr.Plot()
        md4 = gr.Markdown()
        b4.click(comparison_view, outputs=[p4, md4], scroll_to_output=False)
        demo.load(comparison_view, outputs=[p4, md4])

    with gr.Tab("5. Drift Monitoring"):
        b5 = gr.Button("Load drift report", variant="primary")
        p5 = gr.Plot()
        md5 = gr.Markdown()
        b5.click(drift_view, outputs=[p5, md5], scroll_to_output=False)
        demo.load(drift_view, outputs=[p5, md5])

    with gr.Tab("6. Real-time Benchmark"):
        b6 = gr.Button("Load benchmark", variant="primary")
        md6 = gr.Markdown()
        b6.click(stream_view, outputs=[md6], scroll_to_output=False)
        demo.load(stream_view, outputs=[md6])

    gr.Markdown(
        "---\nBuilt by [Muhammad Fikri Wahidin](https://github.com/Fikri645) · "
        "Sparkov dataset · LightGBM · PyTorch Geometric · SHAP · FastAPI")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
