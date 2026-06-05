# 🛡️ Real-Time Credit Card Fraud Detection

[![CI](https://github.com/Fikri645/fraud-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/Fikri645/fraud-detection/actions)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-HuggingFace%20Spaces-orange)](https://huggingface.co/spaces/fikri0o0/fraud-detection)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)

> An end-to-end, production-shaped fraud detection system: rich feature
> engineering, an honest class-imbalance study, three modelling paradigms
> (gradient boosting · graph neural network · autoencoder), SHAP explainability
> for compliance, concept-drift monitoring, and a real-time scoring service.

**[Live Demo →](https://huggingface.co/spaces/fikri0o0/fraud-detection)**

---

## Why this project

Fraud detection is the canonical hard problem in applied ML and the bread-and-butter
of data science at any fintech (GoPay, OVO, Kredivo, Dana). It forces you to confront
the issues tutorials skip:

- **Extreme class imbalance** (~0.5% fraud) — accuracy is meaningless
- **Asymmetric costs** — a missed fraud and a blocked customer are not equal
- **Adversarial drift** — attack patterns change; a static model decays
- **Latency** — decisions must happen in milliseconds, mid-transaction
- **Explainability** — regulators require every automated decline to be justified

This project addresses each one explicitly rather than stopping at a notebook with an
inflated accuracy score.

---

## Dataset — Sparkov

Simulated but realistic credit-card transactions ([Kaggle: kartik2112/fraud-detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection)).

| | |
|:---|:---|
| Transactions | **1.85M** (1.30M train · 0.56M test) |
| Fraud rate | ~0.5% (realistic extreme imbalance) |
| Split | **Temporal** — train = earlier period, test = later (no leakage) |
| Cards / Merchants | 983 / 693 |
| Period | Jan 2019 – Dec 2020 |

Unlike the over-used ULB `creditcard.csv` (PCA-anonymised `V1–V28`), Sparkov keeps
**human-readable columns** — merchant, category, geo-coordinates, timestamps — which
is what makes meaningful feature engineering and a transaction graph possible.

---

## Feature Engineering — the analytical core

All per-card features are computed in strict time order, looking **only at the past**
(`closed='left'` rolling windows, shifted expanding stats) — zero target leakage.

| Family | Features | Catches |
|:---|:---|:---|
| Transaction | amount, log-amount | Large-value fraud |
| Temporal | hour, day-of-week, is_night, is_weekend | Fraud clusters at night |
| Demographic | cardholder age, city population | Population priors |
| **Geo** | haversine distance home↔merchant, distance from previous txn | Impossible-travel |
| **Velocity** | rolling count/sum/mean per card over 1h / 24h / 7d | Transaction bursts |
| **Behavioral** | deviation & ratio vs card's own past mean, secs since last txn, distinct merchants 24h | Out-of-pattern spend |

Signal check (test period, fraud vs legit means):

| Feature | Legit | Fraud | Ratio |
|:---|---:|---:|---:|
| `amt` | 67.6 | 528.4 | **7.8×** |
| `amt_ratio_to_card_mean` | 1.02 | 6.52 | **6.4×** |
| `txn_count_1h` | 0.22 | 0.67 | **3.0×** |
| `is_night` | 0.30 | 0.86 | **2.9×** |

---

## Key Results

### Production model (LightGBM, tested on the later period)

| Metric | Value | What it means |
|:---|:---|:---|
| **PR-AUC** | **0.967** | Primary metric for imbalanced fraud (average precision) |
| ROC-AUC | 0.999 | Near-perfect ranking |
| **Recall @ top 1%** | **98.0%** | Reviewing the riskiest 1% of txns catches 98% of fraud |
| **Precision @ top 100** | **100%** | The 100 highest-risk transactions are *all* fraud |
| Cost-optimal threshold | **0.019** | Minimises expected business cost — **20% cheaper** than the naive 0.5 |

The cost-optimal threshold (0.019, well below 0.5) reflects realistic fraud economics:
a missed fraud loses the transaction amount, so it is worth accepting more false
positives to catch more fraud. At that threshold the model catches **96.6%** of fraud.

### Model comparison (test PR-AUC)

| Model | PR-AUC | ROC-AUC | Role |
|:---|:---|:---|:---|
| **LightGBM** (cost-sensitive, Optuna) | **0.967** | 0.999 | Production workhorse |
| GraphSAGE (GNN, directed card+merchant graph) | 0.368 | 0.985 | Relational burst signal |
| Autoencoder (unsupervised) | 0.135 | 0.866 | Label-free novel-fraud net |

> The GNN uses **strictly directed past→present edges** (a transaction can only
> attend to the card's / merchant's earlier transactions) — no temporal leakage.

> Honest outcome: gradient boosting dominates tabular fraud. The GNN adds
> relational context (and beats the unsupervised baseline), while the autoencoder
> — trained with **no fraud labels at all** — still ranks fraud far above random,
> which is exactly its job as a safety net for novel attacks.

### Imbalance study — the headline finding

An apples-to-apples study (same LightGBM, same matrix, only the sampling varies)
reproduces the 2025 industry consensus:

| Strategy | PR-AUC | Fit time | Train rows | Verdict |
|:---|:---|:---|:---|:---|
| None | 0.682 | 11.7s | 1.10M | Baseline — imbalance cripples it |
| **Cost-sensitive** (`scale_pos_weight`) | **0.980** | 13.0s | 1.10M | **Best ROI** |
| SMOTE | 0.982 | 28.8s | 1.21M | +0.002 PR-AUC for 2.2× the time |
| Undersample | 0.981 | 4.1s | 0.07M | Fast, but discards 94% of data |

> **Cost-sensitive weighting matches SMOTE on PR-AUC at a fraction of the compute**
> (+0.002 PR-AUC is within noise, for 2.2× the runtime). A 2025 review of 821 papers
> found only 6% of scale-focused work uses SMOTE successfully — production has largely
> abandoned it. This project shows why.

### Real-time scoring

| Metric | Value |
|:---|:---|
| Latency **P50** | **7.1 ms** |
| Latency P95 / P99 | 11.2 / 13.0 ms |
| Throughput (single thread) | ~132 txn/sec |

Velocity features are maintained incrementally in an in-memory online store, so
per-transaction scoring stays in the single-digit-millisecond range — well inside
the sub-100ms budget real payment systems require. Replaying the test stream at
the cost-optimal threshold catches **85% of fraud** at a 1.8% decline rate.

### Cross-dataset validation — and a finding that matters

The same recipe was applied to the **real-world ULB dataset** (284,807 genuine
European card transactions, 0.17% fraud, PCA-anonymised):

| Strategy | PR-AUC (ULB, real) | PR-AUC (Sparkov) |
|:---|:---|:---|
| Plain LightGBM | **0.418** | 0.682 |
| Cost-sensitive | 0.025 | **0.980** |

> **Imbalance strategy is dataset-dependent.** Cost-sensitive weighting *dominates*
> on Sparkov (strong engineered features) but **collapses** on ULB (weak PCA
> features) — there, aggressive `scale_pos_weight` floods the score head with false
> positives and a plain model wins. There is no universal imbalance recipe; you
> validate per-dataset. This is why the project ships an imbalance *study*, not a
> single assumed fix.

---

## Architecture

```
Raw transaction stream
        │
        ▼
┌──────────────────┐     offline (batch)          online (real-time)
│ Feature pipeline │ ──► src/features.py     ──►   src/online.py (in-memory state)
└──────────────────┘
        │
        ▼
┌──────────────────────────────────────────────┐
│ Models                                         │
│  • LightGBM  (cost-sensitive + Optuna)         │  ◄── production
│  • GraphSAGE (card-chain transaction graph)    │  ◄── relational
│  • Autoencoder (legit-only reconstruction)     │  ◄── unsupervised
└──────────────────────────────────────────────┘
        │
        ▼
┌──────────────────┐   ┌──────────────┐   ┌───────────────┐
│ Evaluation       │   │ SHAP explain │   │ PSI drift     │
│ PR-AUC + cost    │   │ (compliance) │   │ monitoring    │
└──────────────────┘   └──────────────┘   └───────────────┘
        │
        ▼
   FastAPI /score   +   Gradio dashboard   (HF Spaces)
```

---

## Project Structure

```
fraud-detection/
├── src/
│   ├── config.py        # paths, feature groups, costs, hyperparams
│   ├── data.py          # Sparkov download + load
│   ├── features.py      # batch feature engineering (no leakage)
│   ├── online.py        # incremental online feature store (real-time)
│   ├── preprocess.py    # matrices + temporal split
│   ├── train.py         # LightGBM + imbalance study + Optuna
│   ├── autoencoder.py   # unsupervised anomaly detector (PyTorch)
│   ├── gnn.py           # GraphSAGE on card-chain graph (PyG)
│   ├── evaluate.py      # PR-AUC, business cost, threshold optimization
│   ├── explain.py       # SHAP
│   └── drift.py         # PSI concept-drift
├── scripts/             # download / features / train / autoencoder / gnn / drift
├── api/                 # FastAPI real-time scoring service
├── streaming/           # streaming replay + latency benchmark
├── app/gradio_app.py    # 6-tab interactive dashboard
├── tests/               # pytest (features, evaluate, drift, online)
└── .github/workflows/   # CI
```

---

## Running Locally

```bash
pip install -r requirements.txt

python scripts/download_data.py     # Sparkov (~200 MB)
python scripts/run_features.py      # engineered feature tables (~3 min)
python scripts/run_training.py      # LightGBM + imbalance study + Optuna
python scripts/run_autoencoder.py   # unsupervised baseline (GPU)
python scripts/run_gnn.py           # GraphSAGE (GPU)
python scripts/run_drift.py         # PSI drift report
python streaming/simulate_stream.py # latency benchmark

python app/gradio_app.py            # dashboard at :7860
uvicorn api.main:app --port 8000    # real-time API
pytest tests/ -v                    # tests
```

---

## What This Demonstrates

- **Imbalanced learning done right** — PR-AUC, cost-sensitive learning, business-cost
  threshold optimization (not accuracy, not default 0.5)
- **Feature engineering** — leakage-safe velocity/behavioral/geo features that carry the signal
- **Breadth of modelling** — gradient boosting, graph neural networks, and autoencoders, compared honestly
- **Production thinking** — real-time online features, latency benchmarking, drift monitoring, explainability
- **Engineering** — typed config, unit tests, CI, MLflow tracking, Docker, deployed demo

**Relevant roles:** fraud/risk DS at GoPay, OVO, Kredivo, Akulaku, Dana; any payments or lending team.

---

## References

- Deng et al. — *cost-sensitive vs SMOTE at scale*; 821-paper review (2025) on imbalanced learning in production
- Hamilton, Ying, Leskovec (2017). *Inductive Representation Learning on Large Graphs* (**GraphSAGE**)
- SR 11-7 / FinCEN model-risk guidance — explainability requirements for automated decisions
- Sparkov Data Generator — Brandon Harris (dataset)

---

*Built by [Muhammad Fikri Wahidin](https://github.com/Fikri645) — ML Engineer / Data Scientist portfolio*
