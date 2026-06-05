"""
Train the GraphSAGE fraud model on the card-chain transaction graph.

    python scripts/run_gnn.py
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src import config, preprocess, gnn, evaluate  # noqa: E402


def main():
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    df_tr = preprocess.load_features("train")
    df_te = preprocess.load_features("test")

    # Combined node set: train nodes (labelled in loss) + test nodes (masked)
    Xn_tr, Xn_te, _ = preprocess.encode_numeric_matrix(df_tr, df_te)
    X = np.vstack([Xn_tr.values, Xn_te.values]).astype(np.float32)
    y = np.concatenate([df_tr[config.TARGET].values, df_te[config.TARGET].values]).astype(np.float32)
    cards = np.concatenate([df_tr[config.CARD_COL].values, df_te[config.CARD_COL].values])
    train_mask = np.concatenate([np.ones(len(df_tr), bool), np.zeros(len(df_te), bool)])

    print(f"[gnn] building graph: {len(X):,} nodes "
          f"({'GPU' if torch.cuda.is_available() else 'CPU'})")
    data = gnn.build_graph(X, y, cards, train_mask, k=5)
    print(f"[gnn] edges: {data.edge_index.shape[1]:,}")

    model = gnn.train_gnn(data)

    scores_all = gnn.predict_gnn(model, data)
    test_scores = scores_all[~train_mask]
    y_te = y[~train_mask].astype(int)
    ev = evaluate.evaluate(y_te, test_scores, df_te["amt"].values)
    print(f"\n[gnn] TEST  PR-AUC={ev.pr_auc:.4f}  ROC-AUC={ev.roc_auc:.4f}  "
          f"recall@1%={ev.recall_at_1pct:.3f}")

    torch.save(model.state_dict(), config.GNN_MODEL)
    out = {
        "model": "GraphSAGE (card-chain transaction graph)",
        "trained_at": time.strftime("%Y-%m-%d %H:%M"),
        "test_metrics": ev.to_dict(),
        "n_nodes": int(len(X)), "n_edges": int(data.edge_index.shape[1]),
    }
    with open(config.MODELS_DIR / "gnn_meta.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[gnn] saved {config.GNN_MODEL.name}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
