"""
Graph Neural Network for fraud — the cutting-edge module.

Why a graph? Fraud is relational. A stolen card produces a *burst* of linked
transactions; tabular models see each transaction in isolation, but a GNN can
propagate signal along the card's transaction chain — if the previous few
transactions on this card look fraudulent, that context raises suspicion here.

Graph construction
  Nodes  = transactions (features = the engineered tabular vector)
  Edges  = each transaction linked to the previous K transactions of the SAME
           card (a temporal "card chain"). This injects exactly the relational
           burst signal that catches account-takeover fraud.

Model    = GraphSAGE (inductive; uses neighbour sampling so it scales to the
           1.85M-node graph and runs on a single RTX 3060).

The graph spans train+test nodes (a test transaction can attend to the card's
earlier history — realistic at inference) but the loss is computed only on train
nodes, so test labels never leak.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src import config


# ── Graph construction ──────────────────────────────────────────────────────

def build_card_chain_edges(card_ids: np.ndarray, k: int = 5) -> np.ndarray:
    """
    For data already sorted by (card, time), link each transaction to its
    previous up-to-k transactions on the same card. Returns edge_index [2, E]
    (directed past -> present, plus the reverse for message passing).
    """
    src, dst = [], []
    n = len(card_ids)
    # Find contiguous runs of identical card id (input must be card-grouped)
    start = 0
    for i in range(1, n + 1):
        if i == n or card_ids[i] != card_ids[start]:
            # run [start, i)
            for j in range(start, i):
                lo = max(start, j - k)
                for p in range(lo, j):
                    src.append(p); dst.append(j)   # past -> present
            start = i
    if not src:
        return np.zeros((2, 0), dtype=np.int64)
    e = np.array([src, dst], dtype=np.int64)
    # add reverse edges for bidirectional message passing
    e = np.concatenate([e, e[::-1]], axis=1)
    return e


def build_graph(X: np.ndarray, y: np.ndarray, card_ids: np.ndarray,
                train_mask: np.ndarray, k: int = 5):
    """Assemble a PyG Data object. X must be row-aligned with card_ids/y."""
    from torch_geometric.data import Data

    # Sort everything by card so chain edges are contiguous-run cheap to build
    order = np.argsort(card_ids, kind="stable")
    inv = np.empty_like(order); inv[order] = np.arange(len(order))

    edges_sorted = build_card_chain_edges(card_ids[order], k=k)
    # Map edge endpoints back to original row indices
    edge_index = inv[edges_sorted] if edges_sorted.size else edges_sorted

    data = Data(
        x=torch.tensor(X, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        y=torch.tensor(y, dtype=torch.float32),
    )
    data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
    return data


# ── Model ───────────────────────────────────────────────────────────────────

class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden=config.GNN_HIDDEN, layers=config.GNN_LAYERS):
        super().__init__()
        from torch_geometric.nn import SAGEConv
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden))
        for _ in range(layers - 1):
            self.convs.append(SAGEConv(hidden, hidden))
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=0.2, training=self.training)
        return self.head(x).squeeze(-1)


# ── Training (full-batch on GPU) ────────────────────────────────────────────
# This graph (~1.85M nodes, ~18M edges, ~50 features) fits comfortably on a
# 12 GB GPU, so we train full-batch with a masked loss. This avoids the
# pyg-lib / torch-sparse native dependency that neighbour sampling requires
# (and which is painful to build on Windows), while remaining inductive-style:
# the loss is computed only on train-period nodes; test labels never leak.

def train_gnn(data, epochs=config.GNN_EPOCHS, device=None, verbose=True):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphSAGE(data.num_node_features).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.GNN_LR, weight_decay=1e-5)

    data = data.to(device)
    train_mask = data.train_mask
    n_pos = float(data.y[train_mask].sum())
    n_neg = float(train_mask.sum()) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model.train()
    for ep in range(epochs):
        opt.zero_grad()
        out = model(data.x, data.edge_index)
        loss = loss_fn(out[train_mask], data.y[train_mask])
        loss.backward()
        opt.step()
        if verbose:
            print(f"[gnn] epoch {ep:2d}  loss={loss.item():.4f}")
    return model


@torch.no_grad()
def predict_gnn(model, data, device=None):
    """Full-graph inference."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    data = data.to(device)
    out = torch.sigmoid(model(data.x, data.edge_index))
    return out.cpu().numpy()
