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

def build_directed_chain_edges(group_ids: np.ndarray, k: int = 5) -> np.ndarray:
    """
    For data already sorted by (group, time), link each transaction to its
    previous up-to-k transactions in the same group. Returns edge_index [2, E]
    with edges **strictly directed past -> present**.

    Directed-only is deliberate: GraphSAGE then aggregates a node only from its
    own past, so even multi-hop receptive fields stay in the past. This removes
    the temporal leakage a bidirectional graph would introduce (a transaction
    must never see future transactions of the same card/merchant at score time).
    """
    src, dst = [], []
    n = len(group_ids)
    start = 0
    for i in range(1, n + 1):
        if i == n or group_ids[i] != group_ids[start]:
            for j in range(start, i):
                lo = max(start, j - k)
                for p in range(lo, j):
                    src.append(p); dst.append(j)   # past -> present only
            start = i
    if not src:
        return np.zeros((2, 0), dtype=np.int64)
    return np.array([src, dst], dtype=np.int64)


def _chain_edges_by_group(group_ids: np.ndarray, k: int) -> np.ndarray:
    """Build directed chain edges for an arbitrary grouping, in original index space."""
    order = np.argsort(group_ids, kind="stable")   # stable -> preserves time order within group
    inv = np.empty_like(order); inv[order] = np.arange(len(order))
    e_sorted = build_directed_chain_edges(group_ids[order], k=k)
    return inv[e_sorted] if e_sorted.size else e_sorted


def build_graph(X: np.ndarray, y: np.ndarray, card_ids: np.ndarray,
                train_mask: np.ndarray, merchant_ids: np.ndarray = None,
                k: int = 5):
    """
    Assemble a PyG Data object. X must be row-aligned with card_ids/y.

    Edges (all directed past -> present):
      - card chain:     each txn -> its previous k txns on the same card
                        (captures account-takeover bursts)
      - merchant chain: each txn -> its previous k txns at the same merchant
                        (captures fraud rings hitting one merchant), if provided
    """
    from torch_geometric.data import Data

    edge_sets = [_chain_edges_by_group(card_ids, k)]
    if merchant_ids is not None:
        edge_sets.append(_chain_edges_by_group(merchant_ids, k))
    edge_sets = [e for e in edge_sets if e.size]
    edge_index = np.concatenate(edge_sets, axis=1) if edge_sets else np.zeros((2, 0), np.int64)

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
