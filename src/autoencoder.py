"""
Autoencoder anomaly detector — the unsupervised counterpoint to gradient boosting.

Idea: train a small autoencoder to reconstruct ONLY legitimate transactions.
Fraud, being out-of-distribution, reconstructs poorly → high reconstruction
error = high anomaly score. This needs no fraud labels at training time, which
matters in the real world where labels arrive late (chargebacks take weeks).

We compare its PR-AUC against the supervised model to make the trade-off
explicit: supervised wins on accuracy when labels exist; the autoencoder is the
fallback for novel, never-before-seen fraud patterns.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src import config


class Autoencoder(nn.Module):
    def __init__(self, n_features: int, hidden=None):
        super().__init__()
        hidden = hidden or config.AE_HIDDEN
        dims = [n_features] + hidden
        enc = []
        for i in range(len(dims) - 1):
            enc += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
        self.encoder = nn.Sequential(*enc)
        dec = []
        rdims = list(reversed(dims))
        for i in range(len(rdims) - 1):
            dec += [nn.Linear(rdims[i], rdims[i + 1])]
            if i < len(rdims) - 2:
                dec += [nn.ReLU()]
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_autoencoder(X_legit: np.ndarray, epochs=config.AE_EPOCHS,
                      batch=config.AE_BATCH, lr=config.AE_LR, device=None, verbose=True):
    """Train on legit-only rows. Returns the trained model."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.tensor(np.asarray(X_legit), dtype=torch.float32)
    ds = torch.utils.data.TensorDataset(X)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True)

    model = Autoencoder(X.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for ep in range(epochs):
        total = 0.0
        for (xb,) in dl:
            xb = xb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), xb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            print(f"[ae] epoch {ep:2d}  loss={total / len(X):.5f}")
    return model


@torch.no_grad()
def reconstruction_error(model, X: np.ndarray, device=None, batch=8192) -> np.ndarray:
    """Per-row MSE reconstruction error = anomaly score."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    X = torch.tensor(np.asarray(X), dtype=torch.float32)
    errs = []
    for i in range(0, len(X), batch):
        xb = X[i:i + batch].to(device)
        recon = model(xb)
        err = ((recon - xb) ** 2).mean(dim=1).cpu().numpy()
        errs.append(err)
    return np.concatenate(errs)
