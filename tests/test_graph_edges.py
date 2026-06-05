"""
Tests for the GNN graph edge builder — the critical guarantee is NO temporal
leakage: every edge must point strictly past -> present.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gnn import build_directed_chain_edges


def test_edges_are_strictly_past_to_present():
    """For group-sorted data, every edge src index < dst index (past -> present)."""
    groups = np.array([1, 1, 1, 1, 2, 2, 2])
    e = build_directed_chain_edges(groups, k=3)
    assert e.shape[0] == 2
    # Within group-sorted space, source (past) must precede destination (present)
    assert np.all(e[0] < e[1]), "found an edge pointing to the past (leakage!)"


def test_no_cross_group_edges():
    groups = np.array([1, 1, 2, 2, 2])
    e = build_directed_chain_edges(groups, k=5)
    for s, d in zip(e[0], e[1]):
        assert groups[s] == groups[d], "edge crosses group boundary"


def test_k_limits_fanin():
    """With k=2, the 4th element of a group links to at most its 2 predecessors."""
    groups = np.array([1, 1, 1, 1])
    e = build_directed_chain_edges(groups, k=2)
    # destination index 3 should have exactly 2 incoming edges (from 1 and 2)
    incoming_to_3 = (e[1] == 3).sum()
    assert incoming_to_3 == 2


def test_no_reverse_edges():
    """The builder must not emit bidirectional pairs (would leak the future)."""
    groups = np.array([1, 1, 1])
    e = build_directed_chain_edges(groups, k=5)
    pairs = set(zip(e[0].tolist(), e[1].tolist()))
    for s, d in pairs:
        assert (d, s) not in pairs, "reverse edge present -> temporal leakage"


def test_empty_when_singletons():
    groups = np.array([1, 2, 3])  # every group size 1 -> no chains
    e = build_directed_chain_edges(groups, k=5)
    assert e.shape[1] == 0


def test_first_in_group_has_no_incoming():
    groups = np.array([5, 5, 5])
    e = build_directed_chain_edges(groups, k=5)
    # index 0 is the first txn -> no predecessor -> no incoming edge
    assert (e[1] == 0).sum() == 0
