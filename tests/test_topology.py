import numpy as np
import pandas as pd
import pytest

from network_momentum.graph import GraphSnapshot
from network_momentum.topology import (
    EdgeMaskSpec,
    binary_adjacency,
    mask_snapshot_edges,
    snapshot_topology,
    topology_time_series,
)


def _triangle_plus_pair() -> GraphSnapshot:
    # triângulo A-B-C + par D-E
    adjacency = np.zeros((5, 5))
    for i, j in [(0, 1), (0, 2), (1, 2), (3, 4)]:
        adjacency[i, j] = adjacency[j, i] = 1.0
    return GraphSnapshot(
        date=pd.Timestamp("2022-01-03"),
        assets=("A", "B", "C", "D", "E"),
        adjacency=adjacency,
        raw_adjacency=adjacency,
    )


def test_topology_metrics_on_toy_graph() -> None:
    snapshot = _triangle_plus_pair()
    labels = {"A": "G1", "B": "G1", "C": "G1", "D": "G2", "E": "G2"}
    stats = snapshot_topology(snapshot, edge_threshold=1e-6, group_labels=labels)
    assert stats["n_nodes"] == 5
    assert stats["n_edges"] == 4
    assert stats["edge_sparsity"] == pytest.approx(4 / 10)
    assert stats["average_degree"] == pytest.approx((2 + 2 + 2 + 1 + 1) / 5)
    # clustering: nós do triângulo têm coeficiente 1; D e E têm grau 1 -> 0
    assert stats["clustering_coefficient"] == pytest.approx(3 / 5)
    # todas as arestas são intra-grupo e todos os não-arcos entre grupos estão ausentes
    assert stats["community_ratio"] == pytest.approx(1.0)


def test_jaccard_between_identical_snapshots_is_one() -> None:
    snapshot = _triangle_plus_pair()
    series = topology_time_series([snapshot, snapshot], edge_threshold=1e-6)
    assert np.isnan(series["jaccard_previous"].iloc[0])
    assert series["jaccard_previous"].iloc[1] == pytest.approx(1.0)


def test_edge_masks_zero_correct_entries() -> None:
    snapshot = _triangle_plus_pair()
    labels = {"A": "G1", "B": "G1", "C": "G2", "D": "G2", "E": "G2"}
    intra = mask_snapshot_edges(snapshot, EdgeMaskSpec(mode="intra", labels=labels))
    # sobra apenas A-B (G1-G1) e D-E (G2-G2); somem A-C e B-C (inter)
    assert intra.raw_adjacency[0, 1] == 1.0
    assert intra.raw_adjacency[0, 2] == 0.0
    assert intra.raw_adjacency[3, 4] == 1.0
    inter = mask_snapshot_edges(snapshot, EdgeMaskSpec(mode="inter", labels=labels))
    assert inter.raw_adjacency[0, 1] == 0.0
    assert inter.raw_adjacency[0, 2] == 1.0
    assert inter.raw_adjacency[3, 4] == 0.0


def test_binary_adjacency_threshold_is_relative() -> None:
    adjacency = np.array([[0.0, 1.0, 1e-6], [1.0, 0.0, 0.0], [1e-6, 0.0, 0.0]])
    binary = binary_adjacency(adjacency, edge_threshold=1e-3)
    assert binary[0, 1]
    assert not binary[0, 2]
