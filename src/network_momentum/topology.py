from __future__ import annotations

"""Métricas de topologia dos grafos aprendidos (Seção 5.1 do artigo).

Todas as métricas binarizam a adjacência com um threshold relativo
(``edge_threshold`` × maior peso do snapshot), necessário porque o solver
L-BFGS-B suavizado não produz zeros exatos como o solver convexo do artigo.
O threshold usado é sempre reportado junto com as métricas.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .graph import GraphSnapshot


@dataclass(frozen=True)
class EdgeMaskSpec:
    """Máscara para ablação de arestas intra/inter grupo (região ou setor)."""

    mode: str  # "intra" | "inter"
    labels: dict[str, str]  # ticker -> grupo


def binary_adjacency(adjacency: np.ndarray, edge_threshold: float) -> np.ndarray:
    if adjacency.size == 0:
        return adjacency.astype(bool)
    cutoff = edge_threshold * float(adjacency.max()) if adjacency.max() > 0 else 0.0
    binary = adjacency > max(cutoff, 0.0)
    np.fill_diagonal(binary, False)
    return binary


def snapshot_topology(
    snapshot: GraphSnapshot,
    *,
    edge_threshold: float,
    group_labels: dict[str, str] | None = None,
) -> dict[str, float]:
    adjacency = (
        snapshot.raw_adjacency if snapshot.raw_adjacency is not None else snapshot.adjacency
    )
    n = adjacency.shape[0]
    binary = binary_adjacency(adjacency, edge_threshold)
    n_edges = int(binary.sum()) // 2
    possible = n * (n - 1) / 2.0
    density = n_edges / possible if possible else np.nan

    degrees = binary.sum(axis=1).astype(float)
    average_degree = float(degrees.mean()) if n else np.nan

    binary_float = binary.astype(float)
    triangles = np.diag(binary_float @ binary_float @ binary_float) / 2.0
    with np.errstate(invalid="ignore", divide="ignore"):
        local_clustering = np.where(
            degrees > 1, 2.0 * triangles / (degrees * (degrees - 1.0)), 0.0
        )
    clustering = float(local_clustering.mean()) if n else np.nan

    community_ratio = np.nan
    if group_labels is not None:
        labels = np.array([group_labels.get(a, "") for a in snapshot.assets], dtype=object)
        same_group = labels[:, None] == labels[None, :]
        upper = np.triu(np.ones((n, n), dtype=bool), k=1)
        agreements = (
            int((binary & same_group & upper).sum())
            + int((~binary & ~same_group & upper).sum())
        )
        community_ratio = agreements / possible if possible else np.nan

    return {
        "date": snapshot.date,
        "n_nodes": float(n),
        "n_edges": float(n_edges),
        "edge_sparsity": float(density),
        "average_degree": average_degree,
        "clustering_coefficient": clustering,
        "community_ratio": float(community_ratio),
        "edge_threshold": float(edge_threshold),
    }


def topology_time_series(
    snapshots: list[GraphSnapshot],
    *,
    edge_threshold: float,
    group_labels: dict[str, str] | None = None,
) -> pd.DataFrame:
    rows = [
        snapshot_topology(s, edge_threshold=edge_threshold, group_labels=group_labels)
        for s in snapshots
    ]
    frame = pd.DataFrame(rows).set_index("date")

    jaccard: list[float] = [np.nan]
    for previous, current in zip(snapshots[:-1], snapshots[1:]):
        edges_previous = _edge_set(previous, edge_threshold)
        edges_current = _edge_set(current, edge_threshold)
        union = edges_previous | edges_current
        intersection = edges_previous & edges_current
        jaccard.append(len(intersection) / len(union) if union else np.nan)
    frame["jaccard_previous"] = jaccard
    return frame


def _edge_set(snapshot: GraphSnapshot, edge_threshold: float) -> set[tuple[str, str]]:
    adjacency = (
        snapshot.raw_adjacency if snapshot.raw_adjacency is not None else snapshot.adjacency
    )
    binary = binary_adjacency(adjacency, edge_threshold)
    edge_i, edge_j = np.nonzero(np.triu(binary, k=1))
    return {
        (snapshot.assets[i], snapshot.assets[j]) for i, j in zip(edge_i, edge_j)
    }


def mask_snapshot_edges(snapshot: GraphSnapshot, spec: EdgeMaskSpec) -> GraphSnapshot:
    """Zera arestas intra ou inter grupo no ensemble bruto (para as ablações
    GMOM-Intra / GMOM-Inter da Seção 5.2 do artigo, aplicadas a região ou setor)."""
    adjacency = (
        snapshot.raw_adjacency if snapshot.raw_adjacency is not None else snapshot.adjacency
    ).copy()
    labels = np.array([spec.labels.get(a, "") for a in snapshot.assets], dtype=object)
    same_group = labels[:, None] == labels[None, :]
    if spec.mode == "intra":
        adjacency[~same_group] = 0.0
    elif spec.mode == "inter":
        adjacency[same_group] = 0.0
    else:
        raise ValueError("spec.mode deve ser 'intra' ou 'inter'.")
    np.fill_diagonal(adjacency, 0.0)
    from .graph import normalize_adjacency

    return GraphSnapshot(
        date=snapshot.date,
        assets=snapshot.assets,
        adjacency=normalize_adjacency(adjacency),
        raw_adjacency=adjacency,
    )
