import numpy as np
import pandas as pd

from network_momentum.config import GraphConfig
from network_momentum.graph import (
    build_graph_snapshots,
    learn_adjacency,
    normalize_adjacency,
    propagate_network_features,
)


def test_learned_graph_constraints() -> None:
    rng = np.random.default_rng(11)
    base = rng.normal(size=(1, 80))
    observations = np.vstack(
        [
            base,
            base + rng.normal(scale=0.05, size=(1, 80)),
            rng.normal(size=(1, 80)),
            rng.normal(size=(1, 80)),
        ]
    )
    adjacency = learn_adjacency(
        observations,
        alpha=0.1,
        beta=0.1,
        maxiter=500,
    )
    assert np.allclose(adjacency, adjacency.T)
    assert np.allclose(np.diag(adjacency), 0.0)
    assert np.all(adjacency >= 0)
    assert np.all(adjacency.sum(axis=1) > 0)
    assert adjacency[0, 1] > adjacency[0, 2]

    normalized = normalize_adjacency(adjacency)
    assert np.isfinite(normalized).all()
    assert np.allclose(normalized, normalized.T)


def test_snapshot_ensemble_and_propagation() -> None:
    rng = np.random.default_rng(12)
    dates = pd.bdate_range("2020-01-01", periods=90)
    assets = ("A", "B", "C", "D")
    feature_names = ("f1", "f2")
    columns = pd.MultiIndex.from_product(
        [assets, feature_names],
        names=["ticker", "feature"],
    )
    features = pd.DataFrame(
        rng.normal(size=(len(dates), len(columns))),
        index=dates,
        columns=columns,
    )
    config = GraphConfig(
        lookbacks=(20, 40),
        alpha_grid=(0.1,),
        beta_grid=(0.1,),
        rebalance_every=10,
        min_assets=4,
        maxiter=300,
    )
    snapshots = build_graph_snapshots(
        features,
        feature_names,
        config,
        alpha=0.1,
        beta=0.1,
    )
    propagated = propagate_network_features(features, feature_names, snapshots)
    assert snapshots
    assert propagated.index.names == ["date", "ticker"]
    assert tuple(propagated.columns) == feature_names
    assert np.isfinite(propagated.to_numpy()).all()

