from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .config import GraphConfig


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class GraphSnapshot:
    date: pd.Timestamp
    assets: tuple[str, ...]
    adjacency: np.ndarray  # ensemble normalizado (Eq. 6), para inspeção/compatibilidade
    raw_adjacency: np.ndarray | None = None  # ensemble bruto (Eq. 5), usado na propagação


def normalize_adjacency(adjacency: np.ndarray) -> np.ndarray:
    degrees = adjacency.sum(axis=1)
    inverse_root = np.zeros_like(degrees, dtype=float)
    positive = degrees > 0
    inverse_root[positive] = 1.0 / np.sqrt(degrees[positive])
    normalized = inverse_root[:, None] * adjacency * inverse_root[None, :]
    np.fill_diagonal(normalized, 0.0)
    return normalized


def _pairwise_squared_distances(values: np.ndarray) -> np.ndarray:
    squared_norm = np.einsum("ij,ij->i", values, values)
    distances = squared_norm[:, None] + squared_norm[None, :] - 2.0 * values.dot(values.T)
    return np.maximum(distances, 0.0)


def _learn_adjacency_cvxpy(
    edge_distances: np.ndarray,
    edge_i: np.ndarray,
    edge_j: np.ndarray,
    n_assets: int,
    *,
    alpha: float,
    beta: float,
    tolerance: float,
) -> np.ndarray:
    """Resolve a Eq. (4) com um solver convexo exato (CLARABEL/SCS via CVXPY).

    Equivalente à formulação do artigo (que usa MOSEK). Produz zeros exatos nas
    arestas, ao contrário do caminho L-BFGS-B suavizado.
    """
    import cvxpy as cp

    n_edges = edge_i.size
    incidence = np.zeros((n_assets, n_edges))
    incidence[edge_i, np.arange(n_edges)] = 1.0
    incidence[edge_j, np.arange(n_edges)] = 1.0

    weights = cp.Variable(n_edges, nonneg=True)
    degrees = incidence @ weights
    objective = (
        edge_distances @ weights
        - alpha * cp.sum(cp.log(degrees))
        + 2.0 * beta * cp.sum_squares(weights)
    )
    problem = cp.Problem(cp.Minimize(objective))
    solved = False
    for solver_name in ("CLARABEL", "SCS", "ECOS"):
        if solver_name not in cp.installed_solvers():
            continue
        try:
            problem.solve(solver=solver_name)
        except cp.SolverError:
            continue
        if weights.value is not None and np.isfinite(problem.value):
            solved = True
            break
    if not solved:
        raise RuntimeError("Nenhum solver CVXPY disponível resolveu o grafo.")
    edge_values = np.maximum(np.asarray(weights.value, dtype=float), 0.0)
    edge_values[edge_values < max(tolerance, 1e-12)] = 0.0
    return edge_values


def learn_adjacency(
    observations: np.ndarray,
    *,
    alpha: float,
    beta: float,
    maxiter: int = 300,
    tolerance: float = 1e-7,
    initial_adjacency: np.ndarray | None = None,
    solver: str = "lbfgs",
) -> np.ndarray:
    """Resolve a Eq. (4) do artigo nas arestas do triângulo superior.

    Para A simétrica e sem autoarestas:
      tr(V' L V) = sum_(i<j) A_ij ||V_i - V_j||²
      ||A||²_F = 2 sum_(i<j) A_ij²

    ``solver="lbfgs"`` usa a reparametrização w = (alpha/mediana) x, que preserva o
    ótimo (constante aditiva descartada). ``solver="cvxpy"`` usa o problema convexo
    exato, como no artigo (MOSEK substituído por CLARABEL/SCS/ECOS abertos).
    """
    values = np.asarray(observations, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("observations deve ter formato (ativos, observações), com >= 2 ativos.")
    if not np.isfinite(values).all():
        raise ValueError("observations contém valores não finitos.")
    if alpha <= 0 or beta <= 0:
        raise ValueError("alpha e beta devem ser positivos.")

    n_assets = values.shape[0]
    edge_i, edge_j = np.triu_indices(n_assets, k=1)
    distances = _pairwise_squared_distances(values)
    edge_distances = distances[edge_i, edge_j]

    if solver == "cvxpy":
        learned_edges = _learn_adjacency_cvxpy(
            edge_distances,
            edge_i,
            edge_j,
            n_assets,
            alpha=alpha,
            beta=beta,
            tolerance=tolerance,
        )
        adjacency = np.zeros((n_assets, n_assets), dtype=float)
        adjacency[edge_i, edge_j] = learned_edges
        adjacency[edge_j, edge_i] = learned_edges
        if np.any(adjacency.sum(axis=1) <= 0):
            raise RuntimeError("O grafo aprendido contém ativo isolado.")
        return adjacency

    positive_distances = edge_distances[edge_distances > np.finfo(float).eps]
    distance_scale = (
        float(np.median(positive_distances)) if positive_distances.size else 1.0
    )
    distance_scale = max(distance_scale, np.finfo(float).eps)

    # w = alpha / median(z) * v melhora o condicionamento sem alterar o ótimo.
    weight_scale = alpha / distance_scale
    scaled_distances = edge_distances / distance_scale
    quadratic_coefficient = 2.0 * beta * alpha / (distance_scale**2)

    if (
        initial_adjacency is not None
        and initial_adjacency.shape == (n_assets, n_assets)
        and np.isfinite(initial_adjacency).all()
    ):
        initial = np.maximum(initial_adjacency[edge_i, edge_j] / weight_scale, 0.0)
        degrees = np.bincount(
            np.concatenate([edge_i, edge_j]),
            weights=np.concatenate([initial, initial]),
            minlength=n_assets,
        )
        if np.any(degrees <= 0):
            initial = np.full(edge_i.size, 1.0 / max(n_assets - 1, 1))
    else:
        initial = np.full(edge_i.size, 1.0 / max(n_assets - 1, 1))

    def objective_and_gradient(edge_weights: np.ndarray) -> tuple[float, np.ndarray]:
        degrees = np.bincount(
            np.concatenate([edge_i, edge_j]),
            weights=np.concatenate([edge_weights, edge_weights]),
            minlength=n_assets,
        )
        if np.any(degrees <= 0) or not np.isfinite(degrees).all():
            return np.inf, np.zeros_like(edge_weights)
        objective = (
            float(scaled_distances.dot(edge_weights))
            - float(np.log(degrees).sum())
            + quadratic_coefficient * float(edge_weights.dot(edge_weights))
        )
        inverse_degrees = 1.0 / degrees
        gradient = (
            scaled_distances
            - inverse_degrees[edge_i]
            - inverse_degrees[edge_j]
            + 2.0 * quadratic_coefficient * edge_weights
        )
        return objective, gradient

    result = minimize(
        objective_and_gradient,
        initial,
        method="L-BFGS-B",
        jac=True,
        bounds=[(0.0, None)] * edge_i.size,
        options={
            "maxiter": maxiter,
            "ftol": tolerance,
            "gtol": tolerance,
            "maxls": 50,
        },
    )
    if not np.isfinite(result.fun):
        raise RuntimeError(f"O solver do grafo falhou: {result.message}")
    if not result.success:
        LOGGER.warning("Solver do grafo encerrou sem convergência plena: %s", result.message)

    learned_edges = np.maximum(result.x * weight_scale, 0.0)
    learned_edges[learned_edges < np.finfo(float).eps] = 0.0
    adjacency = np.zeros((n_assets, n_assets), dtype=float)
    adjacency[edge_i, edge_j] = learned_edges
    adjacency[edge_j, edge_i] = learned_edges
    if np.any(adjacency.sum(axis=1) <= 0):
        raise RuntimeError("O grafo aprendido contém ativo isolado.")
    return adjacency


def _active_assets(
    features: pd.DataFrame,
    end_position: int,
    lookback: int,
    tickers: Iterable[str],
) -> tuple[str, ...]:
    window = features.iloc[end_position - lookback + 1 : end_position + 1]
    active: list[str] = []
    for ticker in tickers:
        ticker_values = window.xs(ticker, axis=1, level="ticker")
        if ticker_values.notna().all(axis=None):
            active.append(ticker)
    return tuple(active)


def _observation_matrix(
    features: pd.DataFrame,
    end_position: int,
    lookback: int,
    assets: tuple[str, ...],
    feature_names: tuple[str, ...],
) -> np.ndarray:
    columns = pd.MultiIndex.from_product(
        [assets, feature_names],
        names=["ticker", "feature"],
    )
    window = features.iloc[end_position - lookback + 1 : end_position + 1].reindex(
        columns=columns
    )
    cube = window.to_numpy(dtype=float).reshape(lookback, len(assets), len(feature_names))
    return cube.transpose(1, 0, 2).reshape(len(assets), -1)


def build_graph_snapshots(
    features: pd.DataFrame,
    feature_names: tuple[str, ...],
    config: GraphConfig,
    *,
    alpha: float,
    beta: float,
) -> list[GraphSnapshot]:
    """Aprende os grafos (Eq. 4), agrega por ensemble (Eq. 5) e normaliza (Eq. 6).

    Com ``per_lookback_eligibility=True`` cada lookback usa os ativos com histórico
    completo naquele horizonte (como no artigo); o ensemble faz a média das arestas
    pelos grafos em que o par de ativos era elegível. Com ``False`` reproduz o
    comportamento antigo (elegibilidade pelo maior lookback para todos os grafos).
    """
    min_lookback = min(config.lookbacks)
    max_lookback = max(config.lookbacks)
    start_lookback = min_lookback if config.per_lookback_eligibility else max_lookback
    if len(features) < start_lookback:
        raise ValueError(
            f"São necessárias ao menos {start_lookback} datas para aprender os grafos."
        )
    tickers = tuple(dict.fromkeys(features.columns.get_level_values("ticker")))
    warm_starts: dict[int, tuple[tuple[str, ...], np.ndarray]] = {}
    snapshots: list[GraphSnapshot] = []
    positions = range(start_lookback - 1, len(features), config.rebalance_every)

    for number, position in enumerate(positions, start=1):
        assets_by_lookback: dict[int, tuple[str, ...]] = {}
        for lookback in config.lookbacks:
            if position - lookback + 1 < 0:
                continue
            eligibility_lookback = (
                lookback if config.per_lookback_eligibility else max_lookback
            )
            assets_by_lookback[lookback] = _active_assets(
                features, position, eligibility_lookback, tickers
            )
        assets_by_lookback = {
            lb: assets for lb, assets in assets_by_lookback.items() if len(assets) >= 2
        }
        if not assets_by_lookback:
            continue
        union_assets = tuple(
            sorted(set().union(*[set(a) for a in assets_by_lookback.values()]))
        )
        if len(union_assets) < config.min_assets:
            continue
        index_of = {asset: k for k, asset in enumerate(union_assets)}

        summed = np.zeros((len(union_assets), len(union_assets)))
        counts = np.zeros_like(summed)
        for lookback, assets in assets_by_lookback.items():
            observations = _observation_matrix(
                features,
                position,
                lookback,
                assets,
                feature_names,
            )
            previous = warm_starts.get(lookback)
            initial = previous[1] if previous and previous[0] == assets else None
            adjacency = learn_adjacency(
                observations,
                alpha=alpha,
                beta=beta,
                maxiter=config.maxiter,
                tolerance=config.tolerance,
                initial_adjacency=initial,
                solver=config.solver,
            )
            warm_starts[lookback] = (assets, adjacency)
            locations = np.array([index_of[a] for a in assets])
            summed[np.ix_(locations, locations)] += adjacency
            counts[np.ix_(locations, locations)] += 1.0

        with np.errstate(invalid="ignore", divide="ignore"):
            ensemble = np.where(counts > 0, summed / np.maximum(counts, 1.0), 0.0)
        np.fill_diagonal(ensemble, 0.0)
        connected = ensemble.sum(axis=1) > 0
        if connected.sum() < config.min_assets:
            continue
        kept_assets = tuple(np.asarray(union_assets, dtype=object)[connected])
        ensemble = ensemble[np.ix_(connected, connected)]
        snapshots.append(
            GraphSnapshot(
                date=pd.Timestamp(features.index[position]),
                assets=kept_assets,
                adjacency=normalize_adjacency(ensemble),
                raw_adjacency=ensemble,
            )
        )
        if number % 25 == 0:
            LOGGER.info(
                "Grafos alpha=%g beta=%g: %d snapshots processados.",
                alpha,
                beta,
                number,
            )

    if not snapshots:
        raise RuntimeError(
            "Nenhum grafo foi aprendido. Verifique histórico, NaNs e graph.min_assets."
        )
    LOGGER.info(
        "Grafo alpha=%g beta=%g: %d snapshots entre %s e %s.",
        alpha,
        beta,
        len(snapshots),
        snapshots[0].date.date(),
        snapshots[-1].date.date(),
    )
    return snapshots


def propagate_network_features(
    features: pd.DataFrame,
    feature_names: tuple[str, ...],
    snapshots: list[GraphSnapshot],
) -> pd.DataFrame:
    """Eq. (7): ũ = Ã u, usando o ensemble bruto do snapshot e normalizando uma única
    vez sobre o subconjunto de ativos com features válidas na data."""
    if not snapshots:
        raise ValueError("snapshots está vazio.")
    snapshots = sorted(snapshots, key=lambda item: item.date)
    output: list[pd.DataFrame] = []
    pointer = -1

    for date in features.index:
        while pointer + 1 < len(snapshots) and snapshots[pointer + 1].date <= date:
            pointer += 1
        if pointer < 0:
            continue

        snapshot = snapshots[pointer]
        base_adjacency = (
            snapshot.raw_adjacency if snapshot.raw_adjacency is not None else snapshot.adjacency
        )
        columns = pd.MultiIndex.from_product(
            [snapshot.assets, feature_names],
            names=["ticker", "feature"],
        )
        current = features.loc[date].reindex(columns).to_numpy(dtype=float).reshape(
            len(snapshot.assets), len(feature_names)
        )
        valid = np.isfinite(current).all(axis=1)
        if valid.sum() < 2:
            continue
        active_assets = tuple(np.asarray(snapshot.assets, dtype=object)[valid])
        active_adjacency = normalize_adjacency(base_adjacency[np.ix_(valid, valid)])
        network_values = active_adjacency.dot(current[valid])
        frame = pd.DataFrame(
            network_values,
            index=pd.MultiIndex.from_product(
                [[pd.Timestamp(date)], active_assets],
                names=["date", "ticker"],
            ),
            columns=feature_names,
        )
        output.append(frame)

    if not output:
        raise RuntimeError("Não foi possível propagar features em nenhuma data.")
    return pd.concat(output).sort_index()


def graph_edges(snapshots: list[GraphSnapshot]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for snapshot in snapshots:
        edge_i, edge_j = np.triu_indices(len(snapshot.assets), k=1)
        weights = snapshot.adjacency[edge_i, edge_j]
        for i, j, weight in zip(edge_i, edge_j, weights, strict=True):
            if weight > 0:
                records.append(
                    {
                        "date": snapshot.date,
                        "source": snapshot.assets[i],
                        "target": snapshot.assets[j],
                        "weight": float(weight),
                    }
                )
    return pd.DataFrame.from_records(records)
