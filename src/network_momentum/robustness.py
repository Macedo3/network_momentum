from __future__ import annotations

"""Ablações e análises de robustez.

Dois grupos de ferramentas:

1. **Sem reajuste** (operam nas previsões existentes; aproximação declarada:
   remover ativos/regiões da agregação NÃO reestima a regressão nem o grafo):
   decomposição long/short, métricas por grupo, contribuição por ativo,
   concentração, remoção dos melhores ativos/anos, sensibilidade à data inicial
   e regimes de volatilidade.

2. **Com reajuste** (reexecutam o walk-forward reutilizando os grafos já
   aprendidos): ablação de features na regressão, ablação de arestas
   intra/inter grupo e ablação de lookbacks (esta última reaprende grafos).
"""

from dataclasses import replace
import logging

import numpy as np
import pandas as pd

from .backtest import BacktestResult, NetworkBundle, run_backtest
from .config import GraphConfig, ModelConfig
from .features import FeatureSet
from .graph import build_graph_snapshots, propagate_network_features
from .metrics import performance_metrics
from .topology import EdgeMaskSpec, mask_snapshot_edges

LOGGER = logging.getLogger(__name__)

TRADING_DAYS = 252.0


# ---------------------------------------------------------------------------
# Grupo 1 — análises sem reajuste
# ---------------------------------------------------------------------------

def _daily_mean(predictions: pd.DataFrame, column: str = "gross_asset_return") -> pd.Series:
    return predictions.groupby(level="date")[column].mean()


def long_short_decomposition(predictions: pd.DataFrame) -> pd.DataFrame:
    """Retornos diários das pernas comprada e vendida (contribuições brutas,
    divididas pelo número TOTAL de ativos ativos do dia, para que long + short
    = retorno bruto do portfólio)."""
    frame = predictions.copy()
    active = frame.groupby(level="date")["gross_asset_return"].size()
    long_contribution = frame["gross_asset_return"].where(frame["weight"] > 0, 0.0)
    short_contribution = frame["gross_asset_return"].where(frame["weight"] < 0, 0.0)
    output = pd.DataFrame(
        {
            "long_return": long_contribution.groupby(level="date").sum() / active,
            "short_return": short_contribution.groupby(level="date").sum() / active,
        }
    )
    output["total"] = output["long_return"] + output["short_return"]
    return output


def metrics_by_group(
    predictions: pd.DataFrame,
    labels: pd.Series,
    *,
    group_name: str,
) -> pd.DataFrame:
    """Métricas do retorno bruto médio dentro de cada grupo (região, bolsa, moeda,
    setor). Nota: são carteiras parciais não reescaladas — servem para atribuição,
    não como estratégias independentes."""
    frame = predictions.copy()
    frame[group_name] = frame.index.get_level_values("ticker").map(labels)
    rows: list[pd.Series] = []
    for group, block in frame.groupby(group_name):
        returns = _daily_mean(block)
        row = performance_metrics(returns)
        row.name = group
        rows.append(row)
    return pd.DataFrame(rows).rename_axis(group_name)


def asset_contribution(predictions: pd.DataFrame) -> pd.DataFrame:
    """Contribuição acumulada de cada ativo para o retorno bruto do portfólio
    (soma de peso × retorno ÷ ativos ativos no dia)."""
    frame = predictions.copy()
    active = frame.groupby(level="date")["gross_asset_return"].transform("size")
    frame["portfolio_contribution"] = frame["gross_asset_return"] / active
    grouped = frame.groupby(level="ticker")
    output = pd.DataFrame(
        {
            "total_contribution": grouped["portfolio_contribution"].sum(),
            "days_active": grouped["portfolio_contribution"].size(),
            "hit_rate": grouped["portfolio_contribution"].apply(
                lambda s: float((s > 0).mean())
            ),
        }
    )
    return output.sort_values("total_contribution", ascending=False)


def concentration_statistics(contribution: pd.DataFrame) -> pd.Series:
    """Concentração de retorno (HHI e participação do top-k) sobre contribuições
    absolutas."""
    values = contribution["total_contribution"].abs()
    total = values.sum()
    if total <= 0:
        return pd.Series(dtype=float)
    shares = values / total
    sorted_shares = shares.sort_values(ascending=False)
    return pd.Series(
        {
            "hhi": float((shares**2).sum()),
            "effective_n_assets": float(1.0 / (shares**2).sum()),
            "top_1_share": float(sorted_shares.iloc[:1].sum()),
            "top_5_share": float(sorted_shares.iloc[:5].sum()),
            "top_10_share": float(sorted_shares.iloc[:10].sum()),
        }
    )


def risk_concentration(weights: pd.DataFrame) -> pd.Series:
    """Concentração média de risco (HHI dos |pesos| normalizados, média no tempo)."""
    absolute = weights.abs()
    totals = absolute.sum(axis=1)
    shares = absolute.div(totals.replace(0.0, np.nan), axis=0)
    hhi = (shares**2).sum(axis=1)
    return pd.Series(
        {
            "mean_weight_hhi": float(hhi.mean()),
            "mean_effective_positions": float((1.0 / hhi.replace(0.0, np.nan)).mean()),
        }
    )


def drop_assets_daily_returns(
    predictions: pd.DataFrame,
    exclude: list[str],
) -> pd.Series:
    """Retorno bruto diário recalculado sem os ativos excluídos (sem reajuste da
    regressão — aproximação declarada)."""
    tickers = predictions.index.get_level_values("ticker")
    return _daily_mean(predictions[~tickers.isin(exclude)])


def remove_best_assets_test(
    predictions: pd.DataFrame,
    *,
    top_k_list: tuple[int, ...] = (1, 3, 5),
) -> pd.DataFrame:
    contribution = asset_contribution(predictions)
    rows = []
    base = performance_metrics(_daily_mean(predictions))
    rows.append(pd.Series({**base.to_dict(), "removed": 0, "removed_assets": ""}))
    for k in top_k_list:
        best = contribution.index[:k].tolist()
        returns = drop_assets_daily_returns(predictions, best)
        row = performance_metrics(returns)
        row["removed"] = k
        row["removed_assets"] = ",".join(best)
        rows.append(row)
    return pd.DataFrame(rows).set_index("removed")


def remove_best_years_test(
    returns: pd.Series,
    *,
    top_k_list: tuple[int, ...] = (1, 2, 3),
) -> pd.DataFrame:
    yearly = returns.groupby(returns.index.year).sum().sort_values(ascending=False)
    rows = []
    base = performance_metrics(returns)
    rows.append(pd.Series({**base.to_dict(), "removed_years": ""}, name=0))
    for k in top_k_list:
        best_years = yearly.index[:k].tolist()
        filtered = returns[~returns.index.year.isin(best_years)]
        row = performance_metrics(filtered)
        row["removed_years"] = ",".join(str(y) for y in best_years)
        row.name = k
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.index.name = "n_removed"
    return frame


def start_date_sensitivity(
    returns: pd.Series,
    *,
    offsets_months: tuple[int, ...] = (0, 6, 12, 24, 36),
) -> pd.DataFrame:
    rows = []
    first = returns.index.min()
    for offset in offsets_months:
        start = first + pd.DateOffset(months=offset)
        subset = returns[returns.index >= start]
        if len(subset) < 252:
            continue
        row = performance_metrics(subset)
        row.name = str(start.date())
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.index.name = "start_date"
    return frame


def volatility_regime_metrics(
    strategy_returns: pd.Series,
    market_returns: pd.Series,
    *,
    window: int = 63,
    n_regimes: int = 3,
) -> pd.DataFrame:
    """Métricas por regime de volatilidade do mercado (tercis da volatilidade
    realizada móvel do benchmark long-only — construção sem dados externos)."""
    realized = market_returns.rolling(window, min_periods=window // 2).std(ddof=0)
    realized = realized.reindex(strategy_returns.index).ffill()
    labels = pd.qcut(realized, n_regimes, labels=False, duplicates="drop")
    names = {0: "baixa_vol", 1: "media_vol", 2: "alta_vol"}
    rows = []
    for regime, block in strategy_returns.groupby(labels):
        row = performance_metrics(block)
        row.name = names.get(int(regime), f"regime_{regime}")
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.index.name = "volatility_regime"
    return frame


def calendar_window_metrics(
    returns: pd.Series,
    windows: dict[str, tuple[str, str]] | None = None,
) -> pd.DataFrame:
    windows = windows or {
        "selloff_2015_2016": ("2015-06-01", "2016-02-29"),
        "covid_crash_2020": ("2020-02-15", "2020-04-30"),
        "hiking_cycle_2022": ("2022-01-01", "2022-12-31"),
        "ai_rally_2023_2024": ("2023-01-01", "2024-12-31"),
    }
    rows = []
    for name, (start, end) in windows.items():
        subset = returns[(returns.index >= start) & (returns.index <= end)]
        if subset.empty:
            continue
        row = performance_metrics(subset)
        row.name = name
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.index.name = "window"
    return frame


# ---------------------------------------------------------------------------
# Grupo 2 — ablações com reajuste
# ---------------------------------------------------------------------------

def ablate_bundle_features(
    bundle: NetworkBundle,
    drop_features: tuple[str, ...],
) -> NetworkBundle:
    """Remove features das matrizes propagadas (a regressão deixa de vê-las; o
    grafo permanece aprendido com as oito — escopo declarado da ablação)."""
    trimmed = {
        candidate: frame.drop(columns=list(drop_features), errors="ignore")
        for candidate, frame in bundle.network_features.items()
    }
    return NetworkBundle(snapshots=bundle.snapshots, network_features=trimmed)


def run_feature_ablation(
    feature_set: FeatureSet,
    regions: pd.Series,
    graph_config: GraphConfig,
    model_config: ModelConfig,
    *,
    network_bundle: NetworkBundle,
    groups: dict[str, tuple[str, ...]] | None = None,
) -> pd.DataFrame:
    """Reexecuta o walk-forward removendo grupos de features da regressão.

    Grupos padrão: sem MACD, sem retornos curtos (1/21d), sem retornos longos
    (126/252d). A comparação "apenas momentum individual" vs "apenas network"
    vs "combinação" é feita pelos modos do backtest (linreg/gmom/regcombo).
    """
    groups = groups or {
        "sem_macd": ("macd_8_24", "macd_16_48", "macd_32_96"),
        "sem_retornos_curtos": ("volret_1d", "volret_21d"),
        "sem_retornos_longos": ("volret_126d", "volret_252d"),
    }
    rows = []
    for name, drop in groups.items():
        remaining = tuple(f for f in feature_set.feature_names if f not in drop)
        if len(remaining) < 2:
            LOGGER.warning("Ablação %s deixaria menos de 2 features; ignorada.", name)
            continue
        trimmed_bundle = ablate_bundle_features(network_bundle, drop)
        trimmed_set = FeatureSet(
            features=feature_set.features,
            target_scaled_return=feature_set.target_scaled_return,
            forward_return=feature_set.forward_return,
            annualized_volatility=feature_set.annualized_volatility,
            feature_names=remaining,
            daily_return=feature_set.daily_return,
        )
        result = run_backtest(
            trimmed_set,
            regions,
            graph_config,
            model_config,
            mode="network",
            network_bundle=trimmed_bundle,
        )
        row = performance_metrics(result.daily_returns["strategy_return"])
        row["dropped"] = ",".join(drop)
        row.name = name
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.index.name = "ablation"
    return frame


def masked_edge_bundle(
    feature_set: FeatureSet,
    bundle: NetworkBundle,
    spec: EdgeMaskSpec,
) -> NetworkBundle:
    """Bundle com arestas intra ou inter grupo zeradas e features re-propagadas."""
    masked_snapshots = {
        candidate: [mask_snapshot_edges(s, spec) for s in snapshots]
        for candidate, snapshots in bundle.snapshots.items()
    }
    masked_features = {}
    for candidate, snapshots in masked_snapshots.items():
        usable = [s for s in snapshots if s.raw_adjacency is not None and s.raw_adjacency.sum() > 0]
        if not usable:
            LOGGER.warning("Máscara %s removeu todas as arestas do candidato %s.", spec.mode, candidate)
            continue
        masked_features[candidate] = propagate_network_features(
            feature_set.features,
            feature_set.feature_names,
            usable,
        )
    return NetworkBundle(
        snapshots={c: s for c, s in masked_snapshots.items() if c in masked_features},
        network_features=masked_features,
    )


def run_edge_mask_ablation(
    feature_set: FeatureSet,
    regions: pd.Series,
    graph_config: GraphConfig,
    model_config: ModelConfig,
    *,
    network_bundle: NetworkBundle,
    group_labels: dict[str, dict[str, str]],
) -> pd.DataFrame:
    """GMOM-Intra e GMOM-Inter (Seção 5.2 do artigo) para cada agrupamento
    (ex.: região e setor)."""
    rows = []
    for group_name, labels in group_labels.items():
        for mode in ("intra", "inter"):
            spec = EdgeMaskSpec(mode=mode, labels=labels)
            try:
                masked = masked_edge_bundle(feature_set, network_bundle, spec)
                if not masked.network_features:
                    continue
                result = run_backtest(
                    feature_set,
                    regions,
                    graph_config,
                    model_config,
                    mode="network",
                    network_bundle=masked,
                )
            except (RuntimeError, ValueError) as error:
                LOGGER.warning("Ablação %s/%s falhou: %s", group_name, mode, error)
                continue
            row = performance_metrics(result.daily_returns["strategy_return"])
            row.name = f"{group_name}_{mode}"
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame.index.name = "edge_ablation"
    return frame


def run_lookback_ablation(
    feature_set: FeatureSet,
    regions: pd.Series,
    graph_config: GraphConfig,
    model_config: ModelConfig,
    *,
    lookbacks: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    """Grafo com um único lookback por vez versus o ensemble (Seção 5.3 do artigo).
    Reaprende os grafos — é a ablação mais cara."""
    lookbacks = lookbacks or graph_config.lookbacks
    rows = []
    for lookback in lookbacks:
        single_config = replace(graph_config, lookbacks=(lookback,))
        try:
            result = run_backtest(
                feature_set,
                regions,
                single_config,
                model_config,
                mode="network",
            )
        except (RuntimeError, ValueError) as error:
            LOGGER.warning("Ablação de lookback %d falhou: %s", lookback, error)
            continue
        row = performance_metrics(result.daily_returns["strategy_return"])
        row["mean_turnover"] = float(result.daily_returns["turnover_final"].mean())
        row.name = f"lookback_{lookback}"
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.index.name = "graph_lookback"
    return frame
