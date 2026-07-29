from __future__ import annotations

"""Benchmarks metodológicos (Seção 4 do artigo) e externos.

Metodológicos — todos avaliados sobre o MESMO suporte (data, ativo) da estratégia
principal, com a mesma Eq. (9), o mesmo vol targeting e o mesmo motor de custos:

- ``long_only``: x = 1 com escala de volatilidade (benchmark de mercado do artigo);
- ``equal_weight``: peso unitário (1/N implícito na agregação), sem escala de vol;
- ``macd``: Eq. (10), x = média de φ(y_k), φ(y) = y·exp(−y²/4)/0.89;
- ``linreg``: Eq. (11), regressão nas oito features individuais (mode="individual");
- ``gmom``: Eq. (8), regressão nas oito network features (mode="network");
- ``regcombo``: Eq. (12), features individuais e de rede juntas (mode="combo");
- ``signcombo``: x = ½·sign(LinReg) + ½·sign(GMOM).

Externos — índices/ETFs definidos em ``config/benchmarks.csv``, convertidos para a
moeda-base; ETFs de retorno total via adjusted close são *proxies* investíveis e o
tipo de cada série (price return vs total return) é reportado, nunca misturado em
silêncio.
"""

from dataclasses import dataclass
import logging

import numpy as np
import pandas as pd

from .backtest import BacktestResult, NetworkBundle, run_backtest
from .config import GraphConfig, ModelConfig
from .costs import TickerCosts
from .features import FeatureSet
from .metrics import relative_statistics, sign_agreement
from .portfolio import ImpactInputs, build_portfolio

LOGGER = logging.getLogger(__name__)


def macd_phi(signal: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    """Função de escala de posição da Eq. (10): φ(y) = y·exp(−y²/4)/0.89."""
    return signal * np.exp(-np.square(signal) / 4.0) / 0.89


@dataclass(frozen=True)
class BenchmarkSuite:
    results: dict[str, BacktestResult]
    correlation: pd.DataFrame
    sign_agreement: pd.DataFrame

    def daily_returns_frame(self, column: str = "strategy_return") -> pd.DataFrame:
        return pd.DataFrame(
            {name: res.daily_returns[column] for name, res in self.results.items()}
        )


def _rule_based_result(
    name: str,
    predictions: pd.DataFrame,
    model_config: ModelConfig,
    *,
    ticker_costs: TickerCosts | None,
    impact_inputs: ImpactInputs | None,
) -> BacktestResult:
    portfolio = build_portfolio(
        predictions,
        target_annual_volatility=model_config.target_annual_volatility,
        volatility_scaling=model_config.portfolio_volatility_scaling,
        volatility_span=model_config.portfolio_volatility_span,
        max_leverage=model_config.max_portfolio_leverage,
        pseudo_cost_bps=model_config.transaction_cost_bps,
        ticker_costs=ticker_costs,
        impact_inputs=impact_inputs,
    )
    predictions = predictions.copy()
    weights_long = portfolio.weights.stack()
    weights_long.name = "weight"
    weights_long.index = weights_long.index.set_names(["date", "ticker"])
    predictions = predictions.join(weights_long, how="left")
    predictions["gross_asset_return"] = (
        predictions["weight"] * predictions["forward_return"]
    )
    return BacktestResult(
        daily_returns=portfolio.daily,
        predictions=predictions,
        coefficients=pd.DataFrame(),
        folds=pd.DataFrame(),
        validation=pd.DataFrame(),
        snapshots={},
        mode=name,
        portfolio=portfolio,
    )


def _support_panel(reference: BacktestResult) -> pd.DataFrame:
    columns = [
        "forward_return",
        "target_scaled_return",
        "annualized_volatility",
        "region",
        "fold",
    ]
    if "forward_return_base" in reference.predictions.columns:
        columns.append("forward_return_base")
    return reference.predictions.loc[:, columns].copy()


def _macd_positions(
    feature_set: FeatureSet,
    support_index: pd.MultiIndex,
    macd_columns: tuple[str, ...],
) -> pd.Series:
    stacked = feature_set.features.stack(level="ticker", future_stack=True)
    stacked.index = stacked.index.set_names(["date", "ticker"])
    macd = stacked.loc[:, list(macd_columns)].reindex(support_index)
    return macd_phi(macd).mean(axis=1)


def run_benchmark_suite(
    feature_set: FeatureSet,
    regions: pd.Series,
    graph_config: GraphConfig,
    model_config: ModelConfig,
    *,
    network_bundle: NetworkBundle,
    forward_return_base: pd.DataFrame | None = None,
    ticker_costs: TickerCosts | None = None,
    impact_inputs: ImpactInputs | None = None,
    short_eligible: pd.Series | None = None,
    include_regcombo: bool = True,
) -> BenchmarkSuite:
    results: dict[str, BacktestResult] = {}

    common = dict(
        forward_return_base=forward_return_base,
        ticker_costs=ticker_costs,
        impact_inputs=impact_inputs,
        short_eligible=short_eligible,
    )
    results["gmom"] = run_backtest(
        feature_set,
        regions,
        graph_config,
        model_config,
        mode="network",
        network_bundle=network_bundle,
        **common,
    )
    results["linreg"] = run_backtest(
        feature_set,
        regions,
        graph_config,
        model_config,
        mode="individual",
        **common,
    )
    if include_regcombo:
        results["regcombo"] = run_backtest(
            feature_set,
            regions,
            graph_config,
            model_config,
            mode="combo",
            network_bundle=network_bundle,
            **common,
        )

    support = _support_panel(results["gmom"])
    rule_common = dict(ticker_costs=ticker_costs, impact_inputs=impact_inputs)

    long_only = support.copy()
    long_only["prediction"] = 1.0
    long_only["position"] = 1.0
    results["long_only"] = _rule_based_result(
        "long_only", long_only, model_config, **rule_common
    )

    equal_weight = support.copy()
    equal_weight["prediction"] = 1.0
    # posição = σ/σ_alvo faz o peso da Eq. (9) colapsar para 1 (peso igualitário).
    equal_weight["position"] = (
        equal_weight["annualized_volatility"] / model_config.target_annual_volatility
    )
    results["equal_weight"] = _rule_based_result(
        "equal_weight", equal_weight, model_config, **rule_common
    )

    macd_columns = tuple(
        name for name in feature_set.feature_names if name.startswith("macd_")
    )
    if macd_columns:
        macd = support.copy()
        macd["prediction"] = _macd_positions(feature_set, support.index, macd_columns)
        macd["position"] = macd["prediction"]
        macd = macd.dropna(subset=["position"])
        results["macd"] = _rule_based_result("macd", macd, model_config, **rule_common)

    lin_positions = results["linreg"].predictions["position"]
    gmom_positions = results["gmom"].predictions["position"]
    combo = support.copy()
    combined = 0.5 * np.sign(lin_positions.reindex(support.index)).fillna(0.0) + 0.5 * np.sign(
        gmom_positions.reindex(support.index)
    ).fillna(0.0)
    combo["prediction"] = combined
    combo["position"] = combined
    results["signcombo"] = _rule_based_result(
        "signcombo", combo, model_config, **rule_common
    )

    names = list(results)
    returns_frame = pd.DataFrame(
        {name: results[name].daily_returns["strategy_return"] for name in names}
    )
    correlation = returns_frame.corr()

    agreement = pd.DataFrame(np.nan, index=names, columns=names)
    for row in names:
        for column in names:
            agreement.loc[row, column] = sign_agreement(
                results[row].predictions["position"],
                results[column].predictions["position"],
            )

    return BenchmarkSuite(
        results=results,
        correlation=correlation,
        sign_agreement=agreement,
    )


def load_benchmark_meta(path) -> pd.DataFrame:
    meta = pd.read_csv(path, dtype=str).fillna("")
    required = {"ticker", "name", "currency", "region", "kind"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"Colunas ausentes em {path}: {sorted(missing)}")
    meta["ticker"] = meta["ticker"].str.strip().str.upper()
    meta["currency"] = meta["currency"].str.strip().str.upper()
    return meta.set_index("ticker")


def external_benchmark_returns(
    closes: pd.DataFrame,
    meta: pd.DataFrame,
    rates_to_base: pd.DataFrame,
    *,
    base_currency: str = "USD",
) -> pd.DataFrame:
    """Retornos diários dos benchmarks externos convertidos para a moeda-base."""
    rates = rates_to_base.reindex(closes.index).ffill()
    output: dict[str, pd.Series] = {}
    for ticker in closes.columns:
        if ticker not in meta.index:
            continue
        local = closes[ticker].pct_change(fill_method=None)
        currency = meta.loc[ticker, "currency"]
        if currency == base_currency:
            output[ticker] = local
        elif currency in rates.columns:
            fx_return = rates[currency].pct_change(fill_method=None)
            output[ticker] = (1.0 + local) * (1.0 + fx_return) - 1.0
        else:
            LOGGER.warning("Benchmark %s sem taxa cambial (%s); ignorado.", ticker, currency)
    return pd.DataFrame(output)


def compare_to_external(
    strategy_returns: pd.Series,
    external_returns: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    rows: list[pd.Series] = []
    for ticker in external_returns.columns:
        stats = relative_statistics(
            strategy_returns,
            external_returns[ticker],
            risk_free_rate=risk_free_rate,
        )
        if stats.empty:
            continue
        stats.name = ticker
        stats["benchmark_name"] = meta.loc[ticker, "name"] if ticker in meta.index else ticker
        stats["kind"] = meta.loc[ticker, "kind"] if ticker in meta.index else ""
        rows.append(stats)
    return pd.DataFrame(rows)
