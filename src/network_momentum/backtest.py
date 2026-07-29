from __future__ import annotations

from dataclasses import dataclass, field
import itertools
import logging

import numpy as np
import pandas as pd

from .config import GraphConfig, ModelConfig
from .features import FeatureSet
from .graph import GraphSnapshot, build_graph_snapshots, propagate_network_features
from .model import OLSModel, fit_linear_model
from .portfolio import ImpactInputs, PortfolioResult, build_portfolio
from .costs import TickerCosts
from .splits import ExpandingWindow, expanding_windows


LOGGER = logging.getLogger(__name__)


Candidate = tuple[float, float]

FEATURE_MODES = ("network", "individual", "combo")


@dataclass(frozen=True)
class NetworkBundle:
    """Grafos e features propagadas por candidato (α, β) — a parte cara do pipeline,
    reutilizável entre estratégias (GMOM, RegCombo) e ablações."""

    snapshots: dict[Candidate, list[GraphSnapshot]]
    network_features: dict[Candidate, pd.DataFrame]


@dataclass(frozen=True)
class BacktestResult:
    daily_returns: pd.DataFrame
    predictions: pd.DataFrame
    coefficients: pd.DataFrame
    folds: pd.DataFrame
    validation: pd.DataFrame
    snapshots: dict[Candidate, list[GraphSnapshot]]
    mode: str = "network"
    portfolio: PortfolioResult | None = field(default=None, compare=False)


def build_network_bundle(
    feature_set: FeatureSet,
    graph_config: GraphConfig,
) -> NetworkBundle:
    candidates = list(itertools.product(graph_config.alpha_grid, graph_config.beta_grid))
    snapshots_by_candidate: dict[Candidate, list[GraphSnapshot]] = {}
    network_by_candidate: dict[Candidate, pd.DataFrame] = {}
    for alpha, beta in candidates:
        LOGGER.info("Construindo network features para alpha=%g, beta=%g.", alpha, beta)
        snapshots = build_graph_snapshots(
            feature_set.features,
            feature_set.feature_names,
            graph_config,
            alpha=alpha,
            beta=beta,
        )
        snapshots_by_candidate[(alpha, beta)] = snapshots
        network_by_candidate[(alpha, beta)] = propagate_network_features(
            feature_set.features,
            feature_set.feature_names,
            snapshots,
        )
    return NetworkBundle(
        snapshots=snapshots_by_candidate,
        network_features=network_by_candidate,
    )


def individual_features_long(feature_set: FeatureSet) -> pd.DataFrame:
    """Features individuais em formato longo (date, ticker) — insumo da Eq. (11)."""
    frame = feature_set.features.stack(level="ticker", future_stack=True)
    frame.index = frame.index.set_names(["date", "ticker"])
    return frame.dropna(how="all").sort_index()


def _wide_to_long(frame: pd.DataFrame, name: str) -> pd.Series:
    renamed = frame.rename_axis(index="date", columns="ticker")
    series = renamed.stack()
    series.name = name
    return series


def _base_panel(
    feature_set: FeatureSet,
    regions: pd.Series,
    forward_return_base: pd.DataFrame | None = None,
) -> pd.DataFrame:
    parts = [
        _wide_to_long(feature_set.target_scaled_return, "target_scaled_return"),
        _wide_to_long(feature_set.forward_return, "forward_return"),
        _wide_to_long(feature_set.annualized_volatility, "annualized_volatility"),
    ]
    panel = pd.concat(parts, axis=1)
    if forward_return_base is not None:
        base_long = _wide_to_long(forward_return_base, "forward_return_base")
        panel = panel.join(base_long, how="left")
    panel["region"] = panel.index.get_level_values("ticker").map(regions)
    return panel


def _mode_features(
    mode: str,
    candidate_network: pd.DataFrame,
    individual: pd.DataFrame | None,
    feature_names: tuple[str, ...],
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    if mode == "network":
        return candidate_network, feature_names
    if mode == "individual":
        assert individual is not None
        return individual, feature_names
    if mode == "combo":
        assert individual is not None
        network_renamed = candidate_network.rename(
            columns={name: f"net_{name}" for name in feature_names}
        )
        individual_renamed = individual.rename(
            columns={name: f"ind_{name}" for name in feature_names}
        )
        combined = individual_renamed.join(network_renamed, how="inner")
        names = tuple(individual_renamed.columns) + tuple(network_renamed.columns)
        return combined, names
    raise ValueError(f"mode deve ser um de {FEATURE_MODES}.")


def _date_region_slice(
    model_features: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    regions: tuple[str, ...],
) -> pd.DataFrame:
    dates = model_features.index.get_level_values("date")
    selected = model_features.loc[(dates >= start) & (dates <= end)]
    data = selected.join(panel, how="inner")
    if regions:
        data = data[data["region"].isin(regions)]
    data = data.replace([np.inf, -np.inf], np.nan)
    protected = [c for c in ("forward_return_base",) if c in data.columns]
    check_columns = [c for c in data.columns if c not in protected]
    return data.dropna(subset=check_columns)


def _positions_from_prediction(
    prediction: pd.Series,
    *,
    signal_threshold: float,
) -> pd.Series:
    positions = np.sign(prediction)
    if signal_threshold > 0:
        positions = positions.where(prediction.abs() >= signal_threshold, 0.0)
    return positions


def _validation_sharpe(
    model: OLSModel,
    data: pd.DataFrame,
    feature_names: tuple[str, ...],
    config: ModelConfig,
) -> float:
    if data.empty:
        return -np.inf
    prediction = model.predict(data.loc[:, list(feature_names)])
    positions = _positions_from_prediction(
        prediction, signal_threshold=config.signal_threshold
    )
    contribution = (
        positions
        * config.target_annual_volatility
        / np.sqrt(252.0)
        * data["target_scaled_return"]
    )
    daily = contribution.groupby(level="date").mean()
    volatility = float(daily.std(ddof=0))
    return float(daily.mean() / volatility * np.sqrt(252.0)) if volatility > 0 else -np.inf


def _fit_kwargs(config: ModelConfig) -> dict[str, object]:
    return {
        "method": config.regression_method,
        "ridge_lambda": config.ridge_lambda,
        "lasso_lambda": config.lasso_lambda,
        "elastic_net_l1_ratio": config.elastic_net_l1_ratio,
    }


def _fit_candidate_on_validation(
    model_features: pd.DataFrame,
    panel: pd.DataFrame,
    window: ExpandingWindow,
    feature_names: tuple[str, ...],
    config: ModelConfig,
) -> float:
    train_dates = pd.DatetimeIndex(
        model_features.index.get_level_values("date").unique()
    )
    train_dates = train_dates[
        (train_dates >= window.train_start) & (train_dates <= window.train_end)
    ]
    if len(train_dates) < 2:
        return -np.inf
    validation_size = max(1, int(np.ceil(len(train_dates) * config.validation_fraction)))
    core_end = pd.Timestamp(train_dates[-validation_size - 1])
    validation_start = pd.Timestamp(train_dates[-validation_size])
    core = _date_region_slice(
        model_features,
        panel,
        start=window.train_start,
        end=core_end,
        regions=config.train_regions,
    )
    validation = _date_region_slice(
        model_features,
        panel,
        start=validation_start,
        end=window.train_end,
        regions=config.train_regions,
    )
    try:
        model = fit_linear_model(
            core.loc[:, list(feature_names)],
            core["target_scaled_return"],
            min_samples=config.min_regression_samples,
            **_fit_kwargs(config),
        )
    except ValueError:
        return -np.inf
    return _validation_sharpe(model, validation, feature_names, config)


def run_backtest(
    feature_set: FeatureSet,
    regions: pd.Series,
    graph_config: GraphConfig,
    model_config: ModelConfig,
    *,
    mode: str = "network",
    network_bundle: NetworkBundle | None = None,
    forward_return_base: pd.DataFrame | None = None,
    ticker_costs: TickerCosts | None = None,
    impact_inputs: ImpactInputs | None = None,
    short_eligible: pd.Series | None = None,
) -> BacktestResult:
    """Walk-forward expansivo com seleção nested de (α, β) na validação (últimos
    10% do treino), como no artigo. ``mode`` escolhe as features da regressão:
    ``network`` (GMOM, Eq. 8), ``individual`` (LinReg, Eq. 11) ou ``combo``
    (RegCombo, Eq. 12)."""
    if mode not in FEATURE_MODES:
        raise ValueError(f"mode deve ser um de {FEATURE_MODES}.")

    needs_graph = mode in ("network", "combo")
    if needs_graph:
        if network_bundle is None:
            network_bundle = build_network_bundle(feature_set, graph_config)
        candidates = list(network_bundle.network_features.keys())
        snapshots_by_candidate = network_bundle.snapshots
    else:
        candidates = [(np.nan, np.nan)]
        snapshots_by_candidate = {}

    individual = (
        individual_features_long(feature_set) if mode in ("individual", "combo") else None
    )
    panel = _base_panel(feature_set, regions, forward_return_base)
    windows = expanding_windows(
        feature_set.dates,
        initial_train_years=model_config.initial_train_years,
        test_years=model_config.test_years,
        embargo_days=model_config.embargo_days,
    )

    features_by_candidate: dict[Candidate, tuple[pd.DataFrame, tuple[str, ...]]] = {}
    for candidate in candidates:
        if needs_graph:
            candidate_network = network_bundle.network_features[candidate]
        else:
            candidate_network = pd.DataFrame()
        features_by_candidate[candidate] = _mode_features(
            mode, candidate_network, individual, feature_set.feature_names
        )

    prediction_frames: list[pd.DataFrame] = []
    coefficient_records: list[dict[str, object]] = []
    fold_records: list[dict[str, object]] = []
    validation_records: list[dict[str, object]] = []

    for window in windows:
        scores: dict[Candidate, float] = {}
        for candidate, (model_features, feature_names) in features_by_candidate.items():
            score = _fit_candidate_on_validation(
                model_features,
                panel,
                window,
                feature_names,
                model_config,
            )
            scores[candidate] = score
            validation_records.append(
                {
                    "fold": window.fold,
                    "alpha": candidate[0],
                    "beta": candidate[1],
                    "validation_sharpe": score,
                }
            )
        selected = max(scores, key=scores.get)
        if not np.isfinite(scores[selected]):
            if len(candidates) == 1:
                LOGGER.warning(
                    "Validação indisponível no fold %d; usando o único candidato.",
                    window.fold,
                )
            else:
                raise RuntimeError(
                    f"Nenhum hiperparâmetro válido na validação do fold {window.fold}."
                )

        model_features, feature_names = features_by_candidate[selected]
        train = _date_region_slice(
            model_features,
            panel,
            start=window.train_start,
            end=window.train_end,
            regions=model_config.train_regions,
        )
        test = _date_region_slice(
            model_features,
            panel,
            start=window.test_start,
            end=window.test_end,
            regions=model_config.test_regions,
        )
        model = fit_linear_model(
            train.loc[:, list(feature_names)],
            train["target_scaled_return"],
            min_samples=model_config.min_regression_samples,
            compute_clustered_errors=model_config.regression_method == "ols",
            **_fit_kwargs(model_config),
        )
        test = test.copy()
        test["prediction"] = model.predict(test.loc[:, list(feature_names)])
        test["position"] = _positions_from_prediction(
            test["prediction"], signal_threshold=model_config.signal_threshold
        )
        if short_eligible is not None:
            tickers = test.index.get_level_values("ticker")
            allowed = tickers.map(short_eligible).fillna(True).astype(bool)
            test.loc[~allowed.to_numpy() & (test["position"] < 0), "position"] = 0.0
        test["target_annual_volatility"] = model_config.target_annual_volatility
        test["fold"] = window.fold
        test["alpha"] = selected[0]
        test["beta"] = selected[1]
        prediction_frames.append(test)

        errors = model.standard_error_series()
        coefficient_records.append(
            {
                "fold": window.fold,
                "feature": "intercept",
                "coefficient": model.intercept,
                "standard_error": np.nan,
                "samples": model.sample_count,
                "rank": model.rank,
            }
        )
        for feature, coefficient in model.coefficient_series().items():
            coefficient_records.append(
                {
                    "fold": window.fold,
                    "feature": feature,
                    "coefficient": coefficient,
                    "standard_error": (
                        float(errors[feature]) if errors is not None else np.nan
                    ),
                    "samples": model.sample_count,
                    "rank": model.rank,
                }
            )
        fold_records.append(
            {
                "fold": window.fold,
                "train_start": window.train_start,
                "train_end": window.train_end,
                "test_start": window.test_start,
                "test_end": window.test_end,
                "alpha": selected[0],
                "beta": selected[1],
                "train_sharpe": _validation_sharpe(
                    model, train, feature_names, model_config
                ),
                "validation_sharpe": scores[selected],
                "train_samples": model.sample_count,
                "test_samples": len(test),
            }
        )
        LOGGER.info(
            "Fold %d (%s): treino %s..%s, teste %s..%s, alpha=%s beta=%s.",
            window.fold,
            mode,
            window.train_start.date(),
            window.train_end.date(),
            window.test_start.date(),
            window.test_end.date(),
            selected[0],
            selected[1],
        )

    predictions = pd.concat(prediction_frames).sort_index()
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
    daily = portfolio.daily.copy()
    fold_by_date = pd.Series(
        {
            pd.Timestamp(row["test_start"]): row["fold"]
            for row in fold_records
        }
    ).sort_index()
    daily["fold"] = (
        fold_by_date.reindex(daily.index, method="ffill").fillna(0).astype(int)
    )

    weights_long = portfolio.weights.stack()
    weights_long.name = "weight"
    weights_long.index = weights_long.index.set_names(["date", "ticker"])
    predictions = predictions.join(weights_long, how="left")
    predictions["gross_asset_return"] = (
        predictions["weight"] * predictions["forward_return"]
    )

    return BacktestResult(
        daily_returns=daily,
        predictions=predictions,
        coefficients=pd.DataFrame.from_records(coefficient_records),
        folds=pd.DataFrame.from_records(fold_records),
        validation=pd.DataFrame.from_records(validation_records),
        snapshots=snapshots_by_candidate,
        mode=mode,
        portfolio=portfolio,
    )
