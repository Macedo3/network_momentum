from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np
import pandas as pd

from .config import FeatureConfig


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureSet:
    features: pd.DataFrame
    target_scaled_return: pd.DataFrame
    forward_return: pd.DataFrame
    annualized_volatility: pd.DataFrame
    feature_names: tuple[str, ...]
    daily_return: pd.DataFrame | None = None

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.features.index)

    @property
    def tickers(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self.features.columns.get_level_values("ticker")))


def _ewm_std(
    series: pd.Series,
    *,
    span: int | None = None,
    halflife: int | None = None,
    min_periods: int,
) -> pd.Series:
    kwargs = {
        "adjust": True,
        "min_periods": min_periods,
    }
    if span is not None:
        kwargs["span"] = span
    if halflife is not None:
        kwargs["halflife"] = halflife
    return series.ewm(**kwargs).std(bias=True)


def _winsorize_past_only(
    frame: pd.DataFrame,
    halflife: int,
    limit: float,
    min_periods: int,
) -> pd.DataFrame:
    mean = frame.ewm(
        halflife=halflife,
        adjust=True,
        min_periods=min_periods,
    ).mean()
    std = frame.ewm(
        halflife=halflife,
        adjust=True,
        min_periods=min_periods,
    ).std(bias=True)
    lower = mean - limit * std
    upper = mean + limit * std
    return frame.clip(lower=lower, upper=upper, axis=None)


def _features_for_ticker(
    prices: pd.Series,
    config: FeatureConfig,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    prices = prices.dropna().sort_index()
    daily_return = prices.pct_change(fill_method=None)
    daily_volatility = _ewm_std(
        daily_return,
        span=config.volatility_span,
        min_periods=config.volatility_min_periods,
    )

    columns: dict[str, pd.Series] = {}
    for lookback in config.return_lookbacks:
        name = f"volret_{lookback}d"
        columns[name] = prices.pct_change(lookback, fill_method=None).div(
            daily_volatility * np.sqrt(lookback)
        )

    price_std = prices.rolling(
        config.macd_price_std_window,
        min_periods=config.macd_price_std_window,
    ).std(ddof=0)
    for short, long in config.macd_scales:
        short_ewma = prices.ewm(alpha=1.0 / short, adjust=False).mean()
        long_ewma = prices.ewm(alpha=1.0 / long, adjust=False).mean()
        normalized = (short_ewma - long_ewma).div(price_std)
        normalizer = normalized.rolling(
            config.macd_norm_std_window,
            min_periods=config.macd_norm_std_window,
        ).std(ddof=0)
        columns[f"macd_{short}_{long}"] = normalized.div(normalizer)

    features = pd.DataFrame(columns, index=prices.index).replace([np.inf, -np.inf], np.nan)
    features = _winsorize_past_only(
        features,
        halflife=config.winsor_halflife,
        limit=config.winsor_limit,
        min_periods=config.volatility_min_periods,
    )
    if config.signal_lag_days:
        features = features.shift(config.signal_lag_days)

    forward_return = daily_return.shift(-1)
    target = forward_return.div(daily_volatility)
    annualized_volatility = daily_volatility * np.sqrt(252.0)
    return features, target, forward_return, annualized_volatility


def build_feature_set(prices: pd.DataFrame, config: FeatureConfig) -> FeatureSet:
    if prices.empty:
        raise ValueError("A matriz de preços está vazia.")
    calendar = pd.DatetimeIndex(prices.index).sort_values().unique()
    ticker_frames: dict[str, pd.DataFrame] = {}
    targets: dict[str, pd.Series] = {}
    forward_returns: dict[str, pd.Series] = {}
    annual_volatilities: dict[str, pd.Series] = {}
    daily_returns: dict[str, pd.Series] = {}

    for ticker in prices.columns:
        frame, target, forward, annual_vol = _features_for_ticker(prices[ticker], config)
        ticker_frames[ticker] = frame.reindex(calendar).ffill(limit=config.max_stale_days)
        targets[ticker] = target.reindex(calendar)
        forward_returns[ticker] = forward.reindex(calendar)
        annual_volatilities[ticker] = annual_vol.reindex(calendar)
        daily_returns[ticker] = (
            prices[ticker].dropna().sort_index().pct_change(fill_method=None).reindex(calendar)
        )

    features = pd.concat(ticker_frames, axis=1)
    features.columns = features.columns.set_names(["ticker", "feature"])
    features = features.sort_index(axis=1)
    feature_names = tuple(next(iter(ticker_frames.values())).columns)
    features = features.reindex(
        columns=pd.MultiIndex.from_product(
            [sorted(ticker_frames), feature_names],
            names=["ticker", "feature"],
        )
    )

    feature_set = FeatureSet(
        features=features,
        target_scaled_return=pd.DataFrame(targets, index=calendar),
        forward_return=pd.DataFrame(forward_returns, index=calendar),
        annualized_volatility=pd.DataFrame(annual_volatilities, index=calendar),
        feature_names=feature_names,
        daily_return=pd.DataFrame(daily_returns, index=calendar),
    )
    LOGGER.info(
        "Features calculadas: %d datas, %d ativos, %d sinais.",
        len(calendar),
        len(ticker_frames),
        len(feature_names),
    )
    return feature_set
