import numpy as np
import pandas as pd
from pathlib import Path

from network_momentum.backtest import run_backtest
from network_momentum.config import GraphConfig, ModelConfig
from network_momentum.features import FeatureSet
from network_momentum.reporting import save_results


def test_synthetic_walk_forward_pipeline(safe_tmp_path: Path) -> None:
    rng = np.random.default_rng(19)
    dates = pd.bdate_range("2018-01-01", periods=820)
    assets = ("A", "B", "C", "D", "E", "F")
    feature_names = (
        "volret_1d",
        "volret_21d",
        "volret_63d",
        "volret_126d",
        "volret_252d",
        "macd_8_24",
        "macd_16_48",
        "macd_32_96",
    )
    columns = pd.MultiIndex.from_product(
        [assets, feature_names],
        names=["ticker", "feature"],
    )
    features = pd.DataFrame(
        rng.normal(size=(len(dates), len(columns))),
        index=dates,
        columns=columns,
    )
    target = pd.DataFrame(
        rng.normal(scale=0.8, size=(len(dates), len(assets))),
        index=dates,
        columns=assets,
    )
    forward = target * 0.01
    annual_vol = pd.DataFrame(0.20, index=dates, columns=assets)
    feature_set = FeatureSet(
        features=features,
        target_scaled_return=target,
        forward_return=forward,
        annualized_volatility=annual_vol,
        feature_names=feature_names,
    )
    graph_config = GraphConfig(
        lookbacks=(20, 40),
        alpha_grid=(0.1,),
        beta_grid=(0.1,),
        rebalance_every=20,
        min_assets=4,
        maxiter=150,
    )
    model_config = ModelConfig(
        initial_train_years=1,
        test_years=1,
        validation_fraction=0.1,
        portfolio_volatility_scaling=True,
        min_regression_samples=100,
    )
    regions = pd.Series(
        {
            "A": "Americas",
            "B": "Americas",
            "C": "Europe",
            "D": "Europe",
            "E": "Asia",
            "F": "Asia",
        }
    )
    result = run_backtest(feature_set, regions, graph_config, model_config)
    assert not result.daily_returns.empty
    assert not result.predictions.empty
    assert result.folds.shape[0] >= 2
    assert np.isfinite(result.daily_returns["strategy_return"]).all()
    assert result.predictions.index.is_unique
    paths = save_results(
        result,
        safe_tmp_path / "results",
        config_path=safe_tmp_path / "synthetic.toml",
    )
    assert all(path.exists() for path in paths.values())
