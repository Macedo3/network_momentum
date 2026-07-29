import numpy as np
import pandas as pd
import pytest

from network_momentum.benchmarks import macd_phi, run_benchmark_suite
from network_momentum.backtest import build_network_bundle
from network_momentum.config import GraphConfig, ModelConfig
from network_momentum.features import FeatureSet


def test_macd_phi_known_values() -> None:
    assert macd_phi(np.array([0.0]))[0] == pytest.approx(0.0)
    expected = 1.0 * np.exp(-0.25) / 0.89
    assert macd_phi(np.array([1.0]))[0] == pytest.approx(expected)
    assert macd_phi(np.array([-1.0]))[0] == pytest.approx(-expected)


def _synthetic_feature_set(seed: int = 19, n_days: int = 900):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n_days)
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
    columns = pd.MultiIndex.from_product([assets, feature_names], names=["ticker", "feature"])
    features = pd.DataFrame(rng.normal(size=(n_days, len(columns))), index=dates, columns=columns)
    target = pd.DataFrame(rng.normal(scale=0.8, size=(n_days, len(assets))), index=dates, columns=assets)
    return FeatureSet(
        features=features,
        target_scaled_return=target,
        forward_return=target * 0.01,
        annualized_volatility=pd.DataFrame(0.20, index=dates, columns=assets),
        feature_names=feature_names,
    ), assets


def test_suite_produces_all_strategies() -> None:
    feature_set, assets = _synthetic_feature_set()
    graph_config = GraphConfig(
        lookbacks=(20, 40),
        alpha_grid=(0.1,),
        beta_grid=(0.1,),
        rebalance_every=20,
        min_assets=4,
        maxiter=100,
    )
    model_config = ModelConfig(
        initial_train_years=1,
        test_years=1,
        min_regression_samples=100,
    )
    regions = pd.Series(dict(zip(assets, ["Americas", "Americas", "Europe", "Europe", "Asia", "Asia"])))
    bundle = build_network_bundle(feature_set, graph_config)
    suite = run_benchmark_suite(
        feature_set,
        regions,
        graph_config,
        model_config,
        network_bundle=bundle,
    )
    expected = {"gmom", "linreg", "regcombo", "long_only", "equal_weight", "macd", "signcombo"}
    assert expected.issubset(suite.results.keys())
    for name, result in suite.results.items():
        assert not result.daily_returns.empty, name
        assert np.isfinite(result.daily_returns["strategy_return"]).all(), name
    # long_only tem posição constante = 1 -> sign agreement consigo mesma = 1
    assert suite.sign_agreement.loc["long_only", "long_only"] == pytest.approx(1.0)
    # equal_weight: peso 1 para todos (agregação 1/N) -> exposição bruta = 1
    ew = suite.results["equal_weight"].daily_returns
    unscaled_exposure = ew["gross_exposure"] / ew["portfolio_leverage"]
    assert np.allclose(unscaled_exposure, 1.0, atol=1e-9)
    # correlação é uma matriz simétrica com diagonal 1
    corr = suite.correlation
    assert np.allclose(np.diag(corr.to_numpy()), 1.0)
