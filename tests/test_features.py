import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from network_momentum.config import FeatureConfig
from network_momentum.features import build_feature_set


def _config() -> FeatureConfig:
    return FeatureConfig(
        volatility_span=10,
        volatility_min_periods=5,
        return_lookbacks=(1, 5, 10, 20, 30),
        macd_scales=((3, 6), (4, 8), (5, 10)),
        macd_price_std_window=10,
        macd_norm_std_window=20,
        winsor_halflife=20,
        winsor_limit=5.0,
        signal_lag_days=1,
        max_stale_days=2,
    )


def test_eight_features_and_no_future_leakage() -> None:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2020-01-01", periods=180)
    prices = pd.DataFrame(
        {
            "AAA": 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, len(dates)))),
            "BBB": 80 * np.exp(np.cumsum(rng.normal(0.0001, 0.012, len(dates)))),
        },
        index=dates,
    )
    original = build_feature_set(prices, _config())
    changed = prices.copy()
    changed.iloc[-1, :] *= 1.5
    modified = build_feature_set(changed, _config())

    assert len(original.feature_names) == 8
    assert_frame_equal(
        original.features.iloc[:-1],
        modified.features.iloc[:-1],
        check_exact=False,
        atol=1e-12,
        rtol=1e-12,
    )


def test_forward_target_uses_next_observed_return() -> None:
    dates = pd.bdate_range("2021-01-01", periods=80)
    prices = pd.DataFrame({"AAA": np.linspace(100.0, 140.0, len(dates))}, index=dates)
    result = build_feature_set(prices, _config())
    expected = prices["AAA"].pct_change(fill_method=None).shift(-1)
    pd.testing.assert_series_equal(
        result.forward_return["AAA"],
        expected,
        check_names=False,
    )

