import numpy as np
import pandas as pd
import pytest

from network_momentum.validation import (
    block_bootstrap_sharpe,
    circular_shift_permutation_test,
    deflated_sharpe_ratio,
    multiple_testing_adjustment,
    probability_of_backtest_overfitting,
)


def test_bootstrap_reproducible_and_ci_contains_observed() -> None:
    rng = np.random.default_rng(11)
    returns = pd.Series(
        rng.normal(0.0006, 0.01, 1200),
        index=pd.bdate_range("2019-01-01", periods=1200),
    )
    first = block_bootstrap_sharpe(returns, n_samples=500, block_days=21, seed=42)
    second = block_bootstrap_sharpe(returns, n_samples=500, block_days=21, seed=42)
    np.testing.assert_array_equal(first.samples, second.samples)
    assert first.ci_low < first.observed_sharpe < first.ci_high


def test_permutation_detects_lookahead_signal() -> None:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2019-01-01", periods=800)
    returns = pd.DataFrame(
        rng.normal(0.0, 0.01, size=(800, 5)),
        index=dates,
        columns=list("ABCDE"),
    )
    # pesos construídos COM look-ahead de propósito: sinal perfeito
    weights = np.sign(returns)
    result = circular_shift_permutation_test(
        weights, returns, n_samples=200, min_shift_days=63, seed=42
    )
    assert result.observed_sharpe > 5
    assert result.p_value < 0.01


def test_deflated_sharpe_decreases_with_trials() -> None:
    rng = np.random.default_rng(3)
    returns = pd.Series(
        rng.normal(0.0008, 0.01, 1500),
        index=pd.bdate_range("2018-01-01", periods=1500),
    )
    dsr_1 = deflated_sharpe_ratio(returns, n_trials=1)
    dsr_100 = deflated_sharpe_ratio(returns, n_trials=100)
    assert (
        dsr_100["deflated_sharpe_probability"] < dsr_1["deflated_sharpe_probability"]
    )
    assert 0.0 <= dsr_100["deflated_sharpe_probability"] <= 1.0


def test_pbo_requires_two_configurations() -> None:
    dates = pd.bdate_range("2019-01-01", periods=400)
    single = pd.DataFrame({"only": np.random.default_rng(1).normal(size=400)}, index=dates)
    result = probability_of_backtest_overfitting(single, n_blocks=8)
    assert not result.applicable

    rng = np.random.default_rng(2)
    many = pd.DataFrame(
        {f"c{k}": rng.normal(0, 0.01, 400) for k in range(6)}, index=dates
    )
    result_many = probability_of_backtest_overfitting(many, n_blocks=8, seed=42)
    assert result_many.applicable
    assert 0.0 <= result_many.pbo <= 1.0
    # com configurações de puro ruído, o vencedor IS não deve persistir OOS
    assert result_many.pbo > 0.2


def test_benjamini_hochberg_known_case() -> None:
    p_values = pd.Series({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.005})
    adjusted = multiple_testing_adjustment(p_values)
    assert adjusted.loc["d", "bonferroni"] == pytest.approx(0.02)
    # BH: ordena 0.005,0.01,0.03,0.04 -> ajustes 0.02,0.02,0.04,0.04
    assert adjusted.loc["d", "benjamini_hochberg"] == pytest.approx(0.02)
    assert adjusted.loc["b", "benjamini_hochberg"] == pytest.approx(0.04)
