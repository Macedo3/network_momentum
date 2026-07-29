from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from network_momentum.costs import (
    breakeven_cost_bps,
    build_ticker_costs,
    capacity_estimate,
    load_cost_table,
    sharpe_versus_cost,
)

REPO_COSTS = Path(__file__).resolve().parents[1] / "config" / "costs.csv"


def _mini_universe() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "ticker": ["AAA.L", "BBB.HK", "CCC"],
            "exchange": ["LSE", "HKEX", "Nasdaq"],
            "currency": ["GBP", "HKD", "USD"],
            "borrow_fee_annual_bps_estimate": [np.nan, np.nan, np.nan],
        }
    )
    return frame.set_index("ticker")


def test_statutory_taxes_by_side() -> None:
    table = load_cost_table(REPO_COSTS)
    costs = build_ticker_costs(table, _mini_universe(), scenario="base", base_currency="USD")
    components = costs.components
    # SDRT do Reino Unido: 50 bps SOMENTE na compra.
    assert components.loc["AAA.L", "tax_buy_bps"] == pytest.approx(50.0)
    assert components.loc["AAA.L", "tax_sell_bps"] == pytest.approx(0.0)
    assert costs.buy_bps["AAA.L"] - costs.sell_bps["AAA.L"] == pytest.approx(50.0 - 0.0 + 0.1)
    # Stamp duty de Hong Kong: 10 bps nos DOIS lados.
    assert components.loc["BBB.HK", "tax_buy_bps"] == pytest.approx(10.0)
    assert components.loc["BBB.HK", "tax_sell_bps"] == pytest.approx(10.0)
    # Ativo em USD não paga conversão cambial.
    assert components.loc["CCC", "fx_conversion_bps"] == pytest.approx(0.0)
    assert components.loc["AAA.L", "fx_conversion_bps"] > 0


def test_scenarios_change_only_spread() -> None:
    table = load_cost_table(REPO_COSTS)
    universe = _mini_universe()
    conservative = build_ticker_costs(table, universe, scenario="conservative")
    optimistic = build_ticker_costs(table, universe, scenario="optimistic")
    delta_buy = conservative.buy_bps - optimistic.buy_bps
    delta_spread = (
        conservative.components["half_spread_bps"] - optimistic.components["half_spread_bps"]
    )
    pd.testing.assert_series_equal(delta_buy, delta_spread, check_names=False)


def test_breakeven_and_sweep() -> None:
    dates = pd.bdate_range("2022-01-03", periods=252)
    gross = pd.Series(0.0004, index=dates)
    turnover = pd.Series(0.5, index=dates)
    breakeven = breakeven_cost_bps(gross, turnover)
    assert breakeven == pytest.approx(0.0004 / 0.5 * 1e4)
    sweep = sharpe_versus_cost(gross, turnover, (0.0, breakeven))
    assert sweep.loc[breakeven, "net_annual_return"] == pytest.approx(0.0, abs=1e-12)


def test_capacity_estimate_binding_asset() -> None:
    dates = pd.bdate_range("2022-01-03", periods=100)
    turnover = pd.DataFrame({"AAA": 0.10, "BBB": 0.01}, index=dates)
    adv = pd.DataFrame({"AAA": 1e6, "BBB": 1e6}, index=dates)
    active = pd.Series(2.0, index=dates)
    capacity = capacity_estimate(turnover, adv, active, max_participation=0.05)
    assert capacity.index[0] == "AAA"  # ativo mais restritivo primeiro
    assert capacity.loc["AAA", "max_notional_base"] == pytest.approx(0.05 * 1e6 * 2 / 0.10)
