import numpy as np
import pandas as pd
import pytest

from network_momentum.costs import TickerCosts
from network_momentum.portfolio import build_portfolio


def _predictions(positions: dict[str, list[float]], forward: dict[str, list[float]]) -> pd.DataFrame:
    dates = pd.bdate_range("2022-01-03", periods=len(next(iter(positions.values()))))
    rows = []
    for ticker, values in positions.items():
        for date, position, fwd in zip(dates, values, forward[ticker]):
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "position": position,
                    "forward_return": fwd,
                    "annualized_volatility": 0.30,
                }
            )
    return pd.DataFrame(rows).set_index(["date", "ticker"]).sort_index()


def _flat_costs(tickers: list[str], *, tax_buy: float = 0.0, borrow: float = 0.0) -> TickerCosts:
    index = pd.Index(tickers)
    components = pd.DataFrame(
        {
            "commission_bps": 0.0,
            "exchange_fee_bps": 0.0,
            "regulatory_buy_bps": 0.0,
            "regulatory_sell_bps": 0.0,
            "tax_buy_bps": tax_buy,
            "tax_sell_bps": 0.0,
            "half_spread_bps": 0.0,
            "fx_conversion_bps": 0.0,
            "borrow_fee_annual_bps": borrow,
        },
        index=index,
    )
    return TickerCosts(
        buy_bps=components["tax_buy_bps"],
        sell_bps=components["tax_sell_bps"],
        borrow_daily_bps=components["borrow_fee_annual_bps"] / 252.0,
        components=components,
        scenario="base",
    )


def test_turnover_matches_equation_13() -> None:
    # peso = posição × 0.15/0.30 = ±0.5; sem vol scaling para isolar a Eq. (13)
    predictions = _predictions(
        {"AAA": [1.0, 1.0, -1.0], "BBB": [1.0, 1.0, 1.0]},
        {"AAA": [0.01, 0.01, 0.01], "BBB": [0.0, 0.0, 0.0]},
    )
    result = build_portfolio(
        predictions,
        target_annual_volatility=0.15,
        volatility_scaling=False,
        pseudo_cost_bps=0.0,
    )
    turnover = result.daily.loc[:, "turnover_final"]
    assert turnover.iloc[0] == pytest.approx((0.5 + 0.5) / 2)  # entrada inicial
    assert turnover.iloc[1] == pytest.approx(0.0)
    assert turnover.iloc[2] == pytest.approx(1.0 / 2)  # AAA vira de +0.5 p/ -0.5


def test_asset_leaving_pays_exit_cost() -> None:
    predictions = _predictions(
        {"AAA": [1.0, 1.0, 1.0], "BBB": [1.0, 1.0, 1.0]},
        {"AAA": [0.0, 0.0, 0.0], "BBB": [0.0, 0.0, 0.0]},
    )
    predictions = predictions.drop((predictions.index.levels[0][2], "BBB"))
    result = build_portfolio(
        predictions,
        target_annual_volatility=0.15,
        volatility_scaling=False,
        pseudo_cost_bps=0.0,
    )
    # BBB sai no último dia: |0 - 0.5| = 0.5, dividido por 1 ativo ativo
    assert result.daily["turnover_final"].iloc[2] == pytest.approx(0.5)


def test_tax_applies_only_to_buys_and_borrow_to_shorts() -> None:
    predictions = _predictions(
        {"AAA": [1.0, -1.0], "BBB": [-1.0, -1.0]},
        {"AAA": [0.0, 0.0], "BBB": [0.0, 0.0]},
    )
    costs = _flat_costs(["AAA", "BBB"], tax_buy=100.0, borrow=252.0)
    result = build_portfolio(
        predictions,
        target_annual_volatility=0.15,
        volatility_scaling=False,
        pseudo_cost_bps=0.0,
        ticker_costs=costs,
    )
    daily = result.daily
    # dia 1: compra de AAA (+0.5) paga 100 bps; venda (short) de BBB não paga tax_buy
    assert daily["cost_tax"].iloc[0] == pytest.approx(0.5 * 100 / 1e4 / 2)
    # dia 2: AAA vende 1.0 (sem tax); ninguém compra => tax = 0
    assert daily["cost_tax"].iloc[1] == pytest.approx(0.0)
    # borrow: dia 1 só BBB short (0.5 × 1bp diário); dia 2 AAA e BBB short (1.0 × 1bp)
    assert daily["cost_borrow"].iloc[0] == pytest.approx(0.5 * 1.0 / 1e4 / 2)
    assert daily["cost_borrow"].iloc[1] == pytest.approx(1.0 * 1.0 / 1e4 / 2)


def test_volatility_scaling_is_causal_and_capped() -> None:
    rng = np.random.default_rng(3)
    n = 300
    dates = pd.bdate_range("2021-01-04", periods=n)
    rows = []
    for ticker in ("AAA", "BBB", "CCC"):
        forward = rng.normal(0.0, 0.01, n)
        for date, fwd in zip(dates, forward):
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "position": 1.0,
                    "forward_return": fwd,
                    "annualized_volatility": 0.25,
                }
            )
    predictions = pd.DataFrame(rows).set_index(["date", "ticker"]).sort_index()
    result = build_portfolio(
        predictions,
        target_annual_volatility=0.15,
        volatility_scaling=True,
        max_leverage=2.0,
        pseudo_cost_bps=0.0,
    )
    assert result.daily["portfolio_leverage"].max() <= 2.0 + 1e-12

    # causalidade: alterar o retorno do último dia não muda a alavancagem anterior
    modified = predictions.copy()
    last_date = modified.index.levels[0][-1]
    modified.loc[(last_date, slice(None)), "forward_return"] = 0.10
    result_modified = build_portfolio(
        modified,
        target_annual_volatility=0.15,
        volatility_scaling=True,
        max_leverage=2.0,
        pseudo_cost_bps=0.0,
    )
    pd.testing.assert_series_equal(
        result.daily["portfolio_leverage"].iloc[:-1],
        result_modified.daily["portfolio_leverage"].iloc[:-1],
    )


def test_base_currency_stream() -> None:
    predictions = _predictions(
        {"AAA": [1.0, 1.0]},
        {"AAA": [0.01, 0.01]},
    )
    predictions["forward_return_base"] = [0.02, 0.02]
    result = build_portfolio(
        predictions,
        target_annual_volatility=0.15,
        volatility_scaling=False,
        pseudo_cost_bps=0.0,
    )
    assert result.daily["gross_return_base"].iloc[0] == pytest.approx(0.5 * 0.02)
    assert result.daily["gross_return"].iloc[0] == pytest.approx(0.5 * 0.01)
