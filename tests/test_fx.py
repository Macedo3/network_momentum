import numpy as np
import pandas as pd
import pytest

from network_momentum.fx import (
    convert_returns_to_base,
    currency_decomposition,
    forward_returns_in_base,
    fx_rates_to_base,
    load_fx_map,
)


def test_conversion_identity_and_decomposition() -> None:
    dates = pd.bdate_range("2022-01-03", periods=4)
    local = pd.DataFrame({"AAA": [0.01, -0.02, 0.005, 0.0]}, index=dates)
    rates = pd.DataFrame({"EUR": [1.10, 1.12, 1.09, 1.10]}, index=dates)
    currencies = pd.Series({"AAA": "EUR"})
    converted, fx_returns = convert_returns_to_base(local, currencies, rates, "USD")
    fx_expected = rates["EUR"].pct_change(fill_method=None)
    expected = (1 + local["AAA"]) * (1 + fx_expected) - 1
    pd.testing.assert_series_equal(converted["AAA"], expected, check_names=False)
    decomposition = currency_decomposition(local, converted)
    pd.testing.assert_series_equal(
        decomposition["AAA"], expected - local["AAA"], check_names=False
    )


def test_missing_currency_raises() -> None:
    dates = pd.bdate_range("2022-01-03", periods=3)
    local = pd.DataFrame({"AAA": [0.01, 0.0, 0.01]}, index=dates)
    with pytest.raises(ValueError, match="Sem taxa"):
        convert_returns_to_base(local, pd.Series({"AAA": "JPY"}), pd.DataFrame(index=dates), "USD")


def test_forward_returns_in_base_spans_local_holiday() -> None:
    calendar = pd.bdate_range("2022-01-03", periods=5)
    traded = calendar[[0, 2, 4]]  # o ativo não negocia nos dias 1 e 3
    prices = pd.DataFrame({"AAA": [100.0, 110.0, 99.0]}, index=traded).reindex(calendar)
    rates = pd.DataFrame(
        {"GBP": [1.30, 1.31, 1.32, 1.33, 1.34]}, index=calendar
    )
    forward = forward_returns_in_base(
        prices, pd.Series({"AAA": "GBP"}), rates, "USD", calendar
    )
    # retorno futuro no dia 0 cobre dia 0 -> dia 2, câmbio idem (1.32/1.30)
    expected_first = (110.0 * 1.32) / (100.0 * 1.30) - 1.0
    assert forward.loc[calendar[0], "AAA"] == pytest.approx(expected_first)
    assert np.isnan(forward.loc[calendar[1], "AAA"])
    expected_second = (99.0 * 1.34) / (110.0 * 1.32) - 1.0
    assert forward.loc[calendar[2], "AAA"] == pytest.approx(expected_second)


def test_fx_map_and_rates_inversion(safe_tmp_path) -> None:
    path = safe_tmp_path / "fx.csv"
    path.write_text(
        "currency,pair_ticker,invert,notes\nUSD,,false,base\nBRL,USDBRL=X,true,\nEUR,EURUSD=X,false,\n",
        encoding="utf-8",
    )
    fx_map = load_fx_map(path)
    dates = pd.bdate_range("2022-01-03", periods=2)
    pairs = pd.DataFrame({"USDBRL=X": [5.0, 5.5], "EURUSD=X": [1.1, 1.2]}, index=dates)
    rates = fx_rates_to_base(pairs, fx_map, "USD")
    assert rates.loc[dates[0], "BRL"] == pytest.approx(0.2)
    assert rates.loc[dates[0], "EUR"] == pytest.approx(1.1)
