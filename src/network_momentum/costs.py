from __future__ import annotations

"""Modelo de custos de negociação por bolsa/ticker/lado.

Todos os valores vêm de ``config/costs.csv`` — nada é embutido no código. O CSV
distingue impostos estatutários verificados em fonte oficial (coluna
``tax_official``) de estimativas metodológicas (comissão, spread, borrow). Os
cenários (conservative/base/optimistic) trocam apenas a coluna de meio-spread.

O custo é aplicado sobre a variação efetiva dos pesos finais (compras e vendas
separadas), na mesma convenção da Eq. (14) do artigo: contribuição de custo =
soma_i custo_i / N_t.
"""

from dataclasses import dataclass
import logging
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

TRADING_DAYS = 252.0

COMPONENT_COLUMNS = (
    "commission_bps",
    "exchange_fee_bps",
    "regulatory_buy_bps",
    "regulatory_sell_bps",
    "tax_buy_bps",
    "tax_sell_bps",
    "half_spread_bps_conservative",
    "half_spread_bps_base",
    "half_spread_bps_optimistic",
    "borrow_fee_annual_bps",
    "fx_conversion_bps",
)


def load_cost_table(path: str | Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    required = {"scope", "key", *COMPONENT_COLUMNS}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Colunas ausentes em {path}: {sorted(missing)}")
    table["scope"] = table["scope"].str.strip().str.lower()
    table["key"] = table["key"].astype(str).str.strip()
    for column in COMPONENT_COLUMNS:
        table[column] = pd.to_numeric(table[column], errors="coerce").fillna(0.0)
    return table


@dataclass(frozen=True)
class TickerCosts:
    """Custos lineares por ticker, em bps por lado, mais borrow diário."""

    buy_bps: pd.Series
    sell_bps: pd.Series
    borrow_daily_bps: pd.Series
    components: pd.DataFrame
    scenario: str

    def as_frame(self) -> pd.DataFrame:
        frame = self.components.copy()
        frame["total_buy_bps"] = self.buy_bps
        frame["total_sell_bps"] = self.sell_bps
        frame["borrow_daily_bps"] = self.borrow_daily_bps
        return frame


def _row_for_ticker(table: pd.DataFrame, ticker: str, exchange: str) -> pd.Series:
    ticker_rows = table[(table["scope"] == "ticker") & (table["key"].str.upper() == ticker)]
    if not ticker_rows.empty:
        return ticker_rows.iloc[0]
    exchange_rows = table[(table["scope"] == "exchange") & (table["key"] == exchange)]
    if not exchange_rows.empty:
        return exchange_rows.iloc[0]
    default_rows = table[table["scope"] == "default"]
    if not default_rows.empty:
        LOGGER.warning("Usando custos DEFAULT para %s (%s).", ticker, exchange)
        return default_rows.iloc[0]
    raise ValueError(f"Sem linha de custo para {ticker} ({exchange}) e sem DEFAULT.")


def build_ticker_costs(
    table: pd.DataFrame,
    universe: pd.DataFrame,
    *,
    scenario: str = "base",
    base_currency: str = "USD",
) -> TickerCosts:
    spread_column = f"half_spread_bps_{scenario}"
    if spread_column not in table.columns:
        raise ValueError(f"Cenário desconhecido: {scenario}")
    records: dict[str, dict[str, float]] = {}
    for ticker, row in universe.iterrows():
        cost_row = _row_for_ticker(table, str(ticker), str(row.get("exchange", "")))
        is_base_currency = str(row.get("currency", base_currency)).upper() == base_currency
        fx_bps = 0.0 if is_base_currency else float(cost_row["fx_conversion_bps"])
        borrow_annual = float(cost_row["borrow_fee_annual_bps"])
        borrow_override = row.get("borrow_fee_annual_bps_estimate")
        if borrow_override is not None and np.isfinite(float(borrow_override or np.nan)):
            borrow_annual = float(borrow_override)
        records[str(ticker)] = {
            "commission_bps": float(cost_row["commission_bps"]),
            "exchange_fee_bps": float(cost_row["exchange_fee_bps"]),
            "regulatory_buy_bps": float(cost_row["regulatory_buy_bps"]),
            "regulatory_sell_bps": float(cost_row["regulatory_sell_bps"]),
            "tax_buy_bps": float(cost_row["tax_buy_bps"]),
            "tax_sell_bps": float(cost_row["tax_sell_bps"]),
            "half_spread_bps": float(cost_row[spread_column]),
            "fx_conversion_bps": fx_bps,
            "borrow_fee_annual_bps": borrow_annual,
        }
    components = pd.DataFrame.from_dict(records, orient="index")
    common = (
        components["commission_bps"]
        + components["exchange_fee_bps"]
        + components["half_spread_bps"]
        + components["fx_conversion_bps"]
    )
    buy_bps = common + components["regulatory_buy_bps"] + components["tax_buy_bps"]
    sell_bps = common + components["regulatory_sell_bps"] + components["tax_sell_bps"]
    borrow_daily = components["borrow_fee_annual_bps"] / TRADING_DAYS
    return TickerCosts(
        buy_bps=buy_bps,
        sell_bps=sell_bps,
        borrow_daily_bps=borrow_daily,
        components=components,
        scenario=scenario,
    )


def average_daily_value_base(
    close_local: pd.DataFrame,
    volume: pd.DataFrame,
    currencies: pd.Series,
    rates_to_base: pd.DataFrame,
    *,
    base_currency: str = "USD",
    pence_quoted: set[str] | None = None,
    window: int = 63,
) -> pd.DataFrame:
    """Volume financeiro médio (janela móvel) na moeda-base, por ticker.

    Ações da LSE cotam em pence; ``pence_quoted`` divide o preço por 100 antes do
    cálculo do notional (não afeta retornos em nenhum outro ponto do projeto).
    """
    pence_quoted = pence_quoted or set()
    rates = rates_to_base.reindex(close_local.index).ffill()
    values: dict[str, pd.Series] = {}
    for ticker in close_local.columns:
        if ticker not in volume.columns:
            continue
        price = close_local[ticker].astype(float)
        if ticker in pence_quoted:
            price = price / 100.0
        currency = str(currencies.get(ticker, base_currency)).upper()
        if currency == base_currency:
            fx = 1.0
        elif currency in rates.columns:
            fx = rates[currency]
        else:
            continue
        notional = price * volume[ticker].astype(float) * fx
        values[ticker] = notional.rolling(window, min_periods=max(5, window // 3)).mean()
    return pd.DataFrame(values)


def square_root_impact_bps(
    trade_notional: pd.Series,
    adv_base: pd.Series,
    daily_volatility: pd.Series,
    *,
    impact_coefficient: float,
) -> pd.Series:
    """Impacto pela lei de raiz quadrada: k · σ_diária · sqrt(Q/ADV), em bps.

    Metodologia declarada como estimativa (dados diários não permitem calibração
    fina); ver docs/REFERENCES.md.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        participation = (trade_notional / adv_base).clip(lower=0.0)
    impact = impact_coefficient * daily_volatility * np.sqrt(participation) * 1e4
    return impact.fillna(0.0)


def breakeven_cost_bps(gross_returns: pd.Series, turnover: pd.Series) -> float:
    """Custo linear (bps sobre turnover) que zera o retorno médio: E[r] / E[ζ] × 1e4."""
    mean_turnover = float(turnover.mean())
    if mean_turnover <= 0:
        return float("nan")
    return float(gross_returns.mean() / mean_turnover * 1e4)


def sharpe_versus_cost(
    gross_returns: pd.Series,
    turnover: pd.Series,
    sweep_bps: tuple[float, ...],
) -> pd.DataFrame:
    """Curva Sharpe líquido × pseudo-custo (Eq. 14 do artigo, c em bps)."""
    rows = []
    aligned = pd.concat(
        [gross_returns.rename("gross"), turnover.rename("turnover")], axis=1
    ).dropna()
    for cost in sweep_bps:
        net = aligned["gross"] - aligned["turnover"] * cost / 1e4
        volatility = float(net.std(ddof=0))
        sharpe = float(net.mean() / volatility * np.sqrt(TRADING_DAYS)) if volatility else np.nan
        rows.append(
            {
                "cost_bps": cost,
                "net_annual_return": float(net.mean() * TRADING_DAYS),
                "net_sharpe": sharpe,
            }
        )
    return pd.DataFrame(rows).set_index("cost_bps")


def capacity_estimate(
    turnover_by_ticker: pd.DataFrame,
    adv_base: pd.DataFrame,
    active_count: pd.Series,
    *,
    max_participation: float,
) -> pd.DataFrame:
    """Capacidade estimada: maior PL que mantém a participação mediana de cada
    ativo abaixo do limite. Ordem por ativo-dia = |Δw| × PL / N_t; logo
    PL_max(i) = participação_max × ADV_mediana(i) × N̄ / mediana(|Δw_i|).
    A capacidade do portfólio é o mínimo entre os ativos (restrição mais
    apertada). Estimativa grosseira e declarada como tal: usa medianas e ignora
    a distribuição conjunta dos dias de rebalanceamento."""
    median_active = float(active_count.median())
    rows = []
    for ticker in turnover_by_ticker.columns:
        trades = turnover_by_ticker[ticker]
        trades = trades[trades > 0]
        if trades.empty or ticker not in adv_base.columns:
            continue
        median_trade_weight = float(trades.median())
        median_adv = float(adv_base[ticker].dropna().median())
        if median_trade_weight <= 0 or not np.isfinite(median_adv):
            continue
        rows.append(
            {
                "ticker": ticker,
                "median_trade_weight": median_trade_weight,
                "median_adv_base": median_adv,
                "max_notional_base": max_participation
                * median_adv
                * median_active
                / median_trade_weight,
            }
        )
    frame = pd.DataFrame(rows).set_index("ticker").sort_values("max_notional_base")
    return frame
