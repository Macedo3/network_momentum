from __future__ import annotations

"""Construção do portfólio (Eq. 9), turnover (Eq. 13) e custos (Eq. 14).

Fluxo em duas passadas:

1. pesos não alavancados ``w = sign(previsão) × σ_alvo / σ_ativo`` e retorno bruto
   diário ``média_i(w_i × r_i)`` — exatamente a Eq. (9);
2. alavancagem ex-ante de portfólio (EWMstd defasada do retorno líquido de
   pseudo-custo), como a "camada adicional de volatility scaling" do Painel B do
   artigo, porém implementável (o artigo não especifica a fórmula);
3. o turnover é calculado sobre a variação do **peso final** (incluindo a
   variação da alavancagem), separado em compras e vendas para permitir tributos
   assimétricos por lado (ex.: SDRT só na compra); ativo sem sinal na data tem
   peso zero (posição encerrada), e essa transição também paga custo.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .costs import TickerCosts, square_root_impact_bps

TRADING_DAYS = 252.0


@dataclass(frozen=True)
class ImpactInputs:
    adv_base: pd.DataFrame  # volume financeiro médio na moeda-base (datas × tickers)
    portfolio_notional: float
    impact_coefficient: float
    max_participation: float


@dataclass(frozen=True)
class PortfolioResult:
    daily: pd.DataFrame
    weights: pd.DataFrame  # pesos finais (datas × tickers), já alavancados
    turnover_by_ticker: pd.DataFrame
    cost_breakdown: pd.DataFrame  # custo diário por componente (fração do PL)
    participation_violations: float  # fração de (dia, ativo) acima do limite de participação


def _wide(predictions: pd.DataFrame, column: str) -> pd.DataFrame:
    frame = predictions.reset_index().pivot(index="date", columns="ticker", values=column)
    return frame.sort_index()


def build_portfolio(
    predictions: pd.DataFrame,
    *,
    target_annual_volatility: float,
    volatility_scaling: bool = True,
    volatility_span: int = 60,
    max_leverage: float = 5.0,
    pseudo_cost_bps: float = 0.0,
    ticker_costs: TickerCosts | None = None,
    impact_inputs: ImpactInputs | None = None,
) -> PortfolioResult:
    """``predictions``: frame longo (date, ticker) com colunas ``position``,
    ``forward_return``, ``annualized_volatility`` e opcionalmente
    ``forward_return_base`` (moeda-base). Retorna séries diárias brutas e líquidas,
    pesos finais e decomposição de custos."""
    required = {"position", "forward_return", "annualized_volatility"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Colunas ausentes em predictions: {sorted(missing)}")

    working = predictions.sort_index().copy()
    working["weight_unscaled"] = (
        working["position"]
        * target_annual_volatility
        / working["annualized_volatility"]
    )

    weights_unscaled = _wide(working, "weight_unscaled").fillna(0.0)
    forward_local = _wide(working, "forward_return").fillna(0.0)
    active_count = working.groupby(level="date").size().reindex(weights_unscaled.index)

    gross_unscaled = weights_unscaled.mul(forward_local).sum(axis=1) / active_count
    delta_unscaled = weights_unscaled.diff()
    delta_unscaled.iloc[0] = weights_unscaled.iloc[0]
    turnover_unscaled = delta_unscaled.abs().sum(axis=1) / active_count
    net_unscaled = gross_unscaled - turnover_unscaled * pseudo_cost_bps / 1e4

    if volatility_scaling:
        realized = (
            net_unscaled.ewm(
                span=volatility_span,
                adjust=True,
                min_periods=max(10, volatility_span // 3),
            )
            .std(bias=True)
            .shift(1)
            * np.sqrt(TRADING_DAYS)
        )
        leverage = (target_annual_volatility / realized).clip(
            lower=0.0, upper=max_leverage
        ).fillna(1.0)
    else:
        leverage = pd.Series(1.0, index=weights_unscaled.index)

    weights_final = weights_unscaled.mul(leverage, axis=0)
    delta_final = weights_final.diff()
    delta_final.iloc[0] = weights_final.iloc[0]
    buys = delta_final.clip(lower=0.0)
    sells = (-delta_final).clip(lower=0.0)
    turnover_final = delta_final.abs().sum(axis=1) / active_count
    gross_scaled = gross_unscaled * leverage

    cost_columns: dict[str, pd.Series] = {}
    cost_columns["cost_pseudo"] = turnover_final * pseudo_cost_bps / 1e4

    participation_violations = 0.0
    if ticker_costs is not None:
        tickers = weights_final.columns
        buy_bps = ticker_costs.buy_bps.reindex(tickers)
        sell_bps = ticker_costs.sell_bps.reindex(tickers)
        borrow_bps = ticker_costs.borrow_daily_bps.reindex(tickers)
        for name, series in (("buy", buy_bps), ("sell", sell_bps), ("borrow", borrow_bps)):
            if series.isna().any():
                absent = series[series.isna()].index.tolist()
                raise ValueError(f"Custos ({name}) ausentes para tickers: {absent}")

        components = ticker_costs.components.reindex(tickers)
        both_sides = buys.add(sells)
        per_side_pairs = {
            "cost_commission": (components["commission_bps"], components["commission_bps"]),
            "cost_exchange_fee": (
                components["exchange_fee_bps"],
                components["exchange_fee_bps"],
            ),
            "cost_spread": (components["half_spread_bps"], components["half_spread_bps"]),
            "cost_fx": (components["fx_conversion_bps"], components["fx_conversion_bps"]),
            "cost_regulatory": (
                components["regulatory_buy_bps"],
                components["regulatory_sell_bps"],
            ),
            "cost_tax": (components["tax_buy_bps"], components["tax_sell_bps"]),
        }
        for column, (bps_buy, bps_sell) in per_side_pairs.items():
            cost = (
                buys.mul(bps_buy, axis=1).sum(axis=1)
                + sells.mul(bps_sell, axis=1).sum(axis=1)
            ) / 1e4 / active_count
            cost_columns[column] = cost
        shorts = (-weights_final).clip(lower=0.0)
        cost_columns["cost_borrow"] = (
            shorts.mul(borrow_bps, axis=1).sum(axis=1) / 1e4 / active_count
        )

        if impact_inputs is not None:
            adv = impact_inputs.adv_base.reindex(weights_final.index).ffill()
            daily_vol_wide = (
                _wide(working, "annualized_volatility") / np.sqrt(TRADING_DAYS)
            ).reindex(weights_final.index)
            impact_costs = pd.Series(0.0, index=weights_final.index)
            violations = 0
            trades = 0
            per_asset_notional = impact_inputs.portfolio_notional / active_count
            for ticker in weights_final.columns:
                if ticker not in adv.columns:
                    continue
                trade_notional = delta_final[ticker].abs() * per_asset_notional
                impact_bps = square_root_impact_bps(
                    trade_notional,
                    adv[ticker],
                    daily_vol_wide[ticker],
                    impact_coefficient=impact_inputs.impact_coefficient,
                )
                impact_costs = impact_costs.add(
                    delta_final[ticker].abs() * impact_bps / 1e4,
                    fill_value=0.0,
                )
                with np.errstate(invalid="ignore", divide="ignore"):
                    participation = trade_notional / adv[ticker]
                traded = trade_notional > 0
                trades += int(traded.sum())
                violations += int(
                    (participation[traded] > impact_inputs.max_participation).sum()
                )
            cost_columns["cost_impact"] = impact_costs / active_count
            participation_violations = violations / trades if trades else 0.0

    daily = pd.DataFrame(
        {
            "gross_return": gross_unscaled,
            "turnover": turnover_unscaled,
            "active_assets": active_count.astype(float),
            "portfolio_leverage": leverage,
            "net_return_unscaled": net_unscaled,
            "scaled_gross_return": gross_scaled,
            "turnover_final": turnover_final,
        }
    )
    for column, series in cost_columns.items():
        daily[column] = series
    real_components = [c for c in cost_columns if c != "cost_pseudo"]
    if real_components:
        daily["cost_real_total"] = daily[real_components].sum(axis=1)
        daily["strategy_return"] = daily["scaled_gross_return"] - daily["cost_real_total"]
        daily["net_return_pseudo"] = (
            daily["scaled_gross_return"] - daily["cost_pseudo"]
        )
    else:
        daily["cost_real_total"] = np.nan
        daily["strategy_return"] = daily["scaled_gross_return"] - daily["cost_pseudo"]
        daily["net_return_pseudo"] = daily["strategy_return"]

    if "forward_return_base" in working.columns:
        forward_base = _wide(working, "forward_return_base")
        covered = forward_base.notna() | weights_final.eq(0.0)
        forward_base = forward_base.fillna(0.0)
        gross_base = weights_final.mul(forward_base).sum(axis=1) / active_count
        daily["gross_return_base"] = gross_base
        daily["strategy_return_base"] = gross_base - (
            daily["cost_real_total"].fillna(daily["cost_pseudo"])
        )
        daily["base_coverage"] = covered.mean(axis=1)

    exposures_gross = weights_final.abs().sum(axis=1) / active_count
    exposures_net = weights_final.sum(axis=1) / active_count
    daily["gross_exposure"] = exposures_gross
    daily["net_exposure"] = exposures_net
    daily["n_long"] = weights_final.gt(0).sum(axis=1).astype(float)
    daily["n_short"] = weights_final.lt(0).sum(axis=1).astype(float)

    breakdown = pd.DataFrame(cost_columns)
    return PortfolioResult(
        daily=daily,
        weights=weights_final,
        turnover_by_ticker=delta_final.abs(),
        cost_breakdown=breakdown,
        participation_violations=participation_violations,
    )
