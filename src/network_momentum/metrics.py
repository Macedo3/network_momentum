from __future__ import annotations

"""Métricas de desempenho.

Convenções (documentadas também no notebook e no relatório):
- retornos **aritméticos diários**; anualização com 252 pregões;
- taxa livre de risco configurável (padrão 0, como no artigo — declarado);
- ``annual_return`` é a média aritmética anualizada (convenção do artigo);
  ``cagr`` é o retorno geométrico composto;
- valores ausentes são descartados (dias sem estratégia não entram na média);
- VaR e Expected Shortfall históricos, na frequência diária, em fração do PL.
"""

import numpy as np
import pandas as pd


TRADING_DAYS = 252.0


def maximum_drawdown(returns: pd.Series) -> tuple[float, float]:
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    drawdown = wealth.div(wealth.cummax()).sub(1.0)
    maximum = float(-drawdown.min()) if not drawdown.empty else np.nan
    underwater = drawdown < 0
    if drawdown.empty:
        duration = np.nan
    else:
        run_id = (~underwater).cumsum()
        longest_run = int(underwater.groupby(run_id).sum().max()) if underwater.any() else 0
        duration = float(longest_run / len(drawdown))
    return maximum, duration


def performance_metrics(returns: pd.Series, *, risk_free_rate: float = 0.0) -> pd.Series:
    """Métricas anuais da série de retornos diários. ``risk_free_rate`` é anual."""
    values = returns.dropna().astype(float)
    if values.empty:
        return pd.Series(dtype=float)
    excess = values - risk_free_rate / TRADING_DAYS
    annual_return = float(values.mean() * TRADING_DAYS)
    annual_volatility = float(values.std(ddof=0) * np.sqrt(TRADING_DAYS))
    n_years = len(values) / TRADING_DAYS
    wealth = float((1.0 + values).prod())
    cagr = wealth ** (1.0 / n_years) - 1.0 if wealth > 0 and n_years > 0 else np.nan
    downside = np.minimum(excess.to_numpy(), 0.0)
    downside_deviation = float(np.sqrt(np.mean(downside**2)) * np.sqrt(TRADING_DAYS))
    max_drawdown, drawdown_duration = maximum_drawdown(values)
    gains = values[values > 0]
    losses = values[values < 0]
    average_gain = float(gains.mean()) if not gains.empty else np.nan
    average_loss = float(losses.mean()) if not losses.empty else np.nan
    profit_factor = (
        float(gains.sum() / -losses.sum()) if not losses.empty and losses.sum() != 0 else np.nan
    )
    average_profit_loss = (
        float(average_gain / -average_loss)
        if not gains.empty and not losses.empty
        else np.nan
    )
    centered = values - values.mean()
    sigma = float(values.std(ddof=0))
    skewness = float((centered**3).mean() / sigma**3) if sigma > 0 else np.nan
    kurtosis = float((centered**4).mean() / sigma**4 - 3.0) if sigma > 0 else np.nan
    var_95 = float(-np.percentile(values, 5))
    var_99 = float(-np.percentile(values, 1))
    tail_95 = values[values <= -var_95]
    expected_shortfall_95 = float(-tail_95.mean()) if not tail_95.empty else np.nan
    positive_part = float(np.maximum(excess, 0.0).sum())
    negative_part = float(-np.minimum(excess, 0.0).sum())
    omega = positive_part / negative_part if negative_part > 0 else np.nan
    excess_annual = float(excess.mean() * TRADING_DAYS)
    sharpe = excess_annual / annual_volatility if annual_volatility else np.nan

    return pd.Series(
        {
            "annual_return": annual_return,
            "cagr": float(cagr),
            "annual_volatility": annual_volatility,
            "sharpe": sharpe,
            "downside_deviation": downside_deviation,
            "sortino": excess_annual / downside_deviation if downside_deviation else np.nan,
            "max_drawdown": max_drawdown,
            "drawdown_duration": drawdown_duration,
            "calmar": annual_return / max_drawdown if max_drawdown else np.nan,
            "omega": float(omega),
            "hit_rate": float((values > 0).mean()),
            "profit_factor": profit_factor,
            "average_gain": average_gain,
            "average_loss": average_loss,
            "average_profit_over_loss": average_profit_loss,
            "skewness": skewness,
            "excess_kurtosis": kurtosis,
            "var_95": var_95,
            "var_99": var_99,
            "expected_shortfall_95": expected_shortfall_95,
            "risk_free_rate": float(risk_free_rate),
            "observations": float(len(values)),
        }
    )


def annual_metrics(returns: pd.Series) -> pd.DataFrame:
    rows: list[pd.Series] = []
    for year, year_returns in returns.groupby(returns.index.year):
        row = performance_metrics(year_returns)
        row.name = int(year)
        rows.append(row)
    return pd.DataFrame(rows)


def portfolio_state_metrics(daily: pd.DataFrame) -> pd.Series:
    """Métricas operacionais a partir do frame diário do portfólio."""
    result: dict[str, float] = {}
    if "turnover_final" in daily:
        result["annual_turnover"] = float(daily["turnover_final"].mean() * TRADING_DAYS)
    if "cost_real_total" in daily and daily["cost_real_total"].notna().any():
        result["annual_cost_real"] = float(daily["cost_real_total"].mean() * TRADING_DAYS)
    if "cost_pseudo" in daily:
        result["annual_cost_pseudo"] = float(daily["cost_pseudo"].mean() * TRADING_DAYS)
    if "gross_exposure" in daily:
        result["average_gross_exposure"] = float(daily["gross_exposure"].mean())
        result["average_net_exposure"] = float(daily["net_exposure"].mean())
    if "portfolio_leverage" in daily:
        result["average_leverage"] = float(daily["portfolio_leverage"].mean())
        result["max_leverage"] = float(daily["portfolio_leverage"].max())
    if "active_assets" in daily:
        result["average_active_assets"] = float(daily["active_assets"].mean())
    if "n_long" in daily:
        result["average_n_long"] = float(daily["n_long"].mean())
        result["average_n_short"] = float(daily["n_short"].mean())
    return pd.Series(result)


def relative_statistics(
    strategy: pd.Series,
    benchmark: pd.Series,
    *,
    risk_free_rate: float = 0.0,
) -> pd.Series:
    """Alpha/beta (OLS diária), correlação, tracking error, information ratio e
    capturas de alta/baixa contra um benchmark na mesma moeda e período."""
    aligned = pd.concat(
        [strategy.rename("strategy"), benchmark.rename("benchmark")], axis=1
    ).dropna()
    if len(aligned) < 30:
        return pd.Series(dtype=float)
    rf_daily = risk_free_rate / TRADING_DAYS
    strategy_excess = aligned["strategy"] - rf_daily
    benchmark_excess = aligned["benchmark"] - rf_daily
    variance = float(benchmark_excess.var(ddof=0))
    covariance = float(
        ((strategy_excess - strategy_excess.mean()) * (benchmark_excess - benchmark_excess.mean())).mean()
    )
    beta = covariance / variance if variance > 0 else np.nan
    alpha_daily = float(strategy_excess.mean() - beta * benchmark_excess.mean())
    correlation = float(aligned["strategy"].corr(aligned["benchmark"]))
    active = aligned["strategy"] - aligned["benchmark"]
    tracking_error = float(active.std(ddof=0) * np.sqrt(TRADING_DAYS))
    information_ratio = (
        float(active.mean() * TRADING_DAYS / tracking_error) if tracking_error > 0 else np.nan
    )
    up = aligned[aligned["benchmark"] > 0]
    down = aligned[aligned["benchmark"] < 0]
    up_capture = (
        float(up["strategy"].mean() / up["benchmark"].mean())
        if not up.empty and up["benchmark"].mean() != 0
        else np.nan
    )
    down_capture = (
        float(down["strategy"].mean() / down["benchmark"].mean())
        if not down.empty and down["benchmark"].mean() != 0
        else np.nan
    )
    return pd.Series(
        {
            "alpha_annual": alpha_daily * TRADING_DAYS,
            "beta": beta,
            "correlation": correlation,
            "tracking_error": tracking_error,
            "information_ratio": information_ratio,
            "up_capture": up_capture,
            "down_capture": down_capture,
            "overlap_days": float(len(aligned)),
        }
    )


def sign_agreement(positions_a: pd.Series, positions_b: pd.Series) -> float:
    """Fração de (data, ativo) em que duas estratégias têm o mesmo sinal de posição."""
    aligned = pd.concat([positions_a.rename("a"), positions_b.rename("b")], axis=1).dropna()
    if aligned.empty:
        return float("nan")
    return float((np.sign(aligned["a"]) == np.sign(aligned["b"])).mean())
