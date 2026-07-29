from __future__ import annotations

"""Conversão cambial dos retornos para a moeda-base.

Convenções (documentadas no notebook):

- ``r_local``: retorno diário na moeda de cotação do ativo;
- ``r_fx``: retorno diário do par "moeda local -> moeda-base" (valorização da
  moeda local em relação à base);
- retorno convertido (sem hedge): ``(1 + r_local) * (1 + r_fx) - 1``;
- proxy hedgeada: usar ``r_local`` diretamente. Isso ignora o custo/prêmio do
  hedge (diferencial de juros embutido nos forwards) e é uma aproximação —
  nunca é apresentado como retorno hedgeado exato.

Ações da LSE são cotadas em pence (GBp); como a conversão é feita em nível de
retorno, o fator 100 cancela e não afeta nada aqui (afeta apenas cálculos de
notional, tratados em costs.py).
"""

import logging
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)


def load_fx_map(path: str | Path) -> pd.DataFrame:
    fx_map = pd.read_csv(path, dtype=str).fillna("")
    required = {"currency", "pair_ticker", "invert"}
    missing = required - set(fx_map.columns)
    if missing:
        raise ValueError(f"Colunas ausentes em {path}: {sorted(missing)}")
    fx_map["currency"] = fx_map["currency"].str.strip().str.upper()
    fx_map["pair_ticker"] = fx_map["pair_ticker"].str.strip().str.upper()
    fx_map["invert"] = fx_map["invert"].str.strip().str.lower().eq("true")
    return fx_map.set_index("currency")


def fx_rates_to_base(
    pair_closes: pd.DataFrame,
    fx_map: pd.DataFrame,
    base_currency: str,
) -> pd.DataFrame:
    """Converte fechamentos dos pares em taxas 'unidades da base por 1 unidade local'."""
    rates: dict[str, pd.Series] = {}
    for currency, row in fx_map.iterrows():
        if currency == base_currency:
            continue
        ticker = row["pair_ticker"]
        if not ticker or ticker not in pair_closes.columns:
            LOGGER.warning("Par cambial ausente para %s (%s).", currency, ticker)
            continue
        series = pair_closes[ticker].astype(float)
        rates[currency] = (1.0 / series) if row["invert"] else series
    return pd.DataFrame(rates)


def convert_returns_to_base(
    local_returns: pd.DataFrame,
    currencies: pd.Series,
    rates_to_base: pd.DataFrame,
    base_currency: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retorna (retornos na moeda-base, retornos cambiais usados).

    ``currencies`` mapeia ticker -> moeda. Ativos já na moeda-base passam
    inalterados. Os retornos cambiais são alinhados por ``ffill`` da taxa (o
    câmbio negocia quase continuamente; feriados da bolsa local não têm câmbio
    congelado). Nenhuma moeda é agregada silenciosamente: se faltar taxa para
    uma moeda presente no universo, a função levanta erro.
    """
    fx_returns = pd.DataFrame(index=local_returns.index)
    converted = pd.DataFrame(index=local_returns.index, columns=local_returns.columns, dtype=float)
    rates_aligned = rates_to_base.reindex(local_returns.index).ffill()
    missing_currencies = set()
    for ticker in local_returns.columns:
        currency = str(currencies.get(ticker, base_currency)).upper()
        if currency == base_currency:
            converted[ticker] = local_returns[ticker]
            continue
        if currency not in rates_aligned.columns:
            missing_currencies.add(currency)
            continue
        fx_ret = rates_aligned[currency].pct_change(fill_method=None)
        fx_returns[ticker] = fx_ret
        converted[ticker] = (1.0 + local_returns[ticker]) * (1.0 + fx_ret) - 1.0
    if missing_currencies:
        raise ValueError(
            "Sem taxa de câmbio para moedas do universo: "
            f"{sorted(missing_currencies)}. Preencha config/fx.csv."
        )
    return converted, fx_returns


def currency_decomposition(
    local_returns: pd.DataFrame,
    base_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Componente cambial efetivo por ativo: r_base − r_local (inclui o termo cruzado)."""
    return base_returns.sub(local_returns)


def forward_returns_in_base(
    prices_local: pd.DataFrame,
    currencies: pd.Series,
    rates_to_base: pd.DataFrame,
    base_currency: str,
    calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Retorno futuro de 1 pregão na moeda-base, alinhado como em features.py.

    Converte os PREÇOS para a moeda-base nas datas negociadas de cada ativo e
    só então calcula o retorno; assim o trecho cambial cobre exatamente a mesma
    janela do retorno do ativo, mesmo quando há feriados locais (um retorno de
    2 dias úteis usa a variação cambial dos mesmos 2 dias).
    """
    rates_aligned = rates_to_base.reindex(calendar).ffill()
    output: dict[str, pd.Series] = {}
    for ticker in prices_local.columns:
        series = prices_local[ticker].dropna().sort_index()
        currency = str(currencies.get(ticker, base_currency)).upper()
        if currency == base_currency:
            base_prices = series
        else:
            if currency not in rates_aligned.columns:
                raise ValueError(f"Sem taxa de câmbio para {currency} ({ticker}).")
            rate = rates_aligned[currency].reindex(series.index)
            base_prices = (series * rate).dropna()
        forward = base_prices.pct_change(fill_method=None).shift(-1)
        output[ticker] = forward.reindex(calendar)
    return pd.DataFrame(output, index=calendar)
