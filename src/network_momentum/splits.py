from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ExpandingWindow:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def expanding_windows(
    dates: pd.DatetimeIndex,
    *,
    initial_train_years: int,
    test_years: int,
    embargo_days: int = 0,
) -> list[ExpandingWindow]:
    """Janelas expansivas sem sobreposição, com embargo opcional.

    ``embargo_days`` remove os últimos N pregões do treino imediatamente antes do
    teste. Com alvo de 1 dia o vazamento potencial na fronteira é de 1 dia; o
    embargo padrão de 1 dia elimina qualquer sobreposição entre o último alvo de
    treino e o primeiro retorno de teste.
    """
    dates = pd.DatetimeIndex(dates).sort_values().unique()
    if dates.empty:
        raise ValueError("dates está vazio.")
    if embargo_days < 0:
        raise ValueError("embargo_days não pode ser negativo.")
    first_date = pd.Timestamp(dates[0])
    last_date = pd.Timestamp(dates[-1])
    boundary = first_date + pd.DateOffset(years=initial_train_years)
    windows: list[ExpandingWindow] = []
    fold = 0

    while boundary <= last_date:
        test_candidates = dates[dates >= boundary]
        if test_candidates.empty:
            break
        test_start = pd.Timestamp(test_candidates[0])
        nominal_end = boundary + pd.DateOffset(years=test_years) - pd.Timedelta(days=1)
        test_end = min(pd.Timestamp(nominal_end), last_date)
        train_dates = dates[dates < test_start]
        if embargo_days:
            train_dates = train_dates[: max(len(train_dates) - embargo_days, 0)]
        test_dates = dates[(dates >= test_start) & (dates <= test_end)]
        if train_dates.empty or test_dates.empty:
            break
        fold += 1
        windows.append(
            ExpandingWindow(
                fold=fold,
                train_start=first_date,
                train_end=pd.Timestamp(train_dates[-1]),
                test_start=pd.Timestamp(test_dates[0]),
                test_end=pd.Timestamp(test_dates[-1]),
            )
        )
        boundary += pd.DateOffset(years=test_years)

    if not windows:
        raise ValueError(
            "Histórico insuficiente para formar a primeira janela de teste."
        )
    return windows
