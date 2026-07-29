from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile
from typing import Iterable

import pandas as pd

from .config import DataConfig


LOGGER = logging.getLogger(__name__)


def _ensure_ascii_ca_bundle() -> None:
    """Contorna a limitação do libcurl com caminhos Unicode no Windows."""
    try:
        import certifi
    except ImportError:
        return
    source = Path(certifi.where())
    try:
        str(source).encode("ascii")
        return
    except UnicodeEncodeError:
        pass
    destination_dir = Path(tempfile.gettempdir()) / "network_momentum_certs"
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / "cacert.pem"
    if not destination.exists() or destination.stat().st_mtime < source.stat().st_mtime:
        shutil.copyfile(source, destination)
    os.environ["SSL_CERT_FILE"] = str(destination)
    os.environ["CURL_CA_BUNDLE"] = str(destination)


def _cache_key(config: DataConfig, tickers: Iterable[str], fields: Iterable[str]) -> str:
    end = config.end or _dt.date.today().isoformat()
    payload = "|".join(
        [
            ",".join(sorted(tickers)),
            ",".join(sorted(fields)),
            config.start,
            end,
            str(config.auto_adjust),
            str(config.repair),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [str(column).upper() for column in frame.columns]
    frame.index = pd.to_datetime(frame.index, utc=True).tz_convert(None).normalize()
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    return frame.apply(pd.to_numeric, errors="coerce")


def _extract_field(raw: pd.DataFrame, field: str, tickers: list[str]) -> pd.DataFrame:
    if raw.empty:
        raise RuntimeError("O Yahoo Finance não retornou observações.")
    if isinstance(raw.columns, pd.MultiIndex):
        level_zero = raw.columns.get_level_values(0)
        level_one = raw.columns.get_level_values(1)
        if field in level_zero:
            frame = raw.xs(field, axis=1, level=0)
        elif field in level_one:
            frame = raw.xs(field, axis=1, level=1)
        else:
            raise RuntimeError(f"Coluna {field} ausente no retorno do yfinance.")
    else:
        if field not in raw.columns:
            raise RuntimeError(f"Coluna {field} ausente no retorno do yfinance.")
        if len(tickers) != 1:
            raise RuntimeError("Formato de retorno inesperado para múltiplos tickers.")
        frame = raw[[field]].rename(columns={field: tickers[0]})
    return _normalize_frame(frame)


def download_price_history(
    tickers: Iterable[str],
    config: DataConfig,
    *,
    refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Baixa OHLCV ajustado e devolve um dicionário campo -> DataFrame (datas x tickers).

    O cache é carimbado com a data efetiva de término, de modo que uma execução em
    outro dia com `end` vazio gera novo download em vez de reutilizar dados velhos.
    """
    requested = list(dict.fromkeys(str(ticker).upper() for ticker in tickers))
    fields = list(config.download_fields)
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(config, requested, fields)
    cache_paths = {field: config.cache_dir / f"{field.lower()}_{key}.csv.gz" for field in fields}
    stamp_path = config.cache_dir / f"download_stamp_{key}.json"

    if all(path.exists() for path in cache_paths.values()) and not refresh:
        LOGGER.info("Carregando OHLCV do cache (chave %s).", key)
        frames = {
            field: pd.read_csv(path, index_col=0, parse_dates=True)
            for field, path in cache_paths.items()
        }
        for field in frames:
            frames[field].columns = [str(c).upper() for c in frames[field].columns]
    else:
        _ensure_ascii_ca_bundle()
        try:
            import yfinance as yf
        except ImportError as exc:
            raise RuntimeError(
                "yfinance não está instalado. Execute `python -m pip install -e .`."
            ) from exc
        LOGGER.info("Baixando %d tickers com yfinance.", len(requested))
        yf.set_tz_cache_location(str(config.cache_dir / "yfinance"))
        raw = yf.download(
            tickers=requested,
            start=config.start,
            end=config.end,
            interval="1d",
            group_by="column",
            auto_adjust=config.auto_adjust,
            repair=config.repair,
            actions=False,
            threads=config.threads,
            progress=False,
            keepna=True,
            multi_level_index=True,
        )
        frames = {}
        for field in fields:
            frame = _extract_field(raw, field, requested)
            frames[field] = frame
            frame.to_csv(cache_paths[field], compression="gzip", date_format="%Y-%m-%d")
        stamp_path.write_text(
            json.dumps(
                {
                    "downloaded_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                    "tickers": requested,
                    "fields": fields,
                    "start": config.start,
                    "end": config.end or "latest",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        LOGGER.info("Cache OHLCV gravado (chave %s).", key)

    close = frames.get("Close")
    if close is None:
        raise RuntimeError("Campo Close é obrigatório em download_fields.")
    close = close.where(close > 0)
    available = [t for t in requested if t in close.columns and close[t].notna().any()]
    missing = sorted(set(requested) - set(available))
    if missing:
        LOGGER.warning("Tickers sem preços e removidos: %s", ", ".join(missing))
    for field in frames:
        frames[field] = frames[field].reindex(columns=available)
    frames["Close"] = close.reindex(columns=available).dropna(how="all")
    aligned_index = frames["Close"].index
    for field in frames:
        frames[field] = frames[field].reindex(aligned_index)
    if frames["Close"].shape[1] < config.min_tickers:
        raise RuntimeError(
            f"Apenas {frames['Close'].shape[1]} tickers válidos; mínimo: {config.min_tickers}."
        )
    return frames


def download_adjusted_close(
    tickers: Iterable[str],
    config: DataConfig,
    *,
    refresh: bool = False,
) -> pd.DataFrame:
    """Compatibilidade retroativa: devolve apenas o Close ajustado."""
    return download_price_history(tickers, config, refresh=refresh)["Close"]


def download_auxiliary_close(
    tickers: Iterable[str],
    config: DataConfig,
    *,
    refresh: bool = False,
) -> pd.DataFrame:
    """Fechamento ajustado de séries auxiliares (pares FX, benchmarks)."""
    requested = [str(t).upper() for t in dict.fromkeys(tickers) if str(t).strip()]
    if not requested:
        return pd.DataFrame()
    from dataclasses import replace

    aux_config = replace(
        config,
        download_fields=("Close",),
        min_tickers=1,
    )
    return download_price_history(requested, aux_config, refresh=refresh)["Close"]


def run_data_quality_checks(
    frames: dict[str, pd.DataFrame],
    config: DataConfig,
) -> pd.DataFrame:
    """Verificações por ticker: preços não positivos, duplicatas, gaps, stale prices,
    volume zero, retornos extremos e histórico insuficiente. Retorna um relatório
    (uma linha por ticker) sem alterar os dados."""
    close = frames["Close"]
    volume = frames.get("Volume")
    records: list[dict[str, object]] = []
    for ticker in close.columns:
        series = close[ticker].dropna()
        returns = series.pct_change(fill_method=None).dropna()
        non_positive = int((series <= 0).sum())
        duplicated_dates = int(series.index.duplicated().sum())
        if len(series) > 1:
            gaps = series.index.to_series().diff().dt.days.dropna()
            max_gap_days = int(gaps.max()) if not gaps.empty else 0
        else:
            max_gap_days = 0
        stale = series.diff().eq(0.0)
        if stale.any():
            run_groups = (~stale).cumsum()
            max_stale_run = int(stale.groupby(run_groups).sum().max())
        else:
            max_stale_run = 0
        extreme = int((returns.abs() > config.max_abs_daily_return).sum())
        zero_volume = (
            int((volume[ticker].dropna() == 0).sum()) if volume is not None and ticker in volume else -1
        )
        records.append(
            {
                "ticker": ticker,
                "first_date": series.index.min(),
                "last_date": series.index.max(),
                "observations": int(len(series)),
                "non_positive_prices": non_positive,
                "duplicated_dates": duplicated_dates,
                "max_calendar_gap_days": max_gap_days,
                "max_stale_run_days": max_stale_run,
                "stale_flag": max_stale_run > config.max_stale_run_days,
                "extreme_return_days": extreme,
                "zero_volume_days": zero_volume,
                "median_daily_volume": (
                    float(volume[ticker].dropna().median())
                    if volume is not None and ticker in volume
                    else float("nan")
                ),
            }
        )
    report = pd.DataFrame.from_records(records).set_index("ticker")
    issues = report[
        (report["non_positive_prices"] > 0)
        | (report["duplicated_dates"] > 0)
        | report["stale_flag"]
        | (report["extreme_return_days"] > 0)
    ]
    if not issues.empty:
        LOGGER.warning(
            "Qualidade de dados: %d tickers com apontamentos (%s).",
            len(issues),
            ", ".join(issues.index[:10]),
        )
    return report
