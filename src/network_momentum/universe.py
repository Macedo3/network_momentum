from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = ("ticker", "name", "region", "country", "exchange", "currency")
OPTIONAL_COLUMNS = (
    "sector",
    "short_eligible",
    "borrow_fee_annual_bps_estimate",
    "data_source",
    "note",
)


def load_universe(path: str | Path) -> pd.DataFrame:
    universe = pd.read_csv(path, dtype=str).fillna("")
    missing = [column for column in REQUIRED_COLUMNS if column not in universe.columns]
    if missing:
        raise ValueError(f"Colunas ausentes em {path}: {missing}")
    keep = list(REQUIRED_COLUMNS) + [c for c in OPTIONAL_COLUMNS if c in universe.columns]
    universe = universe.loc[:, keep].copy()
    universe["ticker"] = universe["ticker"].str.strip().str.upper()
    universe["region"] = universe["region"].str.strip()
    universe = universe[universe["ticker"].ne("")]
    if universe["ticker"].duplicated().any():
        duplicates = universe.loc[universe["ticker"].duplicated(), "ticker"].tolist()
        raise ValueError(f"Tickers duplicados no universo: {duplicates}")
    if "sector" not in universe.columns:
        universe["sector"] = ""
    if "short_eligible" in universe.columns:
        universe["short_eligible"] = (
            universe["short_eligible"].str.strip().str.lower().eq("true")
        )
    else:
        universe["short_eligible"] = True
    if "borrow_fee_annual_bps_estimate" in universe.columns:
        universe["borrow_fee_annual_bps_estimate"] = pd.to_numeric(
            universe["borrow_fee_annual_bps_estimate"], errors="coerce"
        )
    return universe.set_index("ticker", drop=False)


def universe_fingerprint(universe: pd.DataFrame) -> str:
    """Hash curto e estável do universo para o manifest e para o cache."""
    import hashlib

    payload = "|".join(sorted(universe["ticker"].astype(str)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
