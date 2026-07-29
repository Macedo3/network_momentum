from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

_matplotlib_cache = Path(tempfile.gettempdir()) / "network_momentum_matplotlib"
_matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_matplotlib_cache))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .backtest import BacktestResult
from .graph import graph_edges
from .metrics import annual_metrics, performance_metrics


def _plot_performance(daily: pd.DataFrame, output_path: Path) -> None:
    strategy = daily["strategy_return"].fillna(0.0)
    wealth = (1.0 + strategy).cumprod()
    drawdown = wealth.div(wealth.cummax()).sub(1.0)

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(11, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    axes[0].plot(wealth.index, wealth, color="#E85D04", linewidth=1.6)
    axes[0].set_title("Network Momentum - riqueza acumulada OOS")
    axes[0].set_ylabel("Capital (início = 1)")
    axes[0].grid(alpha=0.25)
    axes[1].fill_between(
        drawdown.index,
        drawdown.to_numpy(),
        0,
        color="#9D0208",
        alpha=0.65,
    )
    axes[1].set_ylabel("Drawdown")
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _git_commit() -> str | None:
    try:
        output = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return output.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _library_versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for name in ("numpy", "pandas", "scipy", "matplotlib", "yfinance", "cvxpy"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "?")
        except ImportError:
            versions[name] = "ausente"
    return versions


def build_manifest(
    *,
    config,
    result: BacktestResult,
    universe_hash: str = "",
    n_assets: int = 0,
    seed: int | None = None,
    data_stamp: dict | None = None,
    extras: dict | None = None,
) -> dict:
    """Manifesto de execução: tudo que é necessário para reproduzir a rodada."""
    manifest = {
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "platform": platform.platform(),
        "in_colab": "google.colab" in sys.modules,
        "library_versions": _library_versions(),
        "config_path": str(getattr(config, "source_path", "")),
        "config": dataclasses.asdict(config) if dataclasses.is_dataclass(config) else str(config),
        "base_currency": getattr(getattr(config, "data", None), "base_currency", "USD"),
        "seed": seed,
        "universe_hash": universe_hash,
        "n_assets": n_assets,
        "oos_start": str(result.daily_returns.index.min().date()),
        "oos_end": str(result.daily_returns.index.max().date()),
        "folds": int(result.folds.shape[0]) if not result.folds.empty else 0,
        "prediction_rows": int(result.predictions.shape[0]),
        "git_commit": _git_commit(),
        "data_download_stamp": data_stamp,
    }
    if extras:
        manifest.update(extras)
    return _jsonable(manifest)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, _dt.datetime, _dt.date)):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            return str(value)
    return value


def save_results(
    result: BacktestResult,
    output_directory: str | Path,
    *,
    config_path: str | Path,
    manifest_extras: dict | None = None,
) -> dict[str, Path]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)

    paths = {
        "daily_returns": output / "daily_returns.csv",
        "predictions": output / "predictions.csv.gz",
        "coefficients": output / "coefficients.csv",
        "folds": output / "folds.csv",
        "validation": output / "validation.csv",
        "graph_edges": output / "graph_edges.csv.gz",
        "metrics": output / "metrics.csv",
        "annual_metrics": output / "annual_metrics.csv",
        "regional_metrics": output / "regional_metrics.csv",
        "performance_plot": output / "performance.png",
        "manifest": output / "manifest.json",
    }

    result.daily_returns.to_csv(paths["daily_returns"], index_label="date")
    result.predictions.reset_index().to_csv(
        paths["predictions"],
        index=False,
        compression="gzip",
    )
    result.coefficients.to_csv(paths["coefficients"], index=False)
    result.folds.to_csv(paths["folds"], index=False)
    result.validation.to_csv(paths["validation"], index=False)

    edge_frames: list[pd.DataFrame] = []
    for (alpha, beta), snapshots in result.snapshots.items():
        edges = graph_edges(snapshots)
        edges["alpha"] = alpha
        edges["beta"] = beta
        edge_frames.append(edges)
    if edge_frames:
        pd.concat(edge_frames, ignore_index=True).to_csv(
            paths["graph_edges"],
            index=False,
            compression="gzip",
        )
    else:
        pd.DataFrame(columns=["date", "source", "target", "weight"]).to_csv(
            paths["graph_edges"], index=False, compression="gzip"
        )

    metrics = pd.DataFrame(
        {
            "strategy_scaled": performance_metrics(result.daily_returns["strategy_return"]),
            "strategy_unscaled": performance_metrics(
                result.daily_returns["net_return_unscaled"]
            ),
        }
    ).T
    metrics.to_csv(paths["metrics"], index_label="series")
    annual_metrics(result.daily_returns["strategy_return"]).to_csv(
        paths["annual_metrics"],
        index_label="year",
    )

    region_rows: list[pd.Series] = []
    if "region" in result.predictions.columns:
        for region, region_predictions in result.predictions.groupby("region"):
            regional_return = region_predictions.groupby(level="date")[
                "gross_asset_return"
            ].mean()
            row = performance_metrics(regional_return)
            row.name = region
            region_rows.append(row)
    pd.DataFrame(region_rows).to_csv(paths["regional_metrics"], index_label="region")

    _plot_performance(result.daily_returns, paths["performance_plot"])
    manifest = {
        "config": str(Path(config_path).resolve()),
        "oos_start": str(result.daily_returns.index.min().date()),
        "oos_end": str(result.daily_returns.index.max().date()),
        "folds": int(result.folds.shape[0]) if not result.folds.empty else 0,
        "prediction_rows": int(result.predictions.shape[0]),
    }
    if manifest_extras:
        manifest.update(_jsonable(manifest_extras))
    paths["manifest"].write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return paths
