from __future__ import annotations

"""Fábrica de gráficos do relatório.

Regras aplicadas a TODOS os gráficos:
- título, eixos com unidade, legenda, período e fonte no rodapé;
- nota metodológica curta no rodapé;
- 300 dpi, PNG + SVG e os dados de origem exportados em CSV com o mesmo nome;
- paleta Okabe-Ito (segura para daltonismo), eixo y honesto (sem cortes que
  exagerem efeitos), sem dupla escala não sinalizada.
"""

import os
from pathlib import Path
import tempfile

_matplotlib_cache = Path(tempfile.gettempdir()) / "network_momentum_matplotlib"
_matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_matplotlib_cache))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Paleta Okabe-Ito (acessível) + cinza neutro.
PALETTE = [
    "#0072B2",  # azul
    "#E69F00",  # laranja
    "#009E73",  # verde
    "#D55E00",  # vermelho-tijolo
    "#CC79A7",  # rosa
    "#56B4E9",  # azul-claro
    "#F0E442",  # amarelo
    "#000000",  # preto
]
NEGATIVE = "#9D0208"
DPI = 300


class FigureSaver:
    """Salva cada figura em PNG (300 dpi) + SVG e exporta os dados em CSV."""

    def __init__(self, directory: str | Path, *, source: str, period: str):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.source = source
        self.period = period
        self.index: list[dict[str, str]] = []

    def finalize(
        self,
        figure: plt.Figure,
        name: str,
        data: pd.DataFrame | pd.Series,
        *,
        note: str,
    ) -> dict[str, Path]:
        footer = f"Fonte: {self.source} | Período: {self.period} | {note}"
        figure.text(
            0.01,
            0.005,
            footer,
            fontsize=6.5,
            color="#444444",
            ha="left",
            va="bottom",
            wrap=True,
        )
        figure.tight_layout(rect=(0, 0.03, 1, 1))
        paths = {
            "png": self.directory / f"{name}.png",
            "svg": self.directory / f"{name}.svg",
            "csv": self.directory / f"{name}.csv",
        }
        figure.savefig(paths["png"], dpi=DPI, bbox_inches="tight")
        figure.savefig(paths["svg"], bbox_inches="tight")
        plt.close(figure)
        frame = data.to_frame() if isinstance(data, pd.Series) else data
        frame.to_csv(paths["csv"])
        self.index.append({"figure": name, "note": note})
        return paths

    def write_index(self) -> Path:
        path = self.directory / "figures_index.csv"
        pd.DataFrame(self.index).to_csv(path, index=False)
        return path


def _new_axes(figsize=(10, 5.5)):
    figure, axes = plt.subplots(figsize=figsize)
    axes.grid(alpha=0.25)
    return figure, axes


def wealth_curve(returns: pd.Series) -> pd.Series:
    return (1.0 + returns.fillna(0.0)).cumprod()


def plot_equity_curves(
    saver: FigureSaver,
    series: dict[str, pd.Series],
    *,
    name: str,
    title: str,
    note: str,
    log_scale: bool = True,
) -> None:
    figure, axes = _new_axes()
    data = {}
    for color, (label, returns) in zip(PALETTE * 3, series.items()):
        wealth = wealth_curve(returns.dropna())
        axes.plot(wealth.index, wealth, label=label, color=color, linewidth=1.4)
        data[label] = wealth
    if log_scale:
        axes.set_yscale("log")
    axes.set_title(title)
    axes.set_xlabel("Data")
    axes.set_ylabel("Riqueza acumulada (início = 1, escala log)" if log_scale else "Riqueza acumulada (início = 1)")
    axes.legend(fontsize=8, ncol=2)
    saver.finalize(figure, name, pd.DataFrame(data), note=note)


def plot_drawdown(saver: FigureSaver, returns: pd.Series, *, name: str, title: str, note: str) -> None:
    wealth = wealth_curve(returns.dropna())
    drawdown = wealth.div(wealth.cummax()).sub(1.0)
    figure, axes = _new_axes(figsize=(10, 3.8))
    axes.fill_between(drawdown.index, drawdown.to_numpy(), 0, color=NEGATIVE, alpha=0.6)
    axes.set_title(title)
    axes.set_xlabel("Data")
    axes.set_ylabel("Drawdown (fração do pico)")
    saver.finalize(figure, name, drawdown.rename("drawdown"), note=note)


def plot_annual_returns(saver: FigureSaver, returns: pd.Series, *, name: str, title: str, note: str) -> None:
    annual = returns.groupby(returns.index.year).apply(lambda s: (1 + s).prod() - 1)
    figure, axes = _new_axes()
    colors = [PALETTE[0] if v >= 0 else NEGATIVE for v in annual]
    axes.bar(annual.index.astype(str), annual.to_numpy(), color=colors)
    axes.set_title(title)
    axes.set_xlabel("Ano")
    axes.set_ylabel("Retorno composto no ano (fração)")
    axes.axhline(0, color="black", linewidth=0.8)
    plt.setp(axes.get_xticklabels(), rotation=45, ha="right")
    saver.finalize(figure, name, annual.rename("annual_return"), note=note)


def plot_rolling_series(
    saver: FigureSaver,
    series: dict[str, pd.Series],
    *,
    name: str,
    title: str,
    ylabel: str,
    note: str,
    hline: float | None = None,
) -> None:
    figure, axes = _new_axes()
    data = {}
    for color, (label, values) in zip(PALETTE * 3, series.items()):
        axes.plot(values.index, values, label=label, color=color, linewidth=1.2)
        data[label] = values
    if hline is not None:
        axes.axhline(hline, color="black", linewidth=0.8, linestyle="--")
    axes.set_title(title)
    axes.set_xlabel("Data")
    axes.set_ylabel(ylabel)
    axes.legend(fontsize=8)
    saver.finalize(figure, name, pd.DataFrame(data), note=note)


def rolling_sharpe(returns: pd.Series, window: int = 252) -> pd.Series:
    mean = returns.rolling(window, min_periods=window // 2).mean()
    std = returns.rolling(window, min_periods=window // 2).std(ddof=0)
    return (mean / std * np.sqrt(252.0)).rename("rolling_sharpe")


def rolling_volatility(returns: pd.Series, window: int = 63) -> pd.Series:
    return (
        returns.rolling(window, min_periods=window // 2).std(ddof=0) * np.sqrt(252.0)
    ).rename("rolling_volatility")


def rolling_beta(strategy: pd.Series, benchmark: pd.Series, window: int = 252) -> pd.Series:
    aligned = pd.concat([strategy.rename("s"), benchmark.rename("b")], axis=1).dropna()
    covariance = aligned["s"].rolling(window, min_periods=window // 2).cov(aligned["b"])
    variance = aligned["b"].rolling(window, min_periods=window // 2).var(ddof=0)
    return (covariance / variance).rename("rolling_beta")


def plot_bars(
    saver: FigureSaver,
    values: pd.Series,
    *,
    name: str,
    title: str,
    xlabel: str,
    ylabel: str,
    note: str,
    horizontal: bool = False,
) -> None:
    figure, axes = _new_axes(figsize=(10, max(4.0, 0.32 * len(values)) if horizontal else 5.0))
    labels = [str(i) for i in values.index]
    colors = [PALETTE[0] if v >= 0 else NEGATIVE for v in values.to_numpy()]
    if horizontal:
        axes.barh(labels, values.to_numpy(), color=colors)
        axes.set_xlabel(ylabel)
        axes.set_ylabel(xlabel)
        axes.axvline(0, color="black", linewidth=0.8)
    else:
        axes.bar(labels, values.to_numpy(), color=colors)
        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)
        axes.axhline(0, color="black", linewidth=0.8)
        plt.setp(axes.get_xticklabels(), rotation=45, ha="right")
    axes.set_title(title)
    saver.finalize(figure, name, values, note=note)


def plot_stacked_costs(
    saver: FigureSaver,
    cost_breakdown: pd.DataFrame,
    *,
    name: str,
    title: str,
    note: str,
) -> None:
    annual = cost_breakdown.mul(1e4).groupby(cost_breakdown.index.year).mean() * 252.0
    figure, axes = _new_axes()
    bottom = np.zeros(len(annual))
    for color, column in zip(PALETTE * 3, annual.columns):
        axes.bar(annual.index.astype(str), annual[column], bottom=bottom, label=column, color=color)
        bottom += annual[column].to_numpy()
    axes.set_title(title)
    axes.set_xlabel("Ano")
    axes.set_ylabel("Custo anualizado (bps do PL)")
    axes.legend(fontsize=7, ncol=2)
    plt.setp(axes.get_xticklabels(), rotation=45, ha="right")
    saver.finalize(figure, name, annual, note=note)


def plot_sharpe_vs_cost(
    saver: FigureSaver,
    sweep: pd.DataFrame,
    *,
    breakeven_bps: float,
    name: str,
    title: str,
    note: str,
) -> None:
    figure, axes = _new_axes()
    axes.plot(sweep.index, sweep["net_sharpe"], marker="o", color=PALETTE[0], label="Sharpe líquido")
    axes.axhline(0, color="black", linewidth=0.8)
    if np.isfinite(breakeven_bps):
        axes.axvline(
            breakeven_bps,
            color=NEGATIVE,
            linestyle="--",
            label=f"Break-even: {breakeven_bps:.1f} bps",
        )
    axes.set_title(title)
    axes.set_xlabel("Custo linear por unidade de turnover (bps)")
    axes.set_ylabel("Sharpe líquido (a.a.)")
    axes.legend(fontsize=8)
    saver.finalize(figure, name, sweep, note=note)


def plot_heatmap(
    saver: FigureSaver,
    matrix: pd.DataFrame,
    *,
    name: str,
    title: str,
    note: str,
    fmt: str = "{:.2f}",
    cmap: str = "RdBu_r",
    center_zero: bool = True,
) -> None:
    figure, axes = plt.subplots(
        figsize=(max(6.0, 0.65 * matrix.shape[1] + 2), max(4.5, 0.5 * matrix.shape[0] + 1.5))
    )
    values = matrix.to_numpy(dtype=float)
    if center_zero:
        limit = np.nanmax(np.abs(values)) or 1.0
        image = axes.imshow(values, cmap=cmap, vmin=-limit, vmax=limit)
    else:
        image = axes.imshow(values, cmap=cmap)
    axes.set_xticks(range(matrix.shape[1]), [str(c) for c in matrix.columns], rotation=45, ha="right", fontsize=7)
    axes.set_yticks(range(matrix.shape[0]), [str(i) for i in matrix.index], fontsize=7)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isfinite(values[i, j]):
                axes.text(j, i, fmt.format(values[i, j]), ha="center", va="center", fontsize=6.5)
    figure.colorbar(image, ax=axes, shrink=0.8)
    axes.set_title(title)
    saver.finalize(figure, name, matrix, note=note)


def plot_learning_curves(
    saver: FigureSaver,
    learning: pd.DataFrame,
    *,
    name: str,
    title: str,
    note: str,
) -> None:
    figure, axes = _new_axes()
    for color, (split, block) in zip(PALETTE, learning.groupby("split")):
        block = block.sort_values("fraction")
        axes.plot(
            block["fraction"],
            block["signal_sharpe"],
            marker="o",
            label=f"Sharpe do sinal ({split})",
            color=color,
        )
    axes.set_title(title)
    axes.set_xlabel("Fração cronológica do treino utilizada")
    axes.set_ylabel("Sharpe do sinal (a.a.)")
    axes.legend(fontsize=8)
    saver.finalize(figure, name, learning, note=note)


def plot_histogram(
    saver: FigureSaver,
    samples: np.ndarray,
    *,
    observed: float,
    name: str,
    title: str,
    xlabel: str,
    note: str,
) -> None:
    figure, axes = _new_axes()
    axes.hist(samples, bins=50, color=PALETTE[0], alpha=0.75)
    axes.axvline(observed, color=NEGATIVE, linestyle="--", label=f"Observado: {observed:.2f}")
    axes.axvline(0.0, color="black", linewidth=0.8)
    axes.set_title(title)
    axes.set_xlabel(xlabel)
    axes.set_ylabel("Frequência")
    axes.legend(fontsize=8)
    saver.finalize(
        figure,
        name,
        pd.DataFrame({"samples": samples}),
        note=note,
    )


def plot_coefficient_paths(
    saver: FigureSaver,
    coefficients: pd.DataFrame,
    *,
    name: str,
    title: str,
    note: str,
) -> None:
    pivot = (
        coefficients[coefficients["feature"] != "intercept"]
        .pivot(index="fold", columns="feature", values="coefficient")
        .sort_index()
    )
    figure, axes = _new_axes()
    for color, column in zip(PALETTE * 3, pivot.columns):
        axes.plot(pivot.index, pivot[column], marker="o", label=column, color=color, linewidth=1.1)
    axes.axhline(0, color="black", linewidth=0.8)
    axes.set_title(title)
    axes.set_xlabel("Fold do walk-forward")
    axes.set_ylabel("Coeficiente da OLS")
    axes.set_xticks(pivot.index)
    axes.legend(fontsize=7, ncol=2)
    saver.finalize(figure, name, pivot, note=note)


def plot_autocorrelation_bars(
    saver: FigureSaver,
    residuals_by_lag: pd.Series,
    *,
    name: str,
    title: str,
    note: str,
) -> None:
    figure, axes = _new_axes(figsize=(8, 4.2))
    axes.bar(residuals_by_lag.index.astype(str), residuals_by_lag.to_numpy(), color=PALETTE[0])
    axes.axhline(0, color="black", linewidth=0.8)
    axes.set_title(title)
    axes.set_xlabel("Defasagem (dias)")
    axes.set_ylabel("Autocorrelação média dos resíduos")
    saver.finalize(figure, name, residuals_by_lag.rename("autocorrelation"), note=note)


def plot_graph_snapshot(
    saver: FigureSaver,
    snapshot,
    labels: dict[str, str],
    *,
    edge_threshold: float,
    name: str,
    title: str,
    note: str,
    max_edges: int = 150,
) -> None:
    """Grafo em layout circular agrupado por rótulo (região), sem networkx."""
    from .topology import binary_adjacency

    adjacency = (
        snapshot.raw_adjacency if snapshot.raw_adjacency is not None else snapshot.adjacency
    )
    assets = list(snapshot.assets)
    order = sorted(range(len(assets)), key=lambda k: (labels.get(assets[k], ""), assets[k]))
    angles = np.linspace(0, 2 * np.pi, len(assets), endpoint=False)
    positions = {
        order[k]: (np.cos(angles[k]), np.sin(angles[k])) for k in range(len(assets))
    }
    binary = binary_adjacency(adjacency, edge_threshold)
    edge_i, edge_j = np.nonzero(np.triu(binary, k=1))
    weights = adjacency[edge_i, edge_j]
    if len(weights) > max_edges:
        keep = np.argsort(weights)[-max_edges:]
        edge_i, edge_j, weights = edge_i[keep], edge_j[keep], weights[keep]

    groups = sorted({labels.get(a, "") for a in assets})
    group_color = {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(groups)}

    figure, axes = plt.subplots(figsize=(8.5, 8.5))
    max_weight = weights.max() if len(weights) else 1.0
    for i, j, w in zip(edge_i, edge_j, weights):
        x = [positions[i][0], positions[j][0]]
        y = [positions[i][1], positions[j][1]]
        axes.plot(x, y, color="#777777", alpha=min(0.85, 0.15 + 0.7 * w / max_weight), linewidth=0.5 + 2.0 * w / max_weight, zorder=1)
    for k, asset in enumerate(assets):
        x, y = positions[k]
        axes.scatter(x, y, s=110, color=group_color[labels.get(asset, "")], zorder=2, edgecolors="white")
        axes.text(x * 1.12, y * 1.12, asset, fontsize=6, ha="center", va="center")
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=color, label=group, markersize=7)
        for group, color in group_color.items()
    ]
    axes.legend(handles=handles, fontsize=7, loc="upper right")
    axes.set_title(title)
    axes.set_xlim(-1.35, 1.35)
    axes.set_ylim(-1.35, 1.35)
    axes.axis("off")
    edges_frame = pd.DataFrame(
        {
            "source": [assets[i] for i in edge_i],
            "target": [assets[j] for j in edge_j],
            "weight": weights,
        }
    )
    saver.finalize(figure, name, edges_frame, note=note)


def plot_topology_panels(
    saver: FigureSaver,
    topology: pd.DataFrame,
    *,
    name: str,
    title: str,
    note: str,
) -> None:
    columns = [
        ("edge_sparsity", "Densidade de arestas (fração)"),
        ("average_degree", "Grau médio"),
        ("clustering_coefficient", "Coeficiente de clustering"),
        ("community_ratio", "Community ratio"),
        ("jaccard_previous", "Jaccard vs snapshot anterior"),
    ]
    available = [(c, label) for c, label in columns if c in topology.columns]
    figure, axes_list = plt.subplots(len(available), 1, figsize=(10, 2.1 * len(available)), sharex=True)
    if len(available) == 1:
        axes_list = [axes_list]
    for axes, (column, label) in zip(axes_list, available):
        axes.plot(topology.index, topology[column], color=PALETTE[0], linewidth=1.0)
        axes.set_ylabel(label, fontsize=7)
        axes.grid(alpha=0.25)
    axes_list[0].set_title(title)
    axes_list[-1].set_xlabel("Data")
    saver.finalize(figure, name, topology[[c for c, _ in available]], note=note)
