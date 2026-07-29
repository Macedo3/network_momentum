from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class DataConfig:
    universe_path: Path
    cache_dir: Path
    start: str = "2005-01-01"
    end: str | None = None
    auto_adjust: bool = True
    repair: bool = True
    threads: bool = False
    min_tickers: int = 8
    download_fields: tuple[str, ...] = ("Close", "Open", "High", "Low", "Volume")
    base_currency: str = "USD"
    fx_map_path: Path | None = None
    benchmarks_path: Path | None = None
    max_abs_daily_return: float = 0.60
    max_stale_run_days: int = 10


@dataclass(frozen=True)
class FeatureConfig:
    volatility_span: int = 60
    volatility_min_periods: int = 20
    return_lookbacks: tuple[int, ...] = (1, 21, 63, 126, 252)
    macd_scales: tuple[tuple[int, int], ...] = ((8, 24), (16, 48), (32, 96))
    macd_price_std_window: int = 63
    macd_norm_std_window: int = 252
    winsor_halflife: int = 252
    winsor_limit: float = 5.0
    signal_lag_days: int = 1
    max_stale_days: int = 5


@dataclass(frozen=True)
class GraphConfig:
    lookbacks: tuple[int, ...] = (252, 504, 756, 1008, 1260)
    alpha_grid: tuple[float, ...] = (0.1,)
    beta_grid: tuple[float, ...] = (0.1,)
    rebalance_every: int = 21
    min_assets: int = 8
    maxiter: int = 300
    tolerance: float = 1e-7
    solver: str = "lbfgs"  # "lbfgs" (padrão, sem dependências) ou "cvxpy" (verificação)
    edge_threshold: float = 1e-4  # aresta relativa mínima (fração do maior peso) p/ topologia
    per_lookback_eligibility: bool = True


@dataclass(frozen=True)
class ModelConfig:
    initial_train_years: int = 10
    test_years: int = 5
    validation_fraction: float = 0.10
    embargo_days: int = 1
    target_annual_volatility: float = 0.15
    transaction_cost_bps: float = 1.0  # pseudo-custo da Eq. (14); custos reais em costs.csv
    portfolio_volatility_scaling: bool = True
    portfolio_volatility_span: int = 60
    max_portfolio_leverage: float = 5.0
    train_regions: tuple[str, ...] = ()
    test_regions: tuple[str, ...] = ()
    min_regression_samples: int = 250
    regression_method: str = "ols"  # "ols" | "ridge" | "lasso" | "elastic_net"
    ridge_lambda: float = 1e-4
    lasso_lambda: float = 1e-5
    elastic_net_l1_ratio: float = 0.5
    signal_threshold: float = 0.0  # |previsão| mínima para operar (0 = convenção do artigo)


@dataclass(frozen=True)
class CostConfig:
    path: Path | None = None
    scenario: str = "base"  # "conservative" | "base" | "optimistic"
    portfolio_notional_usd: float = 10_000_000.0
    max_participation: float = 0.05
    impact_coefficient: float = 1.0  # multiplicador da lei de raiz quadrada
    pseudo_cost_sweep_bps: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0)


@dataclass(frozen=True)
class ValidationConfig:
    seed: int = 42
    bootstrap_samples: int = 2000
    bootstrap_block_days: int = 21
    permutation_samples: int = 500
    cscv_blocks: int = 8


@dataclass(frozen=True)
class OutputConfig:
    directory: Path


@dataclass(frozen=True)
class AppConfig:
    data: DataConfig
    features: FeatureConfig
    graph: GraphConfig
    model: ModelConfig
    costs: CostConfig
    validation: ValidationConfig
    output: OutputConfig
    source_path: Path = field(repr=False)

    def validate(self) -> None:
        if not 0 < self.model.validation_fraction < 0.5:
            raise ValueError("validation_fraction deve estar entre 0 e 0.5.")
        if self.model.initial_train_years <= 0 or self.model.test_years <= 0:
            raise ValueError("As janelas de treino e teste devem ser positivas.")
        if not self.graph.lookbacks or min(self.graph.lookbacks) < 2:
            raise ValueError("graph.lookbacks deve conter janelas com pelo menos 2 dias.")
        if any(x <= 0 for x in (*self.graph.alpha_grid, *self.graph.beta_grid)):
            raise ValueError("alpha_grid e beta_grid devem conter apenas valores positivos.")
        if self.graph.rebalance_every <= 0:
            raise ValueError("graph.rebalance_every deve ser positivo.")
        if self.graph.solver not in ("lbfgs", "cvxpy"):
            raise ValueError("graph.solver deve ser 'lbfgs' ou 'cvxpy'.")
        if self.features.signal_lag_days < 0:
            raise ValueError("signal_lag_days não pode ser negativo.")
        if self.model.embargo_days < 0:
            raise ValueError("embargo_days não pode ser negativo.")
        if self.model.regression_method not in ("ols", "ridge", "lasso", "elastic_net"):
            raise ValueError("regression_method inválido.")
        if self.costs.scenario not in ("conservative", "base", "optimistic"):
            raise ValueError("costs.scenario deve ser conservative, base ou optimistic.")
        if not 0 < self.costs.max_participation <= 1:
            raise ValueError("costs.max_participation deve estar em (0, 1].")


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _tuplify_pairs(values: list[list[int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return tuple((int(pair[0]), int(pair[1])) for pair in values)


def load_config(path: str | Path) -> AppConfig:
    source = Path(path).resolve()
    with source.open("rb") as stream:
        raw = tomllib.load(stream)
    base = source.parent

    data_raw = dict(raw.get("data", {}))
    data_raw["universe_path"] = _resolve(base, data_raw.get("universe_path", "universe.csv"))
    data_raw["cache_dir"] = _resolve(base, data_raw.get("cache_dir", "../data/cache"))
    if not data_raw.get("end"):
        data_raw["end"] = None
    if "download_fields" in data_raw:
        data_raw["download_fields"] = tuple(str(x) for x in data_raw["download_fields"])
    for optional_path in ("fx_map_path", "benchmarks_path"):
        if data_raw.get(optional_path):
            data_raw[optional_path] = _resolve(base, data_raw[optional_path])
        else:
            data_raw[optional_path] = None

    feature_raw = dict(raw.get("features", {}))
    if "return_lookbacks" in feature_raw:
        feature_raw["return_lookbacks"] = tuple(int(x) for x in feature_raw["return_lookbacks"])
    if "macd_scales" in feature_raw:
        feature_raw["macd_scales"] = _tuplify_pairs(feature_raw["macd_scales"])

    graph_raw = dict(raw.get("graph", {}))
    for key in ("lookbacks", "alpha_grid", "beta_grid"):
        if key in graph_raw:
            graph_raw[key] = tuple(graph_raw[key])

    model_raw = dict(raw.get("model", {}))
    for key in ("train_regions", "test_regions"):
        if key in model_raw:
            model_raw[key] = tuple(str(x) for x in model_raw[key])

    costs_raw = dict(raw.get("costs", {}))
    if costs_raw.get("path"):
        costs_raw["path"] = _resolve(base, costs_raw["path"])
    else:
        costs_raw["path"] = None
    if "pseudo_cost_sweep_bps" in costs_raw:
        costs_raw["pseudo_cost_sweep_bps"] = tuple(
            float(x) for x in costs_raw["pseudo_cost_sweep_bps"]
        )

    validation_raw = dict(raw.get("validation", {}))

    output_raw = dict(raw.get("output", {}))
    output_raw["directory"] = _resolve(base, output_raw.get("directory", "../outputs/latest"))

    config = AppConfig(
        data=DataConfig(**data_raw),
        features=FeatureConfig(**feature_raw),
        graph=GraphConfig(**graph_raw),
        model=ModelConfig(**model_raw),
        costs=CostConfig(**costs_raw),
        validation=ValidationConfig(**validation_raw),
        output=OutputConfig(**output_raw),
        source_path=source,
    )
    config.validate()
    return config
