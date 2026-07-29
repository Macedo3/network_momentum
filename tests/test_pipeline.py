from pathlib import Path

import pandas as pd
import pytest

from network_momentum.config import load_config
from network_momentum.pipeline import (
    PipelineOptions,
    run_full_pipeline,
    synthetic_market_data,
)

REPO_CONFIG = Path(__file__).resolve().parents[1] / "config" / "default.toml"


def test_synthetic_data_is_reproducible() -> None:
    frames_a, universe_a = synthetic_market_data(seed=42)
    frames_b, universe_b = synthetic_market_data(seed=42)
    pd.testing.assert_frame_equal(frames_a["Close"], frames_b["Close"])
    pd.testing.assert_frame_equal(universe_a, universe_b)
    frames_c, _ = synthetic_market_data(seed=7)
    assert not frames_a["Close"].equals(frames_c["Close"])


@pytest.mark.slow
def test_full_pipeline_smoke_end_to_end(safe_tmp_path: Path) -> None:
    """Execução ponta a ponta em dados sintéticos, sem internet: dados ->
    features -> grafos -> GMOM -> benchmarks -> custos -> validação ->
    robustez -> gráficos -> manifest -> respostas do formulário."""
    config = load_config(REPO_CONFIG)
    options = PipelineOptions(
        profile="smoke",
        output_dir=safe_tmp_path / "outputs",
        make_plots=True,
        run_lookback_ablation_flag=False,
        run_edge_ablation_flag=False,
        run_regression_variants=True,
    )
    artifacts = run_full_pipeline(config, options)

    gmom = artifacts["gmom"]
    assert not gmom.daily_returns.empty
    assert gmom.daily_returns["strategy_return"].notna().all()

    output_dir = artifacts["output_dir"]
    for required in (
        "run_manifest.json",
        "form_answers.md",
        "daily_returns.csv",
        "metrics.csv",
    ):
        assert (output_dir / required).exists(), required
    tables = output_dir / "tables"
    for table in (
        "strategies_net_metrics",
        "sharpe_vs_cost",
        "cost_scenarios",
        "coefficient_stability",
        "graph_topology",
        "signal_threshold_study",
        "deflated_sharpe",
    ):
        assert (tables / f"{table}.csv").exists(), table
    figures = output_dir / "figures"
    assert (figures / "figures_index.csv").exists()
    saved_figures = list(figures.glob("*.png"))
    assert len(saved_figures) >= 20
    for png in saved_figures:
        assert png.with_suffix(".svg").exists()
        assert png.with_suffix(".csv").exists()

    suite = artifacts["benchmark_suite"]
    assert {"gmom", "linreg", "macd", "long_only", "equal_weight", "signcombo"}.issubset(
        suite.results
    )
    answers = artifacts["form_answers"]
    assert "9. Nome do Robô" in answers["curta"]
    assert "15. Benchmark" in answers["tecnica"]
