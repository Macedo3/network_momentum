from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import load_config
from .metrics import performance_metrics
from .pipeline import PipelineOptions, run_full_pipeline


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="network-momentum",
        description=(
            "Network Momentum para equities globais — pipeline completo: dados, "
            "features, grafos, backtest, benchmarks, custos, validação, robustez, "
            "gráficos, tabelas e respostas do formulário, em um único comando."
        ),
    )
    parser.add_argument(
        "--config",
        default="config/default.toml",
        help="Arquivo TOML de configuração (padrão: config/default.toml).",
    )
    parser.add_argument(
        "--profile",
        default="full",
        choices=("full", "fast", "smoke"),
        help=(
            "full = tudo, inclusive ablações caras; fast = pula ablações de "
            "lookback/arestas; smoke = dados sintéticos sem internet (~1 min)."
        ),
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignora o cache local e baixa novamente os preços.",
    )
    parser.add_argument(
        "--output",
        help="Sobrescreve o diretório de saída definido no TOML.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Não gera os gráficos (apenas tabelas e manifest).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    config = load_config(args.config)
    options = PipelineOptions(
        profile=args.profile,
        refresh_data=args.refresh,
        output_dir=Path(args.output).resolve() if args.output else None,
        make_plots=not args.no_plots,
        run_lookback_ablation_flag=args.profile == "full",
        run_edge_ablation_flag=args.profile == "full",
        run_regression_variants=args.profile != "smoke",
    )
    artifacts = run_full_pipeline(config, options)

    gmom = artifacts["gmom"]
    metrics = performance_metrics(gmom.daily_returns["strategy_return"])
    breakeven = artifacts.get("breakeven_bps")
    summary = (
        "Concluído | "
        f"OOS {gmom.daily_returns.index.min().date()}..{gmom.daily_returns.index.max().date()} | "
        f"Sharpe líquido {metrics['sharpe']:.3f} | "
        f"Retorno a.a. {metrics['annual_return']:.2%} | "
        f"Vol a.a. {metrics['annual_volatility']:.2%}"
    )
    if breakeven is not None and breakeven == breakeven:
        summary += f" | Break-even {breakeven:.1f} bps"
    print(summary)
    print(f"Saídas: {artifacts['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
