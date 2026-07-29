from __future__ import annotations

"""Pipeline completo — ponto de entrada único do projeto.

`run_full_pipeline` executa, em ordem e com um único comando:

 1. download OHLCV do universo + pares FX + benchmarks externos (com cache);
 2. verificações de qualidade de dados;
 3. features (Eqs. 1–3), grafos (Eqs. 4–6), propagação (Eq. 7);
 4. estratégia principal GMOM (Eq. 8–9) com walk-forward e seleção nested de α/β;
 5. benchmarks metodológicos (LinReg, RegCombo, SignCombo, MACD, LongOnly, EW);
 6. custos reais por bolsa em três cenários + curva Sharpe×custo + break-even;
 7. conversão para a moeda-base (USD) e decomposição cambial;
 8. suíte de validação (bootstrap, permutação, DSR, PBO, learning curve, ...);
 9. robustez (ablações, long/short, regimes, concentração, capacidade);
10. topologia dos grafos (esparsidade, grau, clustering, community, Jaccard);
11. todos os gráficos (300 dpi, PNG+SVG+CSV) e tabelas;
12. manifest de reprodutibilidade e respostas do formulário.

Perfis: "full" roda tudo; "fast" pula as ablações caras (lookback e arestas);
"smoke" usa dados sintéticos, sem rede, para testar o encanamento de ponta a
ponta em ~1 minuto.
"""

from dataclasses import dataclass, replace
import json
import logging
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .backtest import BacktestResult, build_network_bundle, run_backtest
from .benchmarks import (
    compare_to_external,
    external_benchmark_returns,
    load_benchmark_meta,
    run_benchmark_suite,
)
from .config import AppConfig, load_config
from .costs import (
    average_daily_value_base,
    breakeven_cost_bps,
    build_ticker_costs,
    capacity_estimate,
    load_cost_table,
    sharpe_versus_cost,
)
from .data import download_auxiliary_close, download_price_history, run_data_quality_checks
from .features import build_feature_set
from .fx import fx_rates_to_base, forward_returns_in_base, load_fx_map
from .metrics import (
    annual_metrics,
    performance_metrics,
    portfolio_state_metrics,
)
from .portfolio import ImpactInputs, build_portfolio
from .reporting import build_manifest, save_results
from .robustness import (
    asset_contribution,
    calendar_window_metrics,
    concentration_statistics,
    long_short_decomposition,
    metrics_by_group,
    remove_best_assets_test,
    remove_best_years_test,
    risk_concentration,
    run_edge_mask_ablation,
    run_feature_ablation,
    run_lookback_ablation,
    start_date_sensitivity,
    volatility_regime_metrics,
)
from .topology import topology_time_series
from .universe import load_universe, universe_fingerprint
from .validation import (
    block_bootstrap_sharpe,
    circular_shift_permutation_test,
    coefficient_stability,
    deflated_sharpe_ratio,
    learning_curve,
    multiple_testing_adjustment,
    probability_of_backtest_overfitting,
    residual_diagnostics,
    univariate_feature_r2,
)
from .form_answers import answers_markdown, build_run_facts, form_answers

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineOptions:
    profile: str = "full"  # "full" | "fast" | "smoke"
    refresh_data: bool = False
    output_dir: Path | None = None
    make_plots: bool = True
    run_lookback_ablation_flag: bool = True
    run_edge_ablation_flag: bool = True
    run_regression_variants: bool = True


def synthetic_market_data(
    seed: int = 42,
    n_assets: int = 14,
    n_days: int = 1500,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Universo sintético (sem rede) para o perfil smoke: preços lognormais com
    fatores regionais comuns, volume positivo e metadados compatíveis com o
    universo real."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n_days)
    regions = ["United States", "Eurozone", "Japan", "Brazil"]
    exchanges = {"United States": "Nasdaq", "Eurozone": "Xetra", "Japan": "TSE", "Brazil": "B3"}
    currencies = {"United States": "USD", "Eurozone": "USD", "Japan": "USD", "Brazil": "USD"}
    factor = rng.normal(0.0, 0.008, size=(n_days, len(regions)))
    records = []
    prices = {}
    volumes = {}
    for k in range(n_assets):
        region = regions[k % len(regions)]
        ticker = f"SYN{k:02d}"
        idiosyncratic = rng.normal(0.0002, 0.012, n_days)
        returns = idiosyncratic + factor[:, regions.index(region)]
        prices[ticker] = 50.0 * np.exp(np.cumsum(returns))
        volumes[ticker] = rng.integers(500_000, 5_000_000, n_days).astype(float)
        records.append(
            {
                "ticker": ticker,
                "name": f"Synthetic {k}",
                "region": region,
                "country": region,
                "exchange": exchanges[region],
                "currency": currencies[region],
                "sector": ["Financials", "Energy", "Information Technology"][k % 3],
                "short_eligible": True,
                "borrow_fee_annual_bps_estimate": 50.0,
                "data_source": "synthetic",
                "note": "",
            }
        )
    frames = {
        "Close": pd.DataFrame(prices, index=dates),
        "Volume": pd.DataFrame(volumes, index=dates),
    }
    universe = pd.DataFrame.from_records(records).set_index("ticker", drop=False)
    return frames, universe


def smoke_config(base: AppConfig) -> AppConfig:
    """Config reduzida para o perfil smoke (dados sintéticos, ~1 minuto)."""
    return AppConfig(
        data=replace(
            base.data,
            fx_map_path=None,
            benchmarks_path=None,
            min_tickers=8,
        ),
        features=replace(
            base.features,
            return_lookbacks=(1, 5, 21, 42, 63),
            macd_scales=((4, 12), (8, 24), (16, 48)),
            macd_price_std_window=21,
            macd_norm_std_window=63,
            winsor_halflife=63,
        ),
        graph=replace(
            base.graph,
            lookbacks=(63, 126),
            alpha_grid=(0.1,),
            beta_grid=(0.1,),
            rebalance_every=21,
            min_assets=8,
            maxiter=150,
        ),
        model=replace(
            base.model,
            initial_train_years=3,
            test_years=1,
            min_regression_samples=200,
        ),
        costs=replace(base.costs, pseudo_cost_sweep_bps=(0.0, 1.0, 5.0)),
        validation=replace(
            base.validation,
            bootstrap_samples=200,
            permutation_samples=100,
        ),
        output=base.output,
        source_path=base.source_path,
    )


def _save_table(tables_dir: Path, name: str, frame: pd.DataFrame | pd.Series) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    data = frame.to_frame() if isinstance(frame, pd.Series) else frame
    data.to_csv(tables_dir / f"{name}.csv")


def _threshold_study(
    gmom: BacktestResult,
    config: AppConfig,
    quantiles: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6),
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Sinal contínuo vs sign() e thresholds de convicção — sem reajuste."""
    rows = []
    returns_by_variant: dict[str, pd.Series] = {}
    base_predictions = gmom.predictions
    abs_prediction = base_predictions["prediction"].abs()
    variants: dict[str, pd.Series] = {}
    for quantile in quantiles:
        cutoff = float(abs_prediction.quantile(quantile)) if quantile > 0 else 0.0
        positions = np.sign(base_predictions["prediction"]).where(
            abs_prediction >= cutoff, 0.0
        )
        variants[f"sign_q{int(quantile * 100):02d}"] = positions
    scale = abs_prediction.mean()
    if scale > 0:
        variants["continuous"] = base_predictions["prediction"] / scale
    for name, positions in variants.items():
        modified = base_predictions.copy()
        modified["position"] = positions
        portfolio = build_portfolio(
            modified,
            target_annual_volatility=config.model.target_annual_volatility,
            volatility_scaling=config.model.portfolio_volatility_scaling,
            volatility_span=config.model.portfolio_volatility_span,
            max_leverage=config.model.max_portfolio_leverage,
            pseudo_cost_bps=config.model.transaction_cost_bps,
        )
        row = performance_metrics(portfolio.daily["strategy_return"])
        row["mean_turnover"] = float(portfolio.daily["turnover_final"].mean())
        row.name = name
        rows.append(row)
        returns_by_variant[name] = portfolio.daily["strategy_return"]
    frame = pd.DataFrame(rows)
    frame.index.name = "variant"
    return frame, returns_by_variant


def run_full_pipeline(
    config: AppConfig | str | Path,
    options: PipelineOptions | None = None,
) -> dict[str, object]:
    started = time.time()
    options = options or PipelineOptions()
    if not isinstance(config, AppConfig):
        config = load_config(config)
    output_dir = Path(options.output_dir or config.output.directory)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, object] = {"config": config, "output_dir": output_dir}
    seed = config.validation.seed
    rng_note = f"seed={seed}"
    LOGGER.info("Pipeline iniciado (perfil=%s, %s).", options.profile, rng_note)

    # ------------------------------------------------------------------ dados
    if options.profile == "smoke":
        config = smoke_config(config)
        artifacts["config"] = config
        frames, universe = synthetic_market_data(seed=seed)
        prices = frames["Close"]
    else:
        universe = load_universe(config.data.universe_path)
        frames = download_price_history(
            universe["ticker"], config.data, refresh=options.refresh_data
        )
        prices = frames["Close"]
        universe = universe.reindex(prices.columns)
    artifacts["universe"] = universe
    quality = run_data_quality_checks(frames, config.data)
    _save_table(tables_dir, "data_quality", quality)
    artifacts["data_quality"] = quality

    fx_map = load_fx_map(config.data.fx_map_path) if config.data.fx_map_path else None
    rates = pd.DataFrame(index=prices.index)
    if fx_map is not None:
        fx_tickers = [t for t in fx_map["pair_ticker"].unique() if t]
        fx_closes = download_auxiliary_close(
            fx_tickers, config.data, refresh=options.refresh_data
        )
        rates = fx_rates_to_base(fx_closes, fx_map, config.data.base_currency)
    artifacts["fx_rates"] = rates

    benchmark_meta = None
    external_returns = pd.DataFrame()
    if config.data.benchmarks_path:
        benchmark_meta = load_benchmark_meta(config.data.benchmarks_path)
        benchmark_closes = download_auxiliary_close(
            benchmark_meta.index.tolist(), config.data, refresh=options.refresh_data
        )
        external_returns = external_benchmark_returns(
            benchmark_closes,
            benchmark_meta,
            rates,
            base_currency=config.data.base_currency,
        )
    artifacts["external_returns"] = external_returns

    # -------------------------------------------------------------- features
    feature_set = build_feature_set(prices, config.features)
    forward_base = None
    if fx_map is not None and not rates.empty:
        forward_base = forward_returns_in_base(
            prices,
            universe["currency"],
            rates,
            config.data.base_currency,
            pd.DatetimeIndex(feature_set.dates),
        )
    artifacts["feature_set"] = feature_set

    # ---------------------------------------------------------------- custos
    ticker_costs = None
    impact_inputs = None
    adv_base = pd.DataFrame()
    if config.costs.path:
        cost_table = load_cost_table(config.costs.path)
        ticker_costs = build_ticker_costs(
            cost_table,
            universe,
            scenario=config.costs.scenario,
            base_currency=config.data.base_currency,
        )
        _save_table(tables_dir, "cost_model_per_ticker", ticker_costs.as_frame())
        if "Volume" in frames:
            pence = set(universe.index[universe["exchange"] == "LSE"])
            adv_base = average_daily_value_base(
                prices,
                frames["Volume"],
                universe["currency"],
                rates,
                base_currency=config.data.base_currency,
                pence_quoted=pence,
            )
            impact_inputs = ImpactInputs(
                adv_base=adv_base,
                portfolio_notional=config.costs.portfolio_notional_usd,
                impact_coefficient=config.costs.impact_coefficient,
                max_participation=config.costs.max_participation,
            )
    artifacts["ticker_costs"] = ticker_costs

    # ------------------------------------------------------ grafos + backtest
    network_bundle = build_network_bundle(feature_set, config.graph)
    artifacts["network_bundle"] = network_bundle

    suite = run_benchmark_suite(
        feature_set,
        universe["region"],
        config.graph,
        config.model,
        network_bundle=network_bundle,
        forward_return_base=forward_base,
        ticker_costs=ticker_costs,
        impact_inputs=impact_inputs,
        short_eligible=universe["short_eligible"],
    )
    artifacts["benchmark_suite"] = suite
    gmom = suite.results["gmom"]
    artifacts["gmom"] = gmom
    strategy_net = gmom.daily_returns["strategy_return"]

    strategy_metrics = pd.DataFrame(
        {
            name: performance_metrics(res.daily_returns["strategy_return"])
            for name, res in suite.results.items()
        }
    ).T
    gross_metrics = pd.DataFrame(
        {
            name: performance_metrics(res.daily_returns["scaled_gross_return"])
            for name, res in suite.results.items()
        }
    ).T
    _save_table(tables_dir, "strategies_net_metrics", strategy_metrics)
    _save_table(tables_dir, "strategies_gross_metrics", gross_metrics)
    _save_table(tables_dir, "strategies_correlation", suite.correlation)
    _save_table(tables_dir, "strategies_sign_agreement", suite.sign_agreement)
    _save_table(tables_dir, "gmom_operational", portfolio_state_metrics(gmom.daily_returns))
    _save_table(tables_dir, "gmom_annual_metrics", annual_metrics(strategy_net))
    _save_table(tables_dir, "gmom_folds", gmom.folds.set_index("fold"))

    fold_oos = []
    for fold, block in gmom.daily_returns.groupby("fold"):
        row = performance_metrics(block["strategy_return"])
        row.name = int(fold)
        fold_oos.append(row)
    fold_oos_frame = pd.DataFrame(fold_oos).rename_axis("fold")
    _save_table(tables_dir, "gmom_metrics_by_fold", fold_oos_frame)
    artifacts["fold_metrics"] = fold_oos_frame

    # -------------------------------------------- custos: cenários e varredura
    cost_scenarios = {}
    if ticker_costs is not None and config.costs.path:
        cost_table = load_cost_table(config.costs.path)
        for scenario in ("conservative", "base", "optimistic"):
            scenario_costs = build_ticker_costs(
                cost_table,
                universe,
                scenario=scenario,
                base_currency=config.data.base_currency,
            )
            portfolio = build_portfolio(
                gmom.predictions,
                target_annual_volatility=config.model.target_annual_volatility,
                volatility_scaling=config.model.portfolio_volatility_scaling,
                volatility_span=config.model.portfolio_volatility_span,
                max_leverage=config.model.max_portfolio_leverage,
                pseudo_cost_bps=config.model.transaction_cost_bps,
                ticker_costs=scenario_costs,
                impact_inputs=impact_inputs,
            )
            row = performance_metrics(portfolio.daily["strategy_return"])
            row["annual_cost_real"] = float(
                portfolio.daily["cost_real_total"].mean() * 252.0
            )
            cost_scenarios[scenario] = row
        scenario_frame = pd.DataFrame(cost_scenarios).T.rename_axis("scenario")
        _save_table(tables_dir, "cost_scenarios", scenario_frame)
        artifacts["cost_scenarios"] = scenario_frame

    sweep = sharpe_versus_cost(
        gmom.daily_returns["scaled_gross_return"],
        gmom.daily_returns["turnover_final"],
        config.costs.pseudo_cost_sweep_bps,
    )
    breakeven = breakeven_cost_bps(
        gmom.daily_returns["scaled_gross_return"],
        gmom.daily_returns["turnover_final"],
    )
    _save_table(tables_dir, "sharpe_vs_cost", sweep)
    artifacts["sharpe_vs_cost"] = sweep
    artifacts["breakeven_bps"] = breakeven

    capacity = pd.DataFrame()
    if gmom.portfolio is not None and not adv_base.empty:
        capacity = capacity_estimate(
            gmom.portfolio.turnover_by_ticker,
            adv_base,
            gmom.daily_returns["active_assets"],
            max_participation=config.costs.max_participation,
        )
        _save_table(tables_dir, "capacity_estimate", capacity)
    artifacts["capacity"] = capacity

    # ------------------------------------------------------- moeda-base / FX
    if "strategy_return_base" in gmom.daily_returns:
        fx_comparison = pd.DataFrame(
            {
                "local_hedged_proxy": performance_metrics(strategy_net),
                "base_currency_unhedged": performance_metrics(
                    gmom.daily_returns["strategy_return_base"]
                ),
            }
        ).T
        _save_table(tables_dir, "currency_comparison", fx_comparison)
        artifacts["currency_comparison"] = fx_comparison

    # ------------------------------------------------------------ benchmarks
    external_stats = pd.DataFrame()
    if not external_returns.empty and benchmark_meta is not None:
        window = external_returns.reindex(strategy_net.index)
        returns_for_comparison = (
            gmom.daily_returns.get("strategy_return_base", strategy_net)
        )
        external_stats = compare_to_external(
            returns_for_comparison, window, benchmark_meta
        )
        _save_table(tables_dir, "external_benchmarks", external_stats)
        external_metrics = pd.DataFrame(
            {t: performance_metrics(window[t]) for t in window.columns}
        ).T
        _save_table(tables_dir, "external_benchmark_metrics", external_metrics)
    artifacts["external_stats"] = external_stats

    # ------------------------------------------------------------- validação
    bootstrap = block_bootstrap_sharpe(
        strategy_net,
        n_samples=config.validation.bootstrap_samples,
        block_days=config.validation.bootstrap_block_days,
        seed=seed,
    )
    artifacts["bootstrap"] = bootstrap

    permutation = None
    if gmom.portfolio is not None:
        forward_wide = gmom.predictions.reset_index().pivot(
            index="date", columns="ticker", values="forward_return"
        )
        try:
            permutation = circular_shift_permutation_test(
                gmom.portfolio.weights,
                forward_wide,
                n_samples=config.validation.permutation_samples,
                seed=seed,
            )
        except ValueError as error:
            LOGGER.warning("Teste de permutação não executado: %s", error)
    artifacts["permutation"] = permutation

    threshold_frame, variant_returns = _threshold_study(gmom, config)
    _save_table(tables_dir, "signal_threshold_study", threshold_frame)
    artifacts["threshold_study"] = threshold_frame

    regression_variants = pd.DataFrame()
    variant_daily: dict[str, pd.Series] = dict(variant_returns)
    for name, res in suite.results.items():
        variant_daily[name] = res.daily_returns["strategy_return"]
    if options.run_regression_variants:
        rows = []
        for method in ("ridge", "lasso", "elastic_net"):
            model_config = replace(config.model, regression_method=method)
            try:
                variant = run_backtest(
                    feature_set,
                    universe["region"],
                    config.graph,
                    model_config,
                    mode="network",
                    network_bundle=network_bundle,
                )
            except (RuntimeError, ValueError) as error:
                LOGGER.warning("Variante %s falhou: %s", method, error)
                continue
            row = performance_metrics(variant.daily_returns["strategy_return"])
            row.name = method
            rows.append(row)
            variant_daily[f"regression_{method}"] = variant.daily_returns[
                "strategy_return"
            ]
        ols_row = performance_metrics(strategy_net)
        ols_row.name = "ols"
        rows.insert(0, ols_row)
        regression_variants = pd.DataFrame(rows).rename_axis("method")
        _save_table(tables_dir, "regression_variants", regression_variants)
    artifacts["regression_variants"] = regression_variants

    n_trials = (
        len(config.graph.alpha_grid) * len(config.graph.beta_grid)
        + len(variant_daily)
    )
    dsr = deflated_sharpe_ratio(strategy_net, n_trials=n_trials)
    _save_table(tables_dir, "deflated_sharpe", dsr)
    artifacts["deflated_sharpe"] = dsr

    pbo_frame = pd.DataFrame(variant_daily).dropna(how="all")
    pbo = probability_of_backtest_overfitting(
        pbo_frame,
        n_blocks=config.validation.cscv_blocks,
        seed=seed,
    )
    artifacts["pbo"] = pbo
    _save_table(
        tables_dir,
        "pbo_summary",
        pd.Series(
            {
                "pbo": pbo.pbo,
                "n_configurations": pbo.n_configurations,
                "n_splits": pbo.n_splits,
                "applicable": pbo.applicable,
                "note": pbo.note,
            },
            name="value",
        ),
    )

    p_values = pd.Series(dtype=float)
    if permutation is not None:
        p_values.loc["gmom_permutation"] = permutation.p_value
    p_values.loc["gmom_bootstrap_leq0"] = bootstrap.p_value_positive
    adjusted = multiple_testing_adjustment(p_values)
    _save_table(tables_dir, "multiple_testing", adjusted)
    artifacts["multiple_testing"] = adjusted

    stability = coefficient_stability(gmom.coefficients)
    _save_table(tables_dir, "coefficient_stability", stability)
    artifacts["coefficient_stability"] = stability

    residuals = residual_diagnostics(
        gmom.predictions["prediction"], gmom.predictions["target_scaled_return"]
    )
    _save_table(tables_dir, "residual_diagnostics", residuals)
    artifacts["residual_diagnostics"] = residuals

    selected_candidate = max(
        network_bundle.network_features,
        key=lambda c: gmom.folds["alpha"].eq(c[0]).sum() + gmom.folds["beta"].eq(c[1]).sum(),
    )
    network_features = network_bundle.network_features[selected_candidate]
    target_long = (
        feature_set.target_scaled_return.rename_axis(index="date", columns="ticker")
        .stack()
        .rename("target")
    )
    univariate = univariate_feature_r2(network_features, target_long)
    _save_table(tables_dir, "univariate_feature_r2", univariate)

    # A learning curve usa apenas o último bloco de TREINO (nunca o teste OOS).
    last_train_end = pd.Timestamp(gmom.folds["train_end"].max())
    train_dates_mask = network_features.index.get_level_values("date") <= last_train_end
    target_train_mask = target_long.index.get_level_values("date") <= last_train_end
    learning = learning_curve(
        network_features[train_dates_mask],
        target_long[target_train_mask],
        min_samples=config.model.min_regression_samples,
    )
    _save_table(tables_dir, "learning_curve", learning)
    artifacts["learning_curve"] = learning

    # -------------------------------------------------------------- robustez
    decomposition = long_short_decomposition(gmom.predictions)
    _save_table(tables_dir, "long_short_decomposition_metrics", pd.DataFrame(
        {
            "long": performance_metrics(decomposition["long_return"]),
            "short": performance_metrics(decomposition["short_return"]),
        }
    ).T)
    artifacts["long_short"] = decomposition

    group_tables = {}
    for group in ("region", "exchange", "currency", "sector"):
        if group in universe.columns and universe[group].astype(str).str.len().gt(0).any():
            table = metrics_by_group(
                gmom.predictions, universe[group], group_name=group
            )
            group_tables[group] = table
            _save_table(tables_dir, f"metrics_by_{group}", table)
    artifacts["group_metrics"] = group_tables

    contribution = asset_contribution(gmom.predictions)
    _save_table(tables_dir, "asset_contribution", contribution)
    _save_table(tables_dir, "return_concentration", concentration_statistics(contribution))
    if gmom.portfolio is not None:
        _save_table(tables_dir, "risk_concentration", risk_concentration(gmom.portfolio.weights))
    artifacts["contribution"] = contribution

    _save_table(tables_dir, "remove_best_assets", remove_best_assets_test(gmom.predictions))
    _save_table(tables_dir, "remove_best_years", remove_best_years_test(strategy_net))
    _save_table(tables_dir, "start_date_sensitivity", start_date_sensitivity(strategy_net))
    market_returns = suite.results["long_only"].daily_returns["strategy_return"]
    _save_table(
        tables_dir,
        "volatility_regimes",
        volatility_regime_metrics(strategy_net, market_returns),
    )
    _save_table(tables_dir, "calendar_windows", calendar_window_metrics(strategy_net))

    feature_ablation = pd.DataFrame()
    edge_ablation = pd.DataFrame()
    lookback_ablation = pd.DataFrame()
    if options.profile != "smoke":
        feature_ablation = run_feature_ablation(
            feature_set,
            universe["region"],
            config.graph,
            config.model,
            network_bundle=network_bundle,
        )
        _save_table(tables_dir, "feature_ablation", feature_ablation)
        if options.run_edge_ablation_flag and options.profile == "full":
            labels = {"region": universe["region"].to_dict()}
            if "sector" in universe.columns:
                labels["sector"] = universe["sector"].to_dict()
            edge_ablation = run_edge_mask_ablation(
                feature_set,
                universe["region"],
                config.graph,
                config.model,
                network_bundle=network_bundle,
                group_labels=labels,
            )
            _save_table(tables_dir, "edge_ablation", edge_ablation)
        if options.run_lookback_ablation_flag and options.profile == "full":
            lookback_ablation = run_lookback_ablation(
                feature_set,
                universe["region"],
                config.graph,
                config.model,
            )
            _save_table(tables_dir, "lookback_ablation", lookback_ablation)
    artifacts["feature_ablation"] = feature_ablation
    artifacts["edge_ablation"] = edge_ablation
    artifacts["lookback_ablation"] = lookback_ablation

    # -------------------------------------------------------------- topologia
    snapshots = network_bundle.snapshots[selected_candidate]
    topology = topology_time_series(
        snapshots,
        edge_threshold=config.graph.edge_threshold,
        group_labels=universe["region"].to_dict(),
    )
    _save_table(tables_dir, "graph_topology", topology)
    artifacts["topology"] = topology
    artifacts["selected_candidate"] = selected_candidate

    # --------------------------------------------------------------- gráficos
    if options.make_plots:
        from . import plotting

        period = f"{strategy_net.index.min().date()} a {strategy_net.index.max().date()}"
        saver = plotting.FigureSaver(
            figures_dir,
            source="Yahoo Finance via yfinance; elaboração própria",
            period=period,
        )
        _make_all_plots(
            saver,
            config=config,
            artifacts=artifacts,
            universe=universe,
        )
        saver.write_index()
        artifacts["figures_dir"] = figures_dir

    # ------------------------------------------------- resultados canônicos
    data_stamp = None
    cache_dir = Path(config.data.cache_dir)
    if cache_dir.exists():
        stamp_files = sorted(cache_dir.glob("download_stamp_*.json"))
        if stamp_files:
            data_stamp = json.loads(stamp_files[-1].read_text(encoding="utf-8"))
    manifest = build_manifest(
        config=config,
        result=gmom,
        universe_hash=universe_fingerprint(universe),
        n_assets=int(universe.shape[0]),
        seed=seed,
        data_stamp=data_stamp,
        extras={
            "profile": options.profile,
            "breakeven_bps": float(breakeven) if np.isfinite(breakeven) else None,
            "net_sharpe": float(performance_metrics(strategy_net)["sharpe"]),
            "runtime_seconds": round(time.time() - started, 1),
        },
    )
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    save_results(gmom, output_dir, config_path=config.source_path, manifest_extras=manifest)

    # ------------------------------------------------ respostas do formulário
    facts = build_run_facts(
        universe,
        gmom.daily_returns,
        base_currency=config.data.base_currency,
        target_volatility=config.model.target_annual_volatility,
        graph_lookbacks=config.graph.lookbacks,
        rebalance_every=config.graph.rebalance_every,
        benchmark_names=tuple(external_returns.columns),
        net_sharpe=float(performance_metrics(strategy_net)["sharpe"]),
    )
    answers = form_answers(facts)
    (output_dir / "form_answers.md").write_text(
        answers_markdown(answers), encoding="utf-8"
    )
    artifacts["form_answers"] = answers
    artifacts["manifest"] = manifest

    LOGGER.info(
        "Pipeline concluído em %.1f min. Saídas em %s",
        (time.time() - started) / 60.0,
        output_dir,
    )
    return artifacts


def _make_all_plots(saver, *, config: AppConfig, artifacts: dict, universe: pd.DataFrame) -> None:
    from . import plotting

    suite = artifacts["benchmark_suite"]
    gmom: BacktestResult = artifacts["gmom"]
    daily = gmom.daily_returns
    net = daily["strategy_return"]
    gross = daily["scaled_gross_return"]

    plotting.plot_equity_curves(
        saver,
        {"GMOM líquida": net, "GMOM bruta": gross},
        name="01_equity_gross_net",
        title="Curva de capital — bruta vs líquida de custos",
        note="Retornos diários compostos; custos por bolsa (cenário "
        f"{config.costs.scenario}) sobre a variação efetiva dos pesos.",
    )
    benchmark_series = {"GMOM": net}
    for name in ("linreg", "macd", "long_only", "equal_weight"):
        if name in suite.results:
            benchmark_series[name] = suite.results[name].daily_returns["strategy_return"]
    external = artifacts.get("external_returns")
    if external is not None and not external.empty and "ACWI" in external.columns:
        benchmark_series["ACWI (proxy TR)"] = external["ACWI"].reindex(net.index)
    plotting.plot_equity_curves(
        saver,
        benchmark_series,
        name="02_equity_vs_benchmarks",
        title="Curva de capital — estratégia vs benchmarks (mesma moeda e período)",
        note="Benchmarks metodológicos no mesmo universo com a mesma Eq. (9); ACWI é proxy investível de retorno total.",
    )
    plotting.plot_drawdown(
        saver, net, name="03_drawdown", title="Drawdown da estratégia líquida",
        note="Queda em relação ao pico da curva de capital líquida.",
    )
    plotting.plot_annual_returns(
        saver, net, name="04_annual_returns", title="Retorno anual composto (líquido)",
        note="Anos-calendário; ano corrente parcial quando aplicável.",
    )
    plotting.plot_rolling_series(
        saver,
        {"GMOM": plotting.rolling_sharpe(net)},
        name="05_rolling_sharpe",
        title="Sharpe móvel (janela 252 pregões)",
        ylabel="Sharpe anualizado",
        note="Retornos líquidos; rf = 0 (declarada).",
        hline=0.0,
    )
    plotting.plot_rolling_series(
        saver,
        {"GMOM": plotting.rolling_volatility(net)},
        name="06_rolling_volatility",
        title="Volatilidade móvel anualizada (janela 63 pregões)",
        ylabel="Volatilidade a.a. (fração)",
        note=f"Alvo de volatilidade: {config.model.target_annual_volatility:.0%} a.a.",
        hline=config.model.target_annual_volatility,
    )
    market = suite.results["long_only"].daily_returns["strategy_return"]
    plotting.plot_rolling_series(
        saver,
        {"beta vs Long Only": plotting.rolling_beta(net, market)},
        name="07_rolling_beta",
        title="Beta móvel contra o Long Only do universo (janela 252)",
        ylabel="Beta",
        note="Long Only vol-scaled no mesmo universo (benchmark de mercado do artigo).",
        hline=0.0,
    )
    plotting.plot_rolling_series(
        saver,
        {"turnover (final)": daily["turnover_final"].rolling(63).mean()},
        name="08_turnover",
        title="Turnover médio móvel (63 pregões)",
        ylabel="Turnover diário (Σ|Δw|/N)",
        note="Inclui a variação da alavancagem de portfólio (Eq. 13 estendida).",
    )
    cost_columns = [
        c
        for c in (
            "cost_commission",
            "cost_exchange_fee",
            "cost_spread",
            "cost_regulatory",
            "cost_tax",
            "cost_fx",
            "cost_borrow",
            "cost_impact",
        )
        if c in daily.columns
    ]
    if cost_columns:
        plotting.plot_stacked_costs(
            saver,
            daily[cost_columns],
            name="09_cost_decomposition",
            title="Decomposição do custo anualizado por componente",
            note=f"Cenário {config.costs.scenario}; tributos oficiais vs estimativas em config/costs.csv.",
        )
    plotting.plot_sharpe_vs_cost(
        saver,
        artifacts["sharpe_vs_cost"],
        breakeven_bps=artifacts["breakeven_bps"],
        name="10_sharpe_vs_cost",
        title="Sharpe líquido × custo linear (Eq. 14) e break-even",
        note="Custo aplicado por unidade de turnover final; break-even zera o retorno médio.",
    )
    group_metrics = artifacts.get("group_metrics", {})
    for i, (group, table) in enumerate(group_metrics.items()):
        plotting.plot_bars(
            saver,
            table["sharpe"].sort_values(),
            name=f"1{1 + i}_sharpe_by_{group}",
            title=f"Sharpe bruto por {group}",
            xlabel=group,
            ylabel="Sharpe (a.a.)",
            note="Carteiras parciais para atribuição (não reescaladas).",
            horizontal=True,
        )
    decomposition = artifacts["long_short"]
    plotting.plot_equity_curves(
        saver,
        {
            "Perna comprada": decomposition["long_return"],
            "Perna vendida": decomposition["short_return"],
            "Total (bruto)": decomposition["total"],
        },
        name="15_long_short",
        title="Decomposição long vs short (contribuições brutas)",
        note="Long + short = retorno bruto do portfólio; divisão pelo nº total de ativos ativos.",
        log_scale=False,
    )
    contribution = artifacts["contribution"]
    plotting.plot_bars(
        saver,
        contribution["total_contribution"].head(20),
        name="16_asset_contribution",
        title="Contribuição acumulada por ativo (top 20)",
        xlabel="Ativo",
        ylabel="Contribuição acumulada (fração do PL)",
        note="Σ peso × retorno ÷ ativos ativos; sem reescala de portfólio.",
        horizontal=True,
    )
    if gmom.portfolio is not None:
        weights = gmom.portfolio.weights
        hhi = (weights.abs().div(weights.abs().sum(axis=1), axis=0) ** 2).sum(axis=1)
        plotting.plot_rolling_series(
            saver,
            {"HHI dos |pesos|": hhi.rolling(21).mean()},
            name="17_risk_concentration",
            title="Concentração de risco (HHI dos pesos, média móvel 21d)",
            ylabel="HHI",
            note="HHI = Σ share²; 1/HHI = nº efetivo de posições.",
        )
    plotting.plot_heatmap(
        saver,
        suite.correlation,
        name="18_strategy_correlation",
        title="Correlação entre estratégias (retornos líquidos diários)",
        note="Mesmo universo, período e custos.",
    )
    plotting.plot_heatmap(
        saver,
        suite.sign_agreement,
        name="19_sign_agreement",
        title="Sign agreement entre estratégias (fração de posições iguais)",
        note="Comparação por (data, ativo).",
        center_zero=False,
        cmap="viridis",
    )
    external_stats = artifacts.get("external_stats")
    if external_stats is not None and not external_stats.empty:
        heat = external_stats[["alpha_annual", "beta", "correlation", "information_ratio"]]
        plotting.plot_heatmap(
            saver,
            heat.astype(float),
            name="20_alpha_beta_external",
            title="Alpha (a.a.), beta, correlação e IR contra benchmarks externos",
            note="Estratégia em moeda-base vs benchmarks convertidos; proxies TR declarados.",
        )
    lookback_ablation = artifacts.get("lookback_ablation")
    if lookback_ablation is not None and not lookback_ablation.empty:
        plotting.plot_bars(
            saver,
            lookback_ablation["sharpe"],
            name="21_lookback_sensitivity",
            title="Sharpe por lookback individual do grafo vs ensemble",
            xlabel="Configuração do grafo",
            ylabel="Sharpe líquido (a.a.)",
            note="Seção 5.3 do artigo: grafos individuais reaprendidos por lookback.",
        )
    learning = artifacts.get("learning_curve")
    if learning is not None and not learning.empty:
        plotting.plot_learning_curves(
            saver,
            learning,
            name="22_learning_curves",
            title="Curva de aprendizado temporal (prefixos do treino)",
            note="Validação = bloco final fixo do treino; sem embaralhamento temporal.",
        )
    folds = gmom.folds
    if not folds.empty:
        fold_frame = folds.set_index("fold")[["train_sharpe", "validation_sharpe"]].copy()
        fold_frame["test_sharpe"] = artifacts["fold_metrics"]["sharpe"]
        plotting.plot_bars(
            saver,
            fold_frame.stack().rename("sharpe"),
            name="23_train_val_test",
            title="Sharpe de treino vs validação vs teste, por fold",
            xlabel="(fold, etapa)",
            ylabel="Sharpe (a.a.)",
            note="Treino/validação usam o sinal bruto; teste usa a estratégia líquida.",
        )
    bootstrap = artifacts["bootstrap"]
    plotting.plot_histogram(
        saver,
        bootstrap.samples,
        observed=bootstrap.observed_sharpe,
        name="24_bootstrap_sharpe",
        title="Distribuição bootstrap (blocos circulares por data) do Sharpe",
        xlabel="Sharpe anualizado",
        note=f"IC 95%: [{bootstrap.ci_low:.2f}, {bootstrap.ci_high:.2f}]; "
        f"P(Sharpe<=0) = {bootstrap.p_value_positive:.3f}; blocos de "
        f"{config.validation.bootstrap_block_days} dias.",
    )
    permutation = artifacts.get("permutation")
    if permutation is not None:
        plotting.plot_histogram(
            saver,
            permutation.permuted_sharpes,
            observed=permutation.observed_sharpe,
            name="25_permutation_test",
            title="Teste de permutação por deslocamento circular",
            xlabel="Sharpe anualizado sob H0 (sem alinhamento sinal-retorno)",
            note=f"p-valor = {permutation.p_value:.4f}.",
        )
    plotting.plot_coefficient_paths(
        saver,
        gmom.coefficients,
        name="26_coefficient_stability",
        title="Coeficientes da OLS por fold",
        note="Estabilidade entre folds; erros-padrão clusterizados por data nas tabelas.",
    )
    residual = gmom.predictions["target_scaled_return"] - gmom.predictions["prediction"]
    lags = {}
    for lag in range(1, 11):
        shifted = residual.groupby(level="ticker").shift(lag)
        lags[lag] = float(residual.corr(shifted))
    plotting.plot_autocorrelation_bars(
        saver,
        pd.Series(lags),
        name="27_residual_autocorrelation",
        title="Autocorrelação média dos resíduos fora da amostra",
        note="Correlação entre resíduo e resíduo defasado por ativo, agregada.",
    )
    snapshots = artifacts["network_bundle"].snapshots[artifacts["selected_candidate"]]
    labels = universe["region"].to_dict()
    normal_snapshot = snapshots[len(snapshots) // 2]
    plotting.plot_graph_snapshot(
        saver,
        normal_snapshot,
        labels,
        edge_threshold=config.graph.edge_threshold,
        name="28_graph_normal",
        title=f"Grafo em período típico ({normal_snapshot.date.date()})",
        note="Layout circular por região; espessura ∝ peso da aresta; threshold relativo "
        f"{config.graph.edge_threshold:g}.",
    )
    crisis = [s for s in snapshots if pd.Timestamp("2020-02-01") <= s.date <= pd.Timestamp("2020-05-31")]
    crisis_snapshot = crisis[0] if crisis else snapshots[-1]
    plotting.plot_graph_snapshot(
        saver,
        crisis_snapshot,
        labels,
        edge_threshold=config.graph.edge_threshold,
        name="29_graph_crisis",
        title=f"Grafo em período de crise ({crisis_snapshot.date.date()})",
        note="Mesma construção do gráfico anterior, em janela de estresse (COVID-19 se disponível).",
    )
    plotting.plot_topology_panels(
        saver,
        artifacts["topology"],
        name="30_topology_series",
        title="Topologia do grafo ao longo do tempo",
        note="Esparsidade, grau, clustering, community ratio (região) e Jaccard entre snapshots consecutivos.",
    )
    threshold_frame = artifacts.get("threshold_study")
    if threshold_frame is not None and not threshold_frame.empty:
        plotting.plot_bars(
            saver,
            threshold_frame["sharpe"],
            name="31_signal_threshold",
            title="Sharpe por variante de sinal (sign, thresholds, contínuo)",
            xlabel="Variante",
            ylabel="Sharpe líquido (a.a.)",
            note="Thresholds em quantis de |previsão|; variantes sem reajuste da regressão.",
        )
    regression_variants = artifacts.get("regression_variants")
    if regression_variants is not None and not regression_variants.empty:
        plotting.plot_bars(
            saver,
            regression_variants["sharpe"],
            name="32_regression_variants",
            title="OLS vs Ridge vs Lasso vs Elastic Net (robustez)",
            xlabel="Método",
            ylabel="Sharpe líquido (a.a.)",
            note="Modelos auxiliares de robustez; a estratégia principal permanece OLS.",
        )
