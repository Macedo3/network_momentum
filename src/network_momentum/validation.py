from __future__ import annotations

"""Diagnósticos de overfitting/underfitting e inferência estatística.

Todos os procedimentos estocásticos recebem ``seed`` e são reprodutíveis. As
interpretações em português de cada diagnóstico estão no notebook (Seção 8).
"""

from dataclasses import dataclass
from itertools import combinations
import math

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252.0
EULER_GAMMA = 0.5772156649015329


def _annualized_sharpe(returns: np.ndarray) -> float:
    sigma = returns.std(ddof=0)
    if sigma <= 0:
        return np.nan
    return float(returns.mean() / sigma * np.sqrt(TRADING_DAYS))


@dataclass(frozen=True)
class BootstrapResult:
    observed_sharpe: float
    samples: np.ndarray
    ci_low: float
    ci_high: float
    p_value_positive: float  # fração de amostras com Sharpe <= 0


def block_bootstrap_sharpe(
    returns: pd.Series,
    *,
    n_samples: int = 2000,
    block_days: int = 21,
    seed: int = 42,
    ci: float = 0.95,
) -> BootstrapResult:
    """Bootstrap circular em blocos da série DIÁRIA de retornos do portfólio.

    Reamostrar a série agregada por data preserva integralmente a dependência
    transversal entre ativos (todos os ativos do mesmo dia permanecem juntos);
    os blocos preservam a autocorrelação de curto prazo.
    """
    values = returns.dropna().to_numpy(dtype=float)
    if values.size < block_days * 2:
        raise ValueError("Série curta demais para o bootstrap em blocos.")
    rng = np.random.default_rng(seed)
    n_days = values.size
    n_blocks = math.ceil(n_days / block_days)
    starts = rng.integers(0, n_days, size=(n_samples, n_blocks))
    offsets = np.arange(block_days)
    indices = (starts[:, :, None] + offsets[None, None, :]) % n_days
    indices = indices.reshape(n_samples, -1)[:, :n_days]
    resampled = values[indices]
    means = resampled.mean(axis=1)
    sigmas = resampled.std(axis=1, ddof=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        sharpes = means / sigmas * np.sqrt(TRADING_DAYS)
    sharpes = sharpes[np.isfinite(sharpes)]
    lower = float(np.percentile(sharpes, (1 - ci) / 2 * 100))
    upper = float(np.percentile(sharpes, (1 + ci) / 2 * 100))
    return BootstrapResult(
        observed_sharpe=_annualized_sharpe(values),
        samples=sharpes,
        ci_low=lower,
        ci_high=upper,
        p_value_positive=float((sharpes <= 0).mean()),
    )


@dataclass(frozen=True)
class PermutationResult:
    observed_sharpe: float
    permuted_sharpes: np.ndarray
    p_value: float


def circular_shift_permutation_test(
    weights: pd.DataFrame,
    forward_returns: pd.DataFrame,
    *,
    n_samples: int = 500,
    min_shift_days: int = 63,
    seed: int = 42,
) -> PermutationResult:
    """Teste de permutação por deslocamento circular: desloca a matriz de pesos em
    k dias (k >= min_shift) e recalcula o Sharpe. Preserva a autocorrelação de
    pesos e de retornos, mas destrói o alinhamento sinal→retorno (H0: sem skill).
    p-valor = fração de deslocamentos com Sharpe >= observado.
    """
    common_columns = weights.columns.intersection(forward_returns.columns)
    weight_values = weights.loc[:, common_columns].fillna(0.0).to_numpy(dtype=float)
    return_values = (
        forward_returns.loc[:, common_columns]
        .reindex(weights.index)
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    active = (weight_values != 0).sum(axis=1)
    active = np.maximum(active, 1)
    n_days = weight_values.shape[0]
    if n_days <= 2 * min_shift_days:
        raise ValueError("Série curta demais para o teste de permutação.")
    observed = _annualized_sharpe((weight_values * return_values).sum(axis=1) / active)

    rng = np.random.default_rng(seed)
    shifts = rng.integers(min_shift_days, n_days - min_shift_days, size=n_samples)
    permuted = np.empty(n_samples)
    for i, shift in enumerate(shifts):
        rolled = np.roll(weight_values, int(shift), axis=0)
        rolled_active = np.maximum((rolled != 0).sum(axis=1), 1)
        daily = (rolled * return_values).sum(axis=1) / rolled_active
        permuted[i] = _annualized_sharpe(daily)
    permuted = permuted[np.isfinite(permuted)]
    p_value = float((permuted >= observed).mean()) if permuted.size else np.nan
    return PermutationResult(
        observed_sharpe=observed,
        permuted_sharpes=permuted,
        p_value=p_value,
    )


def deflated_sharpe_ratio(
    returns: pd.Series,
    *,
    n_trials: int,
    trial_sharpe_variance: float | None = None,
) -> pd.Series:
    """Deflated Sharpe Ratio de Bailey & López de Prado (2014).

    ``n_trials`` deve refletir o número de configurações efetivamente testadas
    (grid de hiperparâmetros × variantes) — subestimá-lo infla o DSR. Se a
    variância dos Sharpes dos trials não for conhecida, usa-se a variância
    implícita de estimadores de Sharpe sob H0 (1/T em unidades diárias), o que é
    conservador quando os trials são correlacionados.
    """
    values = returns.dropna().to_numpy(dtype=float)
    n_days = values.size
    if n_days < 30:
        raise ValueError("Série curta demais para o DSR.")
    sigma = values.std(ddof=0)
    sharpe_daily = values.mean() / sigma if sigma > 0 else np.nan
    centered = values - values.mean()
    skewness = (centered**3).mean() / sigma**3 if sigma > 0 else 0.0
    kurtosis = (centered**4).mean() / sigma**4 if sigma > 0 else 3.0
    if trial_sharpe_variance is None:
        trial_sharpe_variance = 1.0 / n_days
    n_trials = max(int(n_trials), 1)
    if n_trials > 1:
        z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
        z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
        expected_max = math.sqrt(trial_sharpe_variance) * (
            (1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2
        )
    else:
        expected_max = 0.0
    denominator = math.sqrt(
        max(1.0 - skewness * sharpe_daily + (kurtosis - 1.0) / 4.0 * sharpe_daily**2, 1e-12)
    )
    dsr = stats.norm.cdf((sharpe_daily - expected_max) * math.sqrt(n_days - 1) / denominator)
    return pd.Series(
        {
            "sharpe_annual": sharpe_daily * np.sqrt(TRADING_DAYS) if np.isfinite(sharpe_daily) else np.nan,
            "sharpe_daily": sharpe_daily,
            "expected_max_sharpe_daily_h0": expected_max,
            "n_trials": float(n_trials),
            "skewness": float(skewness),
            "kurtosis": float(kurtosis),
            "deflated_sharpe_probability": float(dsr),
        }
    )


@dataclass(frozen=True)
class PBOResult:
    pbo: float
    logits: np.ndarray
    n_configurations: int
    n_splits: int
    applicable: bool
    note: str


def probability_of_backtest_overfitting(
    returns_by_configuration: pd.DataFrame,
    *,
    n_blocks: int = 8,
    max_splits: int = 200,
    seed: int = 42,
) -> PBOResult:
    """PBO via CSCV (Bailey et al., 2017): divide o período em blocos, combina
    metade como IS e metade como OOS, escolhe a configuração vencedora IS e mede
    seu rank OOS. PBO = fração de splits em que o vencedor IS fica abaixo da
    mediana OOS. Requer >= 2 configurações; caso contrário, não aplicável.
    """
    frame = returns_by_configuration.dropna(how="all").fillna(0.0)
    n_configurations = frame.shape[1]
    if n_configurations < 2:
        return PBOResult(
            pbo=float("nan"),
            logits=np.array([]),
            n_configurations=n_configurations,
            n_splits=0,
            applicable=False,
            note="PBO/CSCV requer pelo menos duas configurações com séries de retorno.",
        )
    if n_blocks % 2 != 0:
        n_blocks += 1
    blocks = np.array_split(np.arange(len(frame)), n_blocks)
    all_splits = list(combinations(range(n_blocks), n_blocks // 2))
    if len(all_splits) > max_splits:
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(all_splits), size=max_splits, replace=False)
        all_splits = [all_splits[i] for i in chosen]
    values = frame.to_numpy(dtype=float)
    logits: list[float] = []
    for in_sample_blocks in all_splits:
        in_rows = np.concatenate([blocks[b] for b in in_sample_blocks])
        out_rows = np.concatenate(
            [blocks[b] for b in range(n_blocks) if b not in in_sample_blocks]
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            is_sharpe = values[in_rows].mean(axis=0) / values[in_rows].std(axis=0, ddof=0)
            oos_sharpe = values[out_rows].mean(axis=0) / values[out_rows].std(axis=0, ddof=0)
        winner = int(np.nanargmax(is_sharpe))
        oos_rank = stats.rankdata(oos_sharpe)[winner]
        relative = oos_rank / (n_configurations + 1.0)
        logits.append(math.log(relative / (1.0 - relative)))
    logits_array = np.asarray(logits)
    return PBOResult(
        pbo=float((logits_array < 0).mean()),
        logits=logits_array,
        n_configurations=n_configurations,
        n_splits=len(all_splits),
        applicable=True,
        note="",
    )


def learning_curve(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    fractions: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 1.0),
    validation_fraction: float = 0.1,
    min_samples: int = 100,
    fit_kwargs: dict | None = None,
) -> pd.DataFrame:
    """Curva de aprendizado temporal: treina no prefixo cronológico de cada fração
    e avalia no bloco final (validação fixa). Treino e validação nunca se cruzam."""
    from .model import fit_linear_model

    fit_kwargs = fit_kwargs or {}
    joined = features.join(target.rename("target"), how="inner").dropna()
    dates = joined.index.get_level_values("date").unique().sort_values()
    n_validation = max(int(len(dates) * validation_fraction), 1)
    validation_dates = dates[-n_validation:]
    train_dates = dates[:-n_validation]
    validation_mask = joined.index.get_level_values("date").isin(validation_dates)
    validation_set = joined[validation_mask]
    rows = []
    for fraction in fractions:
        cut = max(int(len(train_dates) * fraction), 1)
        selected_dates = train_dates[:cut]
        train_mask = joined.index.get_level_values("date").isin(selected_dates)
        train_set = joined[train_mask]
        if len(train_set) < min_samples:
            continue
        model = fit_linear_model(
            train_set.loc[:, features.columns],
            train_set["target"],
            min_samples=min_samples,
            **fit_kwargs,
        )
        for label, subset in (("train", train_set), ("validation", validation_set)):
            prediction = model.predict(subset.loc[:, features.columns])
            residual = subset["target"] - prediction
            total_variance = float(subset["target"].var(ddof=0))
            r2 = 1.0 - float(residual.var(ddof=0)) / total_variance if total_variance > 0 else np.nan
            contribution = (
                np.sign(prediction) * subset["target"]
            ).groupby(level="date").mean()
            rows.append(
                {
                    "fraction": fraction,
                    "train_days": len(selected_dates),
                    "split": label,
                    "r_squared": r2,
                    "signal_sharpe": _annualized_sharpe(contribution.to_numpy()),
                }
            )
    return pd.DataFrame(rows)


def coefficient_stability(coefficients: pd.DataFrame) -> pd.DataFrame:
    """Estabilidade dos coeficientes entre folds: média, desvio, consistência de
    sinal e razão |média|/desvio (uma estatística t entre folds)."""
    grouped = coefficients[coefficients["feature"] != "intercept"].groupby("feature")
    rows = []
    for feature, block in grouped:
        values = block["coefficient"].to_numpy(dtype=float)
        mean = values.mean()
        std = values.std(ddof=0)
        rows.append(
            {
                "feature": feature,
                "mean": mean,
                "std": std,
                "sign_consistency": float((np.sign(values) == np.sign(mean)).mean()),
                "t_like_ratio": mean / std if std > 0 else np.inf,
                "mean_standard_error": float(
                    block["standard_error"].astype(float).mean()
                )
                if "standard_error" in block
                else np.nan,
                "folds": len(values),
            }
        )
    return pd.DataFrame(rows).set_index("feature")


def residual_diagnostics(
    prediction: pd.Series,
    target: pd.Series,
    *,
    max_lag: int = 10,
) -> pd.Series:
    """Resíduos da regressão: média, R², autocorrelação média por ativo e
    estatística de Ljung-Box (agregada) com p-valor."""
    aligned = pd.concat(
        [prediction.rename("prediction"), target.rename("target")], axis=1
    ).dropna()
    residual = aligned["target"] - aligned["prediction"]
    total_variance = float(aligned["target"].var(ddof=0))
    r2 = 1.0 - float(residual.var(ddof=0)) / total_variance if total_variance > 0 else np.nan

    autocorrelations = []
    q_statistics = []
    if isinstance(residual.index, pd.MultiIndex):
        for _, series in residual.groupby(level="ticker"):
            values = series.droplevel("ticker").sort_index().to_numpy(dtype=float)
            n = len(values)
            if n < max_lag * 3:
                continue
            centered = values - values.mean()
            denominator = float((centered**2).sum())
            if denominator <= 0:
                continue
            rhos = [
                float((centered[k:] * centered[:-k]).sum() / denominator)
                for k in range(1, max_lag + 1)
            ]
            autocorrelations.append(rhos[0])
            q = n * (n + 2) * sum(r**2 / (n - k) for k, r in enumerate(rhos, start=1))
            q_statistics.append(q)
    mean_lag1 = float(np.mean(autocorrelations)) if autocorrelations else np.nan
    mean_q = float(np.mean(q_statistics)) if q_statistics else np.nan
    p_value = float(stats.chi2.sf(mean_q, df=max_lag)) if np.isfinite(mean_q) else np.nan
    return pd.Series(
        {
            "residual_mean": float(residual.mean()),
            "residual_std": float(residual.std(ddof=0)),
            "r_squared": r2,
            "mean_lag1_autocorrelation": mean_lag1,
            "mean_ljung_box_q": mean_q,
            "ljung_box_p_value": p_value,
            "assets_tested": float(len(q_statistics)),
        }
    )


def univariate_feature_r2(
    features: pd.DataFrame,
    target: pd.Series,
) -> pd.Series:
    """R² univariado de cada feature contra o alvo (capacidade explicativa isolada)."""
    joined = features.join(target.rename("target"), how="inner").dropna()
    output = {}
    y = joined["target"]
    for column in features.columns:
        correlation = joined[column].corr(y)
        output[str(column)] = float(correlation**2) if np.isfinite(correlation) else np.nan
    return pd.Series(output, name="univariate_r2")


def multiple_testing_adjustment(p_values: pd.Series) -> pd.DataFrame:
    """Correções de Bonferroni e Benjamini-Hochberg para a família de testes."""
    clean = p_values.dropna()
    n = len(clean)
    bonferroni = (clean * n).clip(upper=1.0)
    order = clean.sort_values()
    ranks = pd.Series(np.arange(1, n + 1), index=order.index)
    bh_raw = (order * n / ranks).iloc[::-1].cummin().iloc[::-1]
    benjamini_hochberg = bh_raw.reindex(clean.index).clip(upper=1.0)
    return pd.DataFrame(
        {
            "p_value": clean,
            "bonferroni": bonferroni,
            "benjamini_hochberg": benjamini_hochberg,
        }
    )
