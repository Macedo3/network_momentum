from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OLSModel:
    feature_names: tuple[str, ...]
    intercept: float
    coefficients: np.ndarray
    rank: int
    sample_count: int
    method: str = "ols"
    standard_errors: np.ndarray | None = field(default=None, compare=False)

    def predict(self, features: pd.DataFrame) -> pd.Series:
        values = features.loc[:, self.feature_names].to_numpy(dtype=float)
        prediction = self.intercept + values.dot(self.coefficients)
        return pd.Series(prediction, index=features.index, name="prediction")

    def coefficient_series(self) -> pd.Series:
        return pd.Series(self.coefficients, index=self.feature_names, name="coefficient")

    def standard_error_series(self) -> pd.Series | None:
        if self.standard_errors is None:
            return None
        return pd.Series(
            self.standard_errors, index=self.feature_names, name="standard_error"
        )


def _clustered_standard_errors(
    design: np.ndarray,
    residuals: np.ndarray,
    cluster_codes: np.ndarray,
) -> np.ndarray:
    """Erro-padrão sanduíche clusterizado (por data). Necessário porque a OLS pooled
    tem forte correlação transversal dentro de cada data; o erro-padrão clássico
    superestima gravemente a significância."""
    gram = design.T @ design
    gram_inverse = np.linalg.pinv(gram)
    n_clusters = int(cluster_codes.max()) + 1 if cluster_codes.size else 0
    meat = np.zeros_like(gram)
    weighted = design * residuals[:, None]
    for code in range(n_clusters):
        rows = weighted[cluster_codes == code]
        if rows.size == 0:
            continue
        score = rows.sum(axis=0)
        meat += np.outer(score, score)
    covariance = gram_inverse @ meat @ gram_inverse
    if n_clusters > 1:
        covariance *= n_clusters / (n_clusters - 1)
    return np.sqrt(np.maximum(np.diag(covariance), 0.0))


def _coordinate_descent_elastic_net(
    x: np.ndarray,
    y: np.ndarray,
    *,
    lasso_lambda: float,
    l1_ratio: float,
    max_iterations: int = 1000,
    tolerance: float = 1e-8,
) -> np.ndarray:
    """Elastic Net determinístico por coordinate descent em dados padronizados.

    Minimiza (1/2n)||y − Xw||² + λ(ρ||w||₁ + (1−ρ)/2 ||w||²), colunas de X com
    variância 1. Sem sklearn para manter as dependências mínimas.
    """
    n_samples, n_features = x.shape
    weights = np.zeros(n_features)
    residual = y.copy()
    l1_penalty = lasso_lambda * l1_ratio
    l2_penalty = lasso_lambda * (1.0 - l1_ratio)
    for _ in range(max_iterations):
        max_delta = 0.0
        for j in range(n_features):
            weight_old = weights[j]
            rho = (x[:, j] @ residual) / n_samples + weight_old
            weight_new = np.sign(rho) * max(abs(rho) - l1_penalty, 0.0) / (1.0 + l2_penalty)
            if weight_new != weight_old:
                residual += x[:, j] * (weight_old - weight_new)
                weights[j] = weight_new
                max_delta = max(max_delta, abs(weight_new - weight_old))
        if max_delta < tolerance:
            break
    return weights


def fit_linear_model(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    min_samples: int,
    method: str = "ols",
    ridge_lambda: float = 1e-4,
    lasso_lambda: float = 1e-5,
    elastic_net_l1_ratio: float = 0.5,
    compute_clustered_errors: bool = False,
) -> OLSModel:
    """Ajusta a Eq. (8) do artigo (OLS pooled) ou variantes regularizadas.

    Ridge/Lasso/ElasticNet existem como testes de robustez contra multicolinearidade
    das oito network features; a estratégia principal continua sendo a OLS.
    """
    feature_names = tuple(str(column) for column in features.columns)
    joined = features.join(target.rename("target"), how="inner").dropna()
    if len(joined) < min_samples:
        raise ValueError(
            f"A regressão recebeu {len(joined)} amostras; mínimo: {min_samples}."
        )
    values = joined.loc[:, feature_names].to_numpy(dtype=float)
    y = joined["target"].to_numpy(dtype=float)
    n_samples = len(y)

    if method == "ols":
        design = np.column_stack([np.ones(n_samples), values])
        solution, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
        intercept = float(solution[0])
        coefficients = solution[1:]
        standard_errors = None
        if compute_clustered_errors and isinstance(joined.index, pd.MultiIndex):
            residuals = y - design @ solution
            dates = joined.index.get_level_values("date")
            cluster_codes = pd.factorize(dates)[0]
            all_errors = _clustered_standard_errors(design, residuals, cluster_codes)
            standard_errors = all_errors[1:]
        return OLSModel(
            feature_names=feature_names,
            intercept=intercept,
            coefficients=coefficients,
            rank=int(rank),
            sample_count=n_samples,
            method="ols",
            standard_errors=standard_errors,
        )

    # Métodos regularizados: centrar y, padronizar X, não penalizar o intercepto.
    x_mean = values.mean(axis=0)
    x_std = values.std(axis=0, ddof=0)
    x_std[x_std == 0] = 1.0
    x_standardized = (values - x_mean) / x_std
    y_mean = float(y.mean())
    y_centered = y - y_mean

    if method == "ridge":
        gram = x_standardized.T @ x_standardized / n_samples
        regularized = gram + ridge_lambda * np.eye(gram.shape[0])
        cross = x_standardized.T @ y_centered / n_samples
        weights_standardized = np.linalg.solve(regularized, cross)
    elif method in ("lasso", "elastic_net"):
        l1_ratio = 1.0 if method == "lasso" else elastic_net_l1_ratio
        weights_standardized = _coordinate_descent_elastic_net(
            x_standardized,
            y_centered,
            lasso_lambda=lasso_lambda,
            l1_ratio=l1_ratio,
        )
    else:
        raise ValueError(f"Método desconhecido: {method}")

    coefficients = weights_standardized / x_std
    intercept = y_mean - float(x_mean @ coefficients)
    return OLSModel(
        feature_names=feature_names,
        intercept=intercept,
        coefficients=coefficients,
        rank=int(np.linalg.matrix_rank(x_standardized)),
        sample_count=n_samples,
        method=method,
        standard_errors=None,
    )


def fit_ols(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    min_samples: int,
) -> OLSModel:
    """Compatibilidade retroativa com a API original."""
    return fit_linear_model(features, target, min_samples=min_samples, method="ols")
