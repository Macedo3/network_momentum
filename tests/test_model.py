import numpy as np
import pandas as pd
import pytest

from network_momentum.model import fit_linear_model


def _panel(seed: int = 5, n_dates: int = 120, n_assets: int = 6):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n_dates)
    index = pd.MultiIndex.from_product([dates, [f"A{i}" for i in range(n_assets)]],
                                       names=["date", "ticker"])
    x1 = rng.normal(size=len(index))
    x2 = rng.normal(size=len(index))
    noise = rng.normal(scale=0.5, size=len(index))
    features = pd.DataFrame({"f1": x1, "f2": x2, "f_irrelevante": rng.normal(size=len(index))},
                            index=index)
    target = pd.Series(0.8 * x1 - 0.3 * x2 + noise, index=index, name="target")
    return features, target


def test_ols_matches_lstsq_and_clustered_errors() -> None:
    features, target = _panel()
    model = fit_linear_model(features, target, min_samples=50, method="ols",
                             compute_clustered_errors=True)
    design = np.column_stack([np.ones(len(features)), features.to_numpy()])
    expected, *_ = np.linalg.lstsq(design, target.to_numpy(), rcond=None)
    assert model.intercept == pytest.approx(expected[0], abs=1e-10)
    np.testing.assert_allclose(model.coefficients, expected[1:], atol=1e-10)
    errors = model.standard_error_series()
    assert errors is not None
    assert (errors > 0).all()
    # coeficiente relevante deve ser significativo; irrelevante, não
    t_stats = model.coefficient_series() / errors
    assert abs(t_stats["f1"]) > 3
    assert abs(t_stats["f_irrelevante"]) < 3


def test_ridge_matches_closed_form() -> None:
    features, target = _panel()
    lam = 0.5
    model = fit_linear_model(features, target, min_samples=50, method="ridge",
                             ridge_lambda=lam)
    x = features.to_numpy()
    y = target.to_numpy()
    x_mean, x_std = x.mean(axis=0), x.std(axis=0, ddof=0)
    xs = (x - x_mean) / x_std
    yc = y - y.mean()
    n = len(y)
    w = np.linalg.solve(xs.T @ xs / n + lam * np.eye(x.shape[1]), xs.T @ yc / n)
    np.testing.assert_allclose(model.coefficients, w / x_std, rtol=1e-8)


def test_lasso_zeroes_irrelevant_feature() -> None:
    features, target = _panel(n_dates=200)
    model = fit_linear_model(features, target, min_samples=50, method="lasso",
                             lasso_lambda=0.05)
    coefficients = model.coefficient_series()
    assert coefficients["f_irrelevante"] == pytest.approx(0.0, abs=1e-6)
    assert abs(coefficients["f1"]) > 0.5


def test_elastic_net_between_ridge_and_lasso() -> None:
    features, target = _panel(n_dates=200)
    enet = fit_linear_model(features, target, min_samples=50, method="elastic_net",
                            lasso_lambda=0.05, elastic_net_l1_ratio=0.5)
    assert enet.method == "elastic_net"
    assert np.isfinite(enet.coefficients).all()


def test_min_samples_enforced() -> None:
    features, target = _panel(n_dates=10, n_assets=2)
    with pytest.raises(ValueError, match="amostras"):
        fit_linear_model(features, target, min_samples=10_000, method="ols")
