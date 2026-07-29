import pandas as pd

from network_momentum.metrics import maximum_drawdown


def test_drawdown_duration_is_longest_underwater_run() -> None:
    returns = pd.Series([0.10, -0.05, -0.02, 0.10, -0.01, 0.02])
    _, duration = maximum_drawdown(returns)
    assert duration == 2 / 6
