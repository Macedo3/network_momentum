import pandas as pd

from network_momentum.splits import expanding_windows


def test_expanding_non_overlapping_windows() -> None:
    dates = pd.bdate_range("2005-01-03", "2022-12-30")
    windows = expanding_windows(dates, initial_train_years=10, test_years=5)
    assert len(windows) == 2
    assert windows[0].train_start == pd.Timestamp("2005-01-03")
    assert windows[0].test_start.year == 2015
    assert windows[1].train_start == windows[0].train_start
    assert windows[1].train_end > windows[0].train_end
    assert windows[1].test_start > windows[0].test_end


def test_embargo_removes_final_training_days() -> None:
    dates = pd.bdate_range("2005-01-03", "2022-12-30")
    plain = expanding_windows(dates, initial_train_years=10, test_years=5, embargo_days=0)
    embargoed = expanding_windows(dates, initial_train_years=10, test_years=5, embargo_days=5)
    for window_plain, window_embargo in zip(plain, embargoed):
        assert window_embargo.test_start == window_plain.test_start
        assert window_embargo.train_end < window_plain.train_end
        gap = dates[(dates > window_embargo.train_end) & (dates < window_embargo.test_start)]
        assert len(gap) == 5

