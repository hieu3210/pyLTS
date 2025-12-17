from typing import Iterator, Tuple, List


def rolling_origin_splits(series: List[float], initial_train: int, horizon: int, n_splits: int) -> Iterator[Tuple[List[float], List[float]]]:
    """Yield (train, validation) pairs using expanding-window (rolling-origin).

    series: full time series (list)
    initial_train: size of initial training window
    horizon: validation horizon (forecast length)
    n_splits: number of folds (will slide the origin forward)
    """
    N = len(series)
    if initial_train + horizon > N:
        raise ValueError("initial_train + horizon must be <= len(series)")

    max_start = N - horizon
    step = max(1, (max_start - initial_train) // max(1, n_splits - 1))
    start = initial_train
    for i in range(n_splits):
        train_end = initial_train + i * step
        if train_end >= max_start:
            train_end = max_start
        train = series[:train_end]
        val = series[train_end:train_end + horizon]
        if len(val) < horizon:
            break
        yield train, val
