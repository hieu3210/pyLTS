"""Evaluation harness: run rolling-origin CV on multiple models and report metrics.

This script is runnable directly; ensure repository root is on `sys.path`
so the `NEW` package can be imported when executed as a script.
"""
import os
import sys

# Ensure repo root is on sys.path so `import NEW` works when running this file
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from NEW.src.data_loader import read_txt_dataset
from NEW.src.split import rolling_origin_splits
from NEW.src.models import HAWrapLTS, ARIMAModel, ETSModel, LagMLModel
import math
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def mae(a, b):
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


def mse(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b)) / len(a)


def evaluate_model_on_splits(model, series, initial_train, horizon, n_splits):
    maes = []
    mses = []
    for train, val in rolling_origin_splits(series, initial_train, horizon, n_splits):
        model.fit(train)
        preds = model.predict(len(val))
        maes.append(mae(val, preds))
        mses.append(mse(val, preds))
    return maes, mses


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    data, lb, ub = read_txt_dataset('LTS/Datasets/alabama.txt')
    horizon = 5
    # Choose initial_train as 50% or at least 10, but ensure initial_train + horizon <= len(data)
    initial_train = max(10, int(len(data) * 0.5))
    if initial_train + horizon > len(data):
        # reduce initial_train to allow at least one fold
        initial_train = max(1, len(data) - horizon)
    if initial_train <= 0:
        print(f"Series too short (N={len(data)}) for horizon={horizon}. Cannot evaluate.")
        return
    # compute possible number of splits given data length
    max_start = len(data) - horizon
    possible_splits = max(1, (max_start - initial_train) + 1)
    n_splits = min(5, possible_splits)

    models = [
        ('HA-LTS', HAWrapLTS()),
        ('ARIMA(1,0,0)', ARIMAModel((1,0,0))),
        ('ETS', ETSModel(seasonal=None, seasonal_periods=None)),
        ('Lag-RF', LagMLModel(lags=5)),
    ]

    results = {}
    for name, m in models:
        try:
            maes, mses = evaluate_model_on_splits(m, data, initial_train, horizon, n_splits)
            if len(maes) == 0:
                print(f"{name} failed: no evaluation folds were produced (series too short)")
                continue
            results[name] = (maes, mses)
            print(f"{name}: MAE mean={sum(maes)/len(maes):.4f}, MSE mean={sum(mses)/len(mses):.4f}")
        except Exception as e:
            print(f"{name} failed: {e}")

    if plt and results:
        plt.figure(figsize=(8, 4))
        for name, (maes, mses) in results.items():
            plt.plot(maes, label=name)
        plt.title('MAE per fold')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    main()
