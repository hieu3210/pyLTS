"""Minimal example showing how to load a dataset and produce rolling-origin folds.

This script is runnable directly. Ensure the repository root is on `sys.path`
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


def main():
    data, lb, ub = read_txt_dataset('LTS/Datasets/alabama.txt')
    print(f"Loaded series length: {len(data)}, lb={lb}, ub={ub}")

    initial_train = max(10, int(len(data) * 0.5))
    horizon = 5
    n_splits = 5
    print(f"Creating rolling-origin splits: initial_train={initial_train}, horizon={horizon}, n_splits={n_splits}")
    for i, (train, val) in enumerate(rolling_origin_splits(data, initial_train, horizon, n_splits)):
        print(f"Fold {i+1}: train={len(train)} points, val={len(val)} points")


if __name__ == '__main__':
    main()
