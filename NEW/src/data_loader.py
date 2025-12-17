import os
from typing import List, Tuple


def read_txt_dataset(path: str) -> Tuple[List[float], float, float]:
    """Read dataset in same three-line format used by LTS/Datasets: values, lb, ub."""
    p = path
    if not os.path.isabs(p):
        base = os.path.dirname(__file__)
        p = os.path.join(base, "..", "..", p)
        p = os.path.abspath(p)
    with open(p, "r") as f:
        data = list(map(float, f.readline().strip().split(",")))
        lb = float(f.readline().strip())
        ub = float(f.readline().strip())
    return data, lb, ub


def read_csv_series(path: str, column: str = None):
    import pandas as pd

    df = pd.read_csv(path)
    if column is None:
        for c in df.columns:
            try:
                series = df[c].dropna().astype(float).tolist()
                return series
            except Exception:
                continue
        raise ValueError("No numeric column found in CSV")
    else:
        return df[column].dropna().astype(float).tolist()
