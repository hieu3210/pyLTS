#!/usr/bin/env python3
"""
pyLTS top-level runner

This script provides a small CLI to run the existing LTS and ILTS implementations
in `LTS/` without changing their computation logic. It loads the package modules
in-place and invokes the classes with user-supplied parameters.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from typing import List, Tuple

import pandas as pd
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


ROOT = os.path.dirname(__file__)
LTS_ROOT = os.path.join(ROOT, "LTS")


def _ensure_lts_in_path():
    if LTS_ROOT not in sys.path:
        sys.path.insert(0, LTS_ROOT)


def _load_module(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    # register under the requested name so in-module imports find it
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore
    return module


def load_core_modules():
    """Load available core modules from the `LTS` folder.

    Returns tuple: (ha_module, lts_module, ilts_module_or_None, errors_module_or_None)
    """
    _ensure_lts_in_path()
    ha_path = os.path.join(LTS_ROOT, "HAs", "__init__.py")
    lts_pkg_path = os.path.join(LTS_ROOT, "LTS", "__init__.py")
    ilts_path = os.path.join(LTS_ROOT, "ILTS", "__init__.py")
    errors_path = os.path.join(LTS_ROOT, "Errors", "__init__.py")

    # HA and LTS packages are required
    if not os.path.exists(ha_path):
        raise FileNotFoundError(f"Required module not found: {ha_path}")
    if not os.path.exists(lts_pkg_path):
        raise FileNotFoundError(f"Required module not found: {lts_pkg_path}")

    ha = _load_module("HAs", ha_path)
    lts = _load_module("LTS_pkg", lts_pkg_path)

    # ILTS and Errors are optional — load if present
    ilts = None
    errs = None
    if os.path.exists(ilts_path):
        ilts = _load_module("ILTS_pkg", ilts_path)
    if os.path.exists(errors_path):
        errs = _load_module("Errors", errors_path)

    return ha, lts, ilts, errs


def read_dataset_file(path: str) -> Tuple[List[float], float, float]:
    """Read a dataset in the legacy format used in `LTS/LTS.py`.

    Expected format (three lines):
    - comma separated numeric values (single line)
    - lower bound (single number)
    - upper bound (single number)
    """
    with open(path, "r") as f:
        data = list(map(float, f.readline().strip().split(",")))
        lb = float(f.readline().strip())
        ub = float(f.readline().strip())
    return data, lb, ub


def read_csv_series(path: str, column: str | None = None) -> List[float]:
    df = pd.read_csv(path)
    if column:
        series = df[column].dropna().astype(float).tolist()
    else:
        # take first numeric column
        for col in df.columns:
            try:
                series = df[col].dropna().astype(float).tolist()
                break
            except Exception:
                continue
    return series


def run_model(args):
    ha_mod, lts_mod, ilts_mod, errs = load_core_modules()

    if ha_mod is None or lts_mod is None:
        raise RuntimeError("Required core modules (HAs, LTS) could not be loaded.")

    # Create words using HedgeAlgebras
    HedgeAlgebra = ha_mod.HedgeAlgebras
    theta = args.theta
    alpha = args.alpha
    ha_obj = HedgeAlgebra(theta, alpha)
    words = ha_obj.get_words(args.k)

    # Load data
    if args.dataset:
        ds_path = os.path.join(LTS_ROOT, "Datasets", args.dataset)
        data, lb, ub = read_dataset_file(ds_path)
    elif args.csv:
        series = read_csv_series(args.csv, args.csv_column)
        data = series
        lb = args.lb if args.lb is not None else min(series)
        ub = args.ub if args.ub is not None else max(series)
    else:
        raise RuntimeError("No dataset specified. Use --dataset or --csv")

    # Instantiate model using a local copy of the data to avoid side-effects
    order = args.order
    repeat = args.repeat
    data_for_model = list(data)

    if args.model == "lts":
        ModelClass = getattr(lts_mod, "LTS")
        model = ModelClass(order, repeat, data_for_model, lb, ub, words, theta, alpha)
    else:
        if ilts_mod is None:
            raise RuntimeError("ILTS module is not available in this repository. Use --model lts or add ILTS module.")
        ModelClass = getattr(ilts_mod, "ILTS")
        model = ModelClass(order, repeat, data_for_model, lb, ub, words, theta, alpha, args.length)

    forecasted = model.results

    # Print detailed outputs mirroring LTS/LTS.py
    print(str(len(words)) + " words and their SQM:")
    print(words)
    # semantics in [0,1]
    try:
        print(model.get_semantic())
    except Exception:
        pass
    try:
        print(model.get_real_semantics())
    except Exception:
        pass

    # Data labels
    try:
        labels = model.get_label_of_data()
        print("Data labels (" + str(len(labels)) + " points):")
        print(labels)
    except Exception:
        pass

    # LLRGs (rules)
    try:
        if repeat:
            print(str(len(model.lhs)) + " LLRGs (repeated):")
        else:
            print(str(len(model.lhs)) + " LLRGs (no-repeated):")
        for i in range(len(model.lhs)):
            print(model.lhs[i], end='')
            print("  \u2192  ", end='')
            print(model.rhs[i])
    except Exception:
        pass

    # Forecasted results
    print("Results (" + str(len(forecasted)) + " values):")
    print(forecasted)

    # Optionally save
    if args.output:
        import csv

        out_path = args.output
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["forecasted"])
            for v in forecasted:
                writer.writerow([v])

    # Assessment: align actuals (drop first `order` points) and compute errors
    try:
        # prefer Errors.Measure module if loaded
        if errs is not None and hasattr(errs, 'Measure'):
            MeasureClass = errs.Measure
        else:
            from Errors import Measure as MeasureClass

        actual_for_eval = list(data_for_model)[order:]
        if len(actual_for_eval) > 0 and len(forecasted) > 0:
            m = MeasureClass(actual_for_eval, forecasted)
            print("MAE = " + str(m.mae()))
            print("MSE = " + str(m.mse()))
            print("RMSE = " + str(m.rmse()))
            print("MAPE = " + str(m.mape(2)) + "%")
        else:
            print("No data available for assessment.")
    except Exception:
        pass

    # Plotting: simplified numeric comparison only
    if args.plot:
        if plt is None:
            print("matplotlib not available — install it (pip install matplotlib) to enable plotting.")
        else:
            try:
                # Prepare actual and forecast alignment
                x = list(range(order, order + max(len(actual_for_eval), len(forecasted))))
                # Pad shorter series with None so matplotlib aligns indexes
                actual_plot = list(actual_for_eval) + [None] * (len(x) - len(actual_for_eval))
                forecast_plot = list(forecasted) + [None] * (len(x) - len(forecasted))

                plt.figure(figsize=(10, 5))
                plt.plot(x, actual_plot, label='Actual', marker='o')
                plt.plot(x, forecast_plot, label='Forecast', marker='x')
                plt.title('Actual vs Forecasted')
                plt.xlabel('Index')
                plt.ylabel('Value')
                plt.legend()
                plt.grid(True)
                plt.show()
            except Exception as e:
                print('Plotting failed:', str(e))


def build_parser():
    p = argparse.ArgumentParser(description="pyLTS top-level runner")
    p.add_argument("--model", choices=["lts", "ilts"], default="lts", help="Model to run")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--dataset", help="Name of dataset file under LTS/Datasets/ (e.g. alabama.txt)")
    g.add_argument("--csv", help="Path to a CSV file with a numeric series (first numeric column used)")
    p.add_argument("--csv-column", help="Column name to use from CSV (optional)")
    p.add_argument("--order", type=int, default=1, help="Model order (default: 1)")
    p.add_argument("--repeat", action="store_true", help="Use repeat LLRs (flag)")
    p.add_argument("--lb", type=float, help="Lower bound for numeric series (when using CSV)")
    p.add_argument("--ub", type=float, help="Upper bound for numeric series (when using CSV)")
    p.add_argument("--k", type=int, default=3, help="Max word length (k) for HA.get_words (default: 3)")
    p.add_argument("--theta", type=float, default=0.57, help="HA theta parameter (default: 0.57)")
    p.add_argument("--alpha", type=float, default=0.49, help="HA alpha parameter (default: 0.49)")
    p.add_argument("--length", type=int, default=3, help="(ILTS) max length of words")
    p.add_argument("--output", help="Save forecasted values to CSV")
    p.add_argument("--show", action="store_true", help="Print forecasted values to stdout")
    p.add_argument("--plot", action="store_true", help="Show plots comparing original and forecasted series (requires matplotlib)")
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    run_model(args)


if __name__ == "__main__":
    main()
