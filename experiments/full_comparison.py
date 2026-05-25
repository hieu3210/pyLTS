"""Full comparison experiment: tất cả models trên tất cả datasets.

Chạy: python3 experiments/full_comparison.py
"""
from __future__ import annotations

import math
import sys
import time
sys.path.insert(0, "/Users/macos/Documents/GitHub/pyLTS")

from lts.core.hedge_algebras import HAParams
from lts.data.loader import DataLoader
from lts.metrics.measures import ForecastMetrics
from lts.models.chen1996 import Chen1996
from lts.models.song_chissom1993 import SongChissom1993
from lts.models.lts_model import LTSModel
from lts.models.ho_lts import HOLTSModel
from lts.models.lts_pso import LTSPSOModel
from lts.models.co_lts import COLTSModel, COLTSConfig
from lts.models.sw_lts import SWLTSModel, SWPSOLTSModel
from lts.optimization.pso import PSOConfig

# ─── Metrics ────────────────────────────────────────────────────────────────

def compute_metrics(actual: list[float], predicted: list[float]) -> dict:
    n = len(actual)
    if n == 0:
        return {"MSE": float("nan"), "RMSE": float("nan"), "MAPE": float("nan"), "MAE": float("nan")}
    mse = sum((a - p) ** 2 for a, p in zip(actual, predicted)) / n
    rmse = math.sqrt(mse)
    mae = sum(abs(a - p) for a, p in zip(actual, predicted)) / n
    # MAPE: bỏ qua các điểm actual == 0
    mape_terms = [abs(a - p) / abs(a) * 100 for a, p in zip(actual, predicted) if abs(a) > 1e-9]
    mape = sum(mape_terms) / len(mape_terms) if mape_terms else float("nan")
    return {"MSE": mse, "RMSE": rmse, "MAPE": mape, "MAE": mae}

# ─── Dataset config ──────────────────────────────────────────────────────────

DATASETS = [
    "alabama",
    "car_accident",
    "taifex_1998",
    "spot_gold",
    "temperature",
    "gas_vietnam",
    "sunspot",
    "taiex_monthly",
]

# ─── Helpers ─────────────────────────────────────────────────────────────────

def best_lts(data, lb, ub):
    """Grid search LTS: tìm best (spec, order) theo MSE."""
    best = {"mse": float("inf"), "model": None, "order": 1, "spec": 1}
    params = HAParams.enrollment()
    for spec in [1, 2]:
        for order in [1, 2, 3]:
            try:
                m = LTSModel(params, specificity=spec, order=order, use_repeat=False)
                m.fit(data, lb, ub)
                pred = m.predict()
                mse = sum((a-p)**2 for a,p in zip(data[order:], pred)) / len(pred)
                if mse < best["mse"]:
                    best = {"mse": mse, "model": m, "order": order, "spec": spec}
            except Exception:
                pass
    return best["model"], best["order"], best["spec"]

def best_holts(data, lb, ub):
    """Grid search HO-LTS: tìm best (spec, order)."""
    best = {"mse": float("inf"), "model": None, "order": 2, "spec": 1}
    params = HAParams.enrollment()
    for spec in [1, 2, 3]:
        for order in [1, 2, 3, 4, 5]:
            try:
                m = HOLTSModel(params, order=order, specificity=spec)
                m.fit(data, lb, ub)
                pred = m.predict()
                mse = sum((a-p)**2 for a,p in zip(data[order:], pred)) / len(pred)
                if mse < best["mse"]:
                    best = {"mse": mse, "model": m, "order": order, "spec": spec}
            except Exception:
                pass
    return best["model"], best["order"], best["spec"]

def best_swlts(data, lb, ub):
    """Grid search SW-LTS: tìm best (spec, order, sigma)."""
    best = {"mse": float("inf"), "model": None, "sigma": 0.1, "order": 1, "spec": 1}
    params = HAParams.enrollment()
    for spec in [1, 2]:
        for order in [1, 2, 3]:
            for sigma in [0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8]:
                try:
                    m = SWLTSModel(params, sigma=sigma, specificity=spec, order=order)
                    m.fit(data, lb, ub)
                    pred = m.predict()
                    mse = sum((a-p)**2 for a,p in zip(data[order:], pred)) / len(pred)
                    if mse < best["mse"]:
                        best = {"mse": mse, "model": m, "sigma": sigma, "order": order, "spec": spec}
                except Exception:
                    pass
    return best["model"], best["order"], best["spec"], best["sigma"]

def run_lts_pso(data, lb, ub):
    """LTS-PSO với quick PSO config."""
    try:
        cfg = PSOConfig(n_particles=30, max_iter=100, omega=0.4, c1=2.0, c2=2.0,
                        bounds=[(0.3, 0.7), (0.3, 0.7)], seed=0)
        m = LTSPSOModel(HAParams.enrollment(), specificity=2, order=1)
        m.fit_optimize(data, lb, ub, pso_config=cfg, n_runs=2)
        return m
    except Exception as e:
        return None

def run_colts(data, lb, ub):
    """CO-LTS với cấu hình nhỏ để chạy nhanh."""
    try:
        cfg = COLTSConfig(
            k_max=3, d_w=7,
            outer_n=10, outer_max_iter=15,
            inner_m=15, inner_max_iter=30,
            n_runs=1, order=1,
        )
        m = COLTSModel(cfg)
        m.fit(data, lb, ub)
        return m
    except Exception as e:
        return None

def run_swpso(data, lb, ub):
    """SW-PSO-LTS với quick PSO config."""
    try:
        cfg = PSOConfig(n_particles=30, max_iter=100, omega=0.4, c1=2.0, c2=2.0,
                        bounds=[(0.3, 0.7), (0.3, 0.7), (0.01, 1.0)], seed=0)
        m = SWPSOLTSModel(specificity=1, order=1, n_runs=2)
        m._pso_config = cfg
        m.fit(data, lb, ub)
        return m
    except Exception as e:
        return None

# ─── Main experiment ─────────────────────────────────────────────────────────

def run_experiment():
    results = {}  # {dataset: {method: metrics}}
    configs = {}  # {dataset: {method: config_info}}

    for ds_name in DATASETS:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print('='*60)

        try:
            ds = DataLoader.bundled(ds_name)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        data = ds.values
        lb, ub = ds.lb, ds.ub
        n = len(data)
        results[ds_name] = {}
        configs[ds_name] = {}

        # 1. Song & Chissom (1993)
        print("  [1/7] Song & Chissom 1993...", end=" ", flush=True)
        try:
            m = SongChissom1993(n_intervals=7)
            m.fit(data, lb, ub)
            pred = m.predict()
            actual = data[1:]
            results[ds_name]["Song&Chissom (1993)"] = compute_metrics(actual, pred)
            configs[ds_name]["Song&Chissom (1993)"] = "n_int=7"
            print("OK")
        except Exception as e:
            print(f"FAIL: {e}")

        # 2. Chen (1996)
        print("  [2/7] Chen 1996...", end=" ", flush=True)
        try:
            m = Chen1996(n_intervals=7, order=1)
            m.fit(data, lb, ub)
            pred = m.predict()
            actual = data[1:]
            results[ds_name]["Chen (1996)"] = compute_metrics(actual, pred)
            configs[ds_name]["Chen (1996)"] = "n_int=7"
            print("OK")
        except Exception as e:
            print(f"FAIL: {e}")

        # 3. LTS (2020) — grid search best config
        print("  [3/7] LTS 2020 (grid)...", end=" ", flush=True)
        try:
            m, order, spec = best_lts(data, lb, ub)
            pred = m.predict()
            actual = data[order:]
            results[ds_name]["LTS (2020)"] = compute_metrics(actual, pred)
            configs[ds_name]["LTS (2020)"] = f"sp={spec},ord={order}"
            print(f"OK (sp={spec},ord={order})")
        except Exception as e:
            print(f"FAIL: {e}")

        # 4. HO-LTS (2021) — grid search
        print("  [4/7] HO-LTS 2021 (grid)...", end=" ", flush=True)
        try:
            m, order, spec = best_holts(data, lb, ub)
            pred = m.predict()
            actual = data[order:]
            results[ds_name]["HO-LTS (2021)"] = compute_metrics(actual, pred)
            configs[ds_name]["HO-LTS (2021)"] = f"sp={spec},ord={order}"
            print(f"OK (sp={spec},ord={order})")
        except Exception as e:
            print(f"FAIL: {e}")

        # 5. LTS-PSO (2022)
        print("  [5/7] LTS-PSO 2022 (quick PSO)...", end=" ", flush=True)
        t0 = time.time()
        m = run_lts_pso(data, lb, ub)
        if m:
            pred = m.predict()
            actual = data[1:]
            results[ds_name]["LTS-PSO (2022)"] = compute_metrics(actual, pred)
            configs[ds_name]["LTS-PSO (2022)"] = "N=30,G=100"
            print(f"OK ({time.time()-t0:.1f}s)")
        else:
            print("FAIL")

        # 6. CO-LTS (2023)
        print("  [6/7] CO-LTS 2023 (quick)...", end=" ", flush=True)
        t0 = time.time()
        m = run_colts(data, lb, ub)
        if m:
            pred = m.predict()
            actual = data[1:]
            results[ds_name]["CO-LTS (2023)"] = compute_metrics(actual, pred)
            configs[ds_name]["CO-LTS (2023)"] = "k=3,dw=7"
            print(f"OK ({time.time()-t0:.1f}s)")
        else:
            print("FAIL")

        # 7. SW-PSO-LTS (proposed)
        print("  [7/7] SW-PSO-LTS (grid+PSO)...", end=" ", flush=True)
        t0 = time.time()

        # Grid search first
        m_grid, order_g, spec_g, sigma_g = best_swlts(data, lb, ub)
        pred_grid = m_grid.predict()
        actual_grid = data[order_g:]
        results[ds_name]["SW-LTS (grid)"] = compute_metrics(actual_grid, pred_grid)
        configs[ds_name]["SW-LTS (grid)"] = f"sp={spec_g},ord={order_g},σ={sigma_g}"

        # Also quick PSO
        m_pso = run_swpso(data, lb, ub)
        if m_pso:
            pred_pso = m_pso.predict()
            actual_pso = data[1:]
            results[ds_name]["SW-PSO-LTS"] = compute_metrics(actual_pso, pred_pso)
            th, al, si = m_pso.best_params
            configs[ds_name]["SW-PSO-LTS"] = f"θ={th:.3f},α={al:.3f},σ={si:.3f}"
        print(f"OK ({time.time()-t0:.1f}s)")

    return results, configs

# ─── Print table ─────────────────────────────────────────────────────────────

def print_table(results: dict, configs: dict):
    METHOD_ORDER = [
        "Song&Chissom (1993)", "Chen (1996)",
        "LTS (2020)", "HO-LTS (2021)", "LTS-PSO (2022)", "CO-LTS (2023)",
        "SW-LTS (grid)", "SW-PSO-LTS",
    ]

    print("\n" + "="*120)
    print("COMPREHENSIVE COMPARISON TABLE")
    print("="*120)

    for ds_name, ds_results in results.items():
        ds = DataLoader.bundled(ds_name)
        print(f"\n--- Dataset: {ds_name.upper()} (n={len(ds.values)}, lb={ds.lb:.0f}, ub={ds.ub:.0f}) ---")
        print(f"{'Method':<22} {'Config':<24} {'MSE':>14} {'RMSE':>10} {'MAPE%':>8} {'MAE':>12}")
        print("-" * 95)

        best_mse = min(
            (m["MSE"] for m in ds_results.values() if math.isfinite(m["MSE"])),
            default=float("inf")
        )

        for method in METHOD_ORDER:
            if method not in ds_results:
                continue
            m = ds_results[method]
            cfg = configs.get(ds_name, {}).get(method, "")

            mse_str = f"{m['MSE']:14.1f}"
            rmse_str = f"{m['RMSE']:10.2f}"
            mape_str = f"{m['MAPE']:8.2f}" if math.isfinite(m['MAPE']) else "    N/A "
            mae_str = f"{m['MAE']:12.2f}"

            marker = " ★" if abs(m["MSE"] - best_mse) < 1e-3 * best_mse else ""
            print(f"{method:<22} {cfg:<24} {mse_str} {rmse_str} {mape_str} {mae_str}{marker}")

    # Summary: which method wins most
    print("\n" + "="*80)
    print("SUMMARY: Best method per dataset (lowest MSE)")
    print("="*80)
    wins = {}
    for ds_name, ds_results in results.items():
        if not ds_results:
            continue
        best_method = min(
            ds_results.items(),
            key=lambda kv: kv[1]["MSE"] if math.isfinite(kv[1]["MSE"]) else float("inf")
        )
        wins[ds_name] = best_method[0]
        print(f"  {ds_name:<20} → {best_method[0]} (MSE={best_method[1]['MSE']:.1f})")

    win_counts = {}
    for m in wins.values():
        win_counts[m] = win_counts.get(m, 0) + 1
    print("\nWin counts:")
    for m, c in sorted(win_counts.items(), key=lambda x: -x[1]):
        print(f"  {m:<22}: {c} win(s)")


if __name__ == "__main__":
    t_total = time.time()
    results, configs = run_experiment()
    print_table(results, configs)
    print(f"\nTotal elapsed: {time.time()-t_total:.1f}s")
