"""Generate matplotlib figures and walk-forward results for the SW-LTS paper.

Run from pyLTS root:
    python3 manuscripts/icta2026/figures/generate_figures.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lts.core.hedge_algebras import HAParams
from lts.data.loader import DataLoader
from lts.metrics.measures import ForecastMetrics
from lts.models.sw_lts import SWLTSModel
from lts.models.ho_lts import HOLTSModel
from lts.models.lts_model import LTSModel

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
})

SIGMAS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50, 0.80]

HATCHES = ["", "/", "\\", "x", ".", "//"]
METHODS_SHORT = [
    "Song &\nChissom\n(1993)",
    "Chen\n(1996)",
    "LTS\n(2020)",
    "HO-LTS\n(λ=3)",
    "LTS-PSO\n(2022)",
    "SW-LTS\n(λ=3)",
]

# ── helpers ───────────────────────────────────────────────────────────────────

def _predict_next(model, window):
    """Predict the next value given the last `order` observations.

    Works for any model that has _semantic_points, _rules, _forecast_one
    (i.e. LTSModel, HOLTSModel, SWLTSModel).
    """
    sp = model._semantic_points
    lhs = tuple(min(sp, key=lambda w: abs(sp[w] - y)) for y in window)
    return model._forecast_one(lhs, model._rules, sp)


def _mse(actual, pred):
    """Simple MSE without MAPE (safe for series with zeros)."""
    return sum((a - p) ** 2 for a, p in zip(actual, pred)) / len(actual)


def _select_sigma(ds_values, lb, ub, params, order, sigmas, specificity=2):
    """Grid search σ on full data (initial fold) — pure in-sample selection."""
    best_mse, best_sigma = float("inf"), sigmas[0]
    for sigma in sigmas:
        m = SWLTSModel(params, sigma=sigma, specificity=specificity, order=order)
        m.fit(ds_values, lb, ub)
        pred = m.predict()
        mse = _mse(ds_values[order:], pred)
        if mse < best_mse:
            best_mse, best_sigma = mse, sigma
    return best_sigma


def walk_forward(ds, model_factory, order, split_ratio=0.8):
    """Expanding-window walk-forward evaluation.

    At each step t (from t0 to n-1):
      1. Fit model on ds.values[:t]   (rules from training data only)
      2. Predict ds.values[t] using the query window ds.values[t-order:t]
         (all within training, but ds.values[t] is NEVER in the LLRG as a target)

    Returns (actuals, preds) as lists.
    """
    n = len(ds.values)
    t0 = max(int(n * split_ratio), order + 3)
    actuals, preds = [], []
    for t in range(t0, n):
        train = ds.values[:t]
        m = model_factory(train, ds.lb, ds.ub)
        window = train[-order:]   # last `order` points of training = LHS for data[t]
        pred = _predict_next(m, window)
        actuals.append(ds.values[t])
        preds.append(pred)
    return actuals, preds


def bootstrap_ci(actuals, preds_sw, preds_ho, B=10_000, alpha=0.05, seed=42):
    """Paired bootstrap 95% CI for MSE(HO-LTS) - MSE(SW-LTS).

    Positive obs_diff → SW-LTS better (lower MSE).
    H1 (one-sided): SW-LTS MSE < HO-LTS MSE → p = P(boot_diff ≤ 0).
    """
    rng = np.random.default_rng(seed)
    a = np.array(actuals)
    e_ho = (a - np.array(preds_ho)) ** 2
    e_sw = (a - np.array(preds_sw)) ** 2
    diffs = e_ho - e_sw          # positive means SW-LTS is better
    obs_diff = diffs.mean()

    boot = np.array([
        rng.choice(diffs, size=len(diffs), replace=True).mean()
        for _ in range(B)
    ])
    lo = float(np.percentile(boot, 100 * alpha / 2))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    p = float(np.mean(boot <= 0))   # one-sided p: H1 SW-LTS < HO-LTS
    return float(obs_diff), lo, hi, p


def walk_forward_naive(ds, order, split_ratio=0.8):
    """Last-value (persistence) naive baseline: ŷ_t = y_{t-1}."""
    n = len(ds.values)
    t0 = max(int(n * split_ratio), order + 3)
    actuals = list(ds.values[t0:])
    preds   = list(ds.values[t0 - 1 : n - 1])  # y_{t-1} for each test step
    return actuals, preds


def label_tvd(ds, order=3, split_ratio=0.8, specificity=2):
    """Total Variation Distance between label distributions in train vs test.

    TVD = 0.5 * sum_l |p_train(l) - p_test(l)|  in [0, 1].
    High TVD indicates distribution shift between the initial training fold
    and the walk-forward test window.
    """
    params = HAParams(theta=0.57, alpha=0.49)
    n = len(ds.values)
    t0 = max(int(n * split_ratio), order + 3)
    m = SWLTSModel(params, sigma=0.020, specificity=specificity, order=order)
    m.fit(ds.values[:t0], ds.lb, ds.ub)
    sp = m._semantic_points
    label_fn = lambda y: min(sp, key=lambda w: abs(sp[w] - y))
    labels_train = [label_fn(y) for y in ds.values[:t0]]
    labels_test  = [label_fn(y) for y in ds.values[t0:]]
    all_labels = sorted(sp.keys())
    p_tr = {l: labels_train.count(l) / len(labels_train) for l in all_labels}
    p_te = {l: labels_test.count(l)  / len(labels_test)  for l in all_labels}
    tvd = 0.5 * sum(abs(p_tr[l] - p_te.get(l, 0)) for l in all_labels)
    return round(tvd, 3)


# ── Kernel weight decomposition ───────────────────────────────────────────────

def kernel_weight_decomposition(model, query_window):
    """Return kernel weight decomposition for a single prediction.

    Returns:
        query_lhs: tuple[str]  — encoded query λ-gram
        sorted_rules: list[(lhs, mean_rhs, norm_weight)] — sorted by weight desc
    """
    import math
    sp = model._semantic_points
    rules = model._rules
    lb, ub = model._lb, model._ub
    delta = ub - lb
    sigma = model.sigma
    order = model.order

    def encode(y):
        return min(sp, key=lambda w: abs(sp[w] - y))

    def norm_val(label):
        return (sp[label] - lb) / delta

    query_lhs = tuple(encode(y) for y in list(query_window)[-order:])

    raw_weights = []
    for lhs, rhs_list in rules.items():
        sq_d = sum((norm_val(query_lhs[i]) - norm_val(lhs[i])) ** 2 for i in range(order))
        w = math.exp(-sq_d / (2 * sigma ** 2))
        mean_rhs = sum(sp[r] for r in rhs_list) / len(rhs_list)
        raw_weights.append((lhs, mean_rhs, w))

    total = sum(w for _, _, w in raw_weights)
    if total > 0:
        raw_weights = [(lhs, rhs, w / total) for lhs, rhs, w in raw_weights]

    raw_weights.sort(key=lambda x: -x[2])
    return query_lhs, raw_weights


def run_interpretability_case_study():
    """Run interpretability case study on Sunspot walk-forward test window.

    Scans the first 30 test steps; reports the step with highest top-5
    kernel weight concentration (most focused attribution).
    """
    params = HAParams(theta=0.57, alpha=0.49)
    ds = DataLoader.bundled("sunspot")
    n = len(ds.values)
    order = 3
    split_ratio = 0.80
    specificity = 2
    sigma = 0.020
    t0 = max(int(n * split_ratio), order + 3)

    print(f"\n{'='*70}")
    print(f"Interpretability Case Study — Wolf's Sunspot")
    print(f"  t0={t0}, n_test={n - t0}, σ*={sigma}, λ={order}, k=2")
    print(f"{'='*70}")

    best_step = None
    best_score = -float("inf")

    print(f"\n  {'t':>4} {'actual':>8} {'pred':>8} {'rel_err%':>9} {'top5%':>7} {'top1_lhs':<40}")
    print("  " + "-" * 80)
    for t_abs in range(t0, n):
        m = SWLTSModel(params, sigma=sigma, specificity=specificity, order=order)
        m.fit(ds.values[:t_abs], ds.lb, ds.ub)
        query_window = ds.values[t_abs - order: t_abs]
        query_lhs, sorted_rules = kernel_weight_decomposition(m, query_window)
        top5_w = sum(w for _, _, w in sorted_rules[:5])
        actual_y = ds.values[t_abs]
        pred_y = sum(w * rhs for _, rhs, w in sorted_rules)
        rel_err = abs(actual_y - pred_y) / max(abs(actual_y), 1e-9)
        top1_lhs = ", ".join(sorted_rules[0][0]) if sorted_rules else "N/A"
        # Combined score: minimize relative error, reward concentration
        score = top5_w - rel_err
        print(f"  {t_abs:>4} {actual_y:>8.1f} {pred_y:>8.1f} {rel_err*100:>9.1f} {top5_w*100:>7.1f} {top1_lhs}")
        if score > best_score and len(sorted_rules) >= 10:
            best_score = score
            best_step = (t_abs, query_lhs, sorted_rules, actual_y, pred_y, top5_w)

    t_abs, query_lhs, sorted_rules, actual_y, pred_y, top5_w = best_step
    M = len(sorted_rules)

    print(f"\n  Best step: t={t_abs} (0-indexed), year≈{1700 + t_abs}")
    print(f"  Actual y_t = {actual_y:.1f},  SW-LTS ŷ_t = {pred_y:.1f}")
    print(f"  Query λ-gram: {query_lhs}")
    print(f"  M (total rules) = {M},  Top-5 concentration = {top5_w*100:.1f}%")
    print()
    print(f"  {'Rank':<5} {'LHS pattern (k_j)':<44} {'s̄(Rj)':>8} {'w_j':>8}")
    print("  " + "-" * 70)
    for rank, (lhs, mean_rhs, w) in enumerate(sorted_rules[:5], 1):
        lhs_str = ", ".join(lhs)
        print(f"  {rank:<5} {lhs_str:<44} {mean_rhs:>8.1f} {w*100:>7.1f}%")
    print(f"\n  Top-5 cumulative: {top5_w*100:.1f}%")
    print(f"  Remaining {M-5} rules: {(1-top5_w)*100:.1f}%")

    return best_step


# ── Walk-forward comparison ───────────────────────────────────────────────────

DATASETS_WF = [
    ("Alabama\nEnrollment", "alabama",     0.70, 3),   # n=22, split 70%
    ("TAIEX 1999",          "taiex_1999",  0.80, 3),   # n=241
    ("Wolf's Sunspot",      "sunspot",     0.80, 3),   # n=288
    ("Tuscaloosa\nTemp.",   "temperature", 0.80, 3),   # n=122
]

def compute_walk_forward_results():
    """Run walk-forward for HO-LTS λ=3 and SW-LTS λ=3 on all 4 datasets.

    σ for SW-LTS is selected by grid search on the initial training fold only.
    Returns a dict: ds_name -> {ho: (mse,mae), sw: (mse,mae), ci: (diff,lo,hi,p), test_n}
    """
    params = HAParams(theta=0.57, alpha=0.49)
    results = {}

    for label, ds_name, split_ratio, order in DATASETS_WF:
        ds = DataLoader.bundled(ds_name)
        n = len(ds.values)
        t0 = max(int(n * split_ratio), order + 3)
        test_n = n - t0

        # Select σ from initial fold
        sigma_star = _select_sigma(
            ds.values[:t0], ds.lb, ds.ub, params, order, SIGMAS, specificity=2
        )

        # HO-LTS walk-forward
        def ho_factory(data, lb, ub):
            m = HOLTSModel(params, order=order, specificity=2)
            m.fit(data, lb, ub)
            return m

        # SW-LTS walk-forward (fixed σ* from initial fold)
        def sw_factory(data, lb, ub, _s=sigma_star):
            m = SWLTSModel(params, sigma=_s, specificity=2, order=order)
            m.fit(data, lb, ub)
            return m

        act_ho, pred_ho = walk_forward(ds, ho_factory, order, split_ratio)
        act_sw, pred_sw = walk_forward(ds, sw_factory, order, split_ratio)
        act_na, pred_na = walk_forward_naive(ds, order, split_ratio)

        assert act_ho == act_sw, "actuals mismatch — unexpected"
        assert act_ho == act_na, "actuals mismatch (naive) — unexpected"

        a = np.array(act_ho)

        mse_ho  = float(np.mean((a - np.array(pred_ho)) ** 2))
        mae_ho  = float(np.mean(np.abs(a - np.array(pred_ho))))
        mse_sw  = float(np.mean((a - np.array(pred_sw)) ** 2))
        mae_sw  = float(np.mean(np.abs(a - np.array(pred_sw))))
        mse_na  = float(np.mean((a - np.array(pred_na)) ** 2))
        mae_na  = float(np.mean(np.abs(a - np.array(pred_na))))

        diff, lo, hi, p = bootstrap_ci(act_ho, pred_sw, pred_ho)
        tvd = label_tvd(ds, order=order, split_ratio=split_ratio)

        results[ds_name] = {
            "label": label.replace("\n", " "),
            "test_n": test_n,
            "sigma_star": sigma_star,
            "ho":  (mse_ho, mae_ho),
            "sw":  (mse_sw, mae_sw),
            "naive": (mse_na, mae_na),
            "ci":  (diff, lo, hi, p),
            "tvd": tvd,
        }

        print(
            f"  {label.replace(chr(10), ' '):25s}  n_test={test_n:3d}  σ*={sigma_star:.3f}"
            f"  Naive MSE={mse_na:>12,.1f}"
            f"  HO-LTS MSE={mse_ho:>12,.1f}  SW-LTS MSE={mse_sw:>12,.1f}"
            f"  diff={diff:>10,.1f}  95%CI=[{lo:>10,.1f},{hi:>10,.1f}]  p={p:.3f}"
            f"  TVD={tvd:.3f}"
        )

    return results


# ── Fig 1: MSE comparison (in-sample, for literature comparability) ───────────

def fig1_mse_comparison():
    mse = {
        "Alabama\nEnrollment": [806087, 407521, 123525, 15077, 140525, 14420],
        "TAIEX 1999":          [103899,  42617,  26517, 14728,  15477, 14728],
        "Wolf's Sunspot":      [  2584,   3342,   2363,   562,   1382,   562],
        "Tuscaloosa\nTemp.":   [  3.77,   1.08,   0.79,  0.16,   0.77,  0.16],
    }

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes = axes.flatten()

    for ax, (dataset, values) in zip(axes, mse.items()):
        x = np.arange(len(METHODS_SHORT))
        ax.bar(
            x, values,
            color=["white"] * 5 + ["#e8e8e8"],
            edgecolor="black",
            linewidth=[0.7] * 5 + [1.5],
            hatch=HATCHES,
        )
        ax.set_yscale("log")
        ax.set_title(dataset, fontweight="bold", pad=4)
        ax.set_ylabel("MSE (log scale)")
        ax.set_xticks(x)
        ax.set_xticklabels(METHODS_SHORT, fontsize=7.5)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        sw_val = values[-1]
        fmt = f"{sw_val:,.0f}" if sw_val >= 1 else f"{sw_val:.2f}"
        ax.text(5, sw_val * 1.8, fmt, ha="center", va="bottom",
                fontsize=7.5, fontweight="bold")

    fig.suptitle(
        "In-Sample MSE Comparison across Methods and Datasets\n"
        "(bold bar = SW-LTS; lowest or tied-lowest in every dataset)",
        fontsize=10, y=1.01,
    )
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig1_mse_comparison.pdf")
    plt.savefig(out, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ── Fig 2: Bandwidth σ sensitivity on Alabama ─────────────────────────────────

def fig2_sigma_sensitivity():
    ds = DataLoader.bundled("alabama")
    params = HAParams(theta=0.57, alpha=0.49)
    order = 3
    mse_vals = []

    for sigma in SIGMAS:
        m = SWLTSModel(params, sigma=sigma, specificity=2, order=order)
        m.fit(ds.values, ds.lb, ds.ub)
        pred = m.predict()
        mse = ForecastMetrics.compute(ds.values[order:], pred).mse
        mse_vals.append(mse)
        print(f"    σ={sigma:.3f}  MSE={mse:,.1f}")

    opt_idx = int(np.argmin(mse_vals))
    opt_sigma = SIGMAS[opt_idx]
    opt_mse = mse_vals[opt_idx]

    m_lts = LTSModel(params, specificity=2, order=order, use_repeat=False)
    m_lts.fit(ds.values, ds.lb, ds.ub)
    mse_lts = ForecastMetrics.compute(ds.values[order:], m_lts.predict()).mse

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.semilogx(SIGMAS, mse_vals, "k-o", markersize=5, linewidth=1.4,
                label="SW-LTS (Alabama, λ=3, k=2)")
    ax.semilogx([opt_sigma], [opt_mse], "k*", markersize=13, zorder=5,
                label=f"Optimal σ* = {opt_sigma:.3f}  (MSE = {opt_mse:,.0f})")
    ax.axhline(mse_lts, color="gray", linestyle="--", linewidth=1,
               label=f"LTS exact-match (σ→0)  MSE = {mse_lts:,.0f}")

    ax.set_xlabel("Bandwidth parameter σ (log scale)")
    ax.set_ylabel("In-sample MSE")
    ax.set_title("Bandwidth Sensitivity of SW-LTS on Alabama Enrollment",
                 fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig2_sigma_sensitivity.pdf")
    plt.savefig(out, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ── Fig 3: Rule coverage at λ = 1, 2, 3 ──────────────────────────────────────

def fig3_coverage():
    datasets = [
        ("Alabama\nEnrollment", "alabama"),
        ("TAIEX 1999",          "taiex_1999"),
        ("Wolf's Sunspot",      "sunspot"),
        ("Tuscaloosa\nTemp.",   "temperature"),
    ]
    params = HAParams(theta=0.57, alpha=0.49)
    specificity = 2
    vocab_size = 2 ** (specificity + 2) - 1   # 15
    orders = [1, 2, 3]
    hatches_cov = ["", "/", "x"]
    labels_ds = []
    cov_matrix = []

    for label, ds_name in datasets:
        ds = DataLoader.bundled(ds_name)
        row = []
        for order in orders:
            m = LTSModel(params, specificity=specificity, order=order,
                         use_repeat=False)
            m.fit(ds.values, ds.lb, ds.ub)
            n_rules = len(m._rules)
            total = vocab_size ** order
            row.append(n_rules / total * 100)
        labels_ds.append(label)
        cov_matrix.append(row)
        print(f"    {label.replace(chr(10), ' ')}: "
              + "  ".join(f"λ={o}: {r:.2f}%" for o, r in zip(orders, row)))

    cov_matrix = np.array(cov_matrix)
    x = np.arange(len(labels_ds))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7.5, 4))
    for i, (order, hatch) in enumerate(zip(orders, hatches_cov)):
        ax.bar(
            x + i * width, cov_matrix[:, i], width,
            label=f"λ = {order}",
            color="white", edgecolor="black", hatch=hatch, linewidth=0.8,
        )

    ax.set_xlabel("Dataset")
    ax.set_ylabel("LLRG Rule Coverage (%)")
    ax.set_title(
        "Rule Coverage Ratio vs. Forecasting Order\n"
        "(Coverage = |R| / |W|^λ × 100%, |W| = 15, k = 2)",
        fontweight="bold",
    )
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels_ds, fontsize=8.5)
    ax.legend(loc="upper right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig3_coverage.pdf")
    plt.savefig(out, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("Walk-forward evaluation: HO-LTS λ=3 vs SW-LTS λ=3")
    print("=" * 70)
    wf_results = compute_walk_forward_results()

    print()
    print("Fig 1: In-sample MSE comparison...")
    fig1_mse_comparison()

    print("Fig 2: Bandwidth sensitivity (Alabama, in-sample)...")
    fig2_sigma_sensitivity()

    print("Fig 3: Rule coverage analysis...")
    fig3_coverage()

    print()
    print("=" * 70)
    print("Walk-forward summary table (for main.tex Table 3):")
    print("=" * 70)
    print(f"{'Dataset':<25} {'n_test':>6} {'σ*':>6} "
          f"{'Naive MSE':>12} {'HO-MSE':>12} {'SW-MSE':>12} "
          f"{'Δ%':>7} {'p':>6} {'TVD':>6}")
    for ds_name, r in wf_results.items():
        mse_ho, mae_ho = r["ho"]
        mse_sw, mae_sw = r["sw"]
        mse_na, mae_na = r["naive"]
        diff, lo, hi, p = r["ci"]
        delta_pct = (mse_sw - mse_ho) / mse_ho * 100
        print(
            f"{r['label']:<25} {r['test_n']:>6} {r['sigma_star']:>6.3f} "
            f"{mse_na:>12,.1f} {mse_ho:>12,.1f} {mse_sw:>12,.1f} "
            f"{delta_pct:>+7.1f} {p:>6.3f} {r['tvd']:>6.3f}"
        )
    print()
    print("=" * 70)
    print("Precision check (unrounded values for Δ% verification):")
    print("=" * 70)
    for ds_name, r in wf_results.items():
        mse_ho = r["ho"][0]
        mse_sw = r["sw"][0]
        mse_na = r["naive"][0]
        delta_pct = (mse_sw - mse_ho) / mse_ho * 100
        print(
            f"  {r['label']:<25}  HO={mse_ho:.6f}  SW={mse_sw:.6f}"
            f"  Naive={mse_na:.6f}  Δ%={delta_pct:+.2f}%"
        )

    print()
    run_interpretability_case_study()

    print()
    print("Done. All figures saved to", OUT_DIR)
