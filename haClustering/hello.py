#!/usr/bin/env python3
"""
Refactored semantic-value 1D clustering (clean, consistent, testable).
- CLI via argparse
- Config dataclass
- Clear separation: io, preprocessing, clustering, plotting
- Lightweight logging and error handling
"""
from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np

# Optional imports (import on demand to keep startup light)
_pd = None
_plt = None

# Default sample dataset (height, weight)
_SAMPLE_DATA = np.array(
    [
        [150, 50],
        [152, 48],
        [155, 52],
        [158, 54],
        [160, 60],
        [162, 58],
        [165, 65],
        [168, 66],
        [170, 70],
        [172, 72],
        [173, 74],
        [175, 75],
        [177, 88],
        [178, 78],
        [180, 82],
        [182, 85],
        [185, 90],
        [190, 95],
        [195, 100],
        [145, 45],
        [148, 47],
        [159, 57],
        [167, 64],
        [169, 68],
    ],
    dtype=float,
)


@dataclass
class Config:
    n_clusters: int = 2
    init_method: str = "linear"  # 'linear', 'sample', 'uniform'
    seed: Optional[int] = 42
    max_iter: int = 100
    tol: float = 1e-6
    csv_path: Optional[str] = None
    no_plot: bool = False
    verbose: bool = False


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format="%(levelname)s: %(message)s", level=level)


def minmax_norm(data: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Min-max normalize per-column. If max==min for a column, result is 0 for that column.
    Returns array with same shape.
    """
    data = np.asarray(data, dtype=float)
    if data.size == 0:
        return data.copy()
    mins = np.min(data, axis=axis, keepdims=True)
    maxs = np.max(data, axis=axis, keepdims=True)
    denom = maxs - mins
    denom_safe = np.where(denom == 0, 1.0, denom)
    normalized = (data - mins) / denom_safe
    # columns with zero range -> set to 0
    if np.any(denom == 0):
        normalized = np.where(denom == 0, 0.0, normalized)
    return normalized


def semantic_value(row: Iterable[float]) -> float:
    """
    Compute semantic value for a single normalized data row.
    Default: mean of features.
    """
    arr = np.asarray(list(row), dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr))


def load_height_weight(csv_path: Optional[str] = None):
    """
    Load height/weight data from CSV (if provided or found via env/current dir/~/data),
    otherwise return embedded sample.
    """
    global _pd
    candidates = [
        csv_path,
        os.environ.get("HEIGHT_WEIGHT_CSV"),
        "height_weight.csv",
        os.path.expanduser("~/data/height_weight.csv"),
    ]
    for p in candidates:
        if not p:
            continue
        if os.path.exists(p):
            # lazy import pandas
            if _pd is None:
                try:
                    import pandas as pd  # type: ignore

                    _pd = pd
                except Exception as exc:  # pragma: no cover - environment dependent
                    raise RuntimeError(
                        "pandas is required to load a CSV dataset. Install it: pip install pandas"
                    ) from exc
            df = _pd.read_csv(p)
            # heuristics for height/weight columns
            possible_h = ["height", "Height", "stature", "Stature", "ht", "HT"]
            possible_w = ["weight", "Weight", "mass", "Mass", "wt", "WT"]
            for h in possible_h:
                for w in possible_w:
                    if h in df.columns and w in df.columns:
                        return df[[h, w]].to_numpy(dtype=float)
            numeric = df.select_dtypes(include=["number"]).columns
            if len(numeric) >= 2:
                return df[numeric[:2]].to_numpy(dtype=float)
            raise RuntimeError(f"Found file {p} but could not locate height/weight columns.")
    return _SAMPLE_DATA.copy()


class KMeansSemantic:
    """
    Minimal K-means style clustering on 1D semantic values.
    """

    def __init__(self, n_clusters: int, init_method: str = "linear", seed: Optional[int] = None):
        if n_clusters <= 0:
            raise ValueError("n_clusters must be > 0")
        self.n_clusters = n_clusters
        self.init_method = init_method
        self.rng = np.random.default_rng(seed)
        self.centers = np.zeros(n_clusters, dtype=float)

    def _init_centers(self, semantic_values: np.ndarray) -> np.ndarray:
        sv = np.asarray(semantic_values).reshape(-1)
        N = self.n_clusters
        if self.init_method == "linear":
            return np.linspace(0.0, 1.0, N + 2)[1:-1]
        if self.init_method == "sample" and sv.size >= N:
            # sample without replacement
            return self.rng.choice(sv, size=N, replace=False)
        # fallback: uniform random in [0,1]
        return self.rng.uniform(0.0, 1.0, size=N)

    @staticmethod
    def _assign(semantic_values: np.ndarray, centers: np.ndarray) -> np.ndarray:
        sv = np.asarray(semantic_values).reshape(-1)
        cs = np.asarray(centers).reshape(-1)
        dists = np.abs(sv[:, None] - cs[None, :])
        return np.argmin(dists, axis=1)

    def _update_centers(
        self, semantic_values: np.ndarray, assignments: np.ndarray, prev_centers: Optional[np.ndarray]
    ) -> np.ndarray:
        sv = np.asarray(semantic_values).reshape(-1)
        centers = np.zeros(self.n_clusters, dtype=float)
        for k in range(self.n_clusters):
            members = sv[assignments == k]
            if members.size > 0:
                centers[k] = float(np.mean(members))
            else:
                centers[k] = float(prev_centers[k]) if prev_centers is not None else float(self.rng.choice(sv) if sv.size else 0.5)
        return centers

    def fit(
        self,
        semantic_values: np.ndarray,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        sv = np.asarray(semantic_values).reshape(-1)
        centers = self._init_centers(sv)
        for it in range(1, max_iter + 1):
            assignments = self._assign(sv, centers)
            new_centers = self._update_centers(sv, assignments, prev_centers=centers)
            if np.allclose(centers, new_centers, atol=tol, rtol=0):
                self.centers = new_centers
                return new_centers, assignments, it
            centers = new_centers
        self.centers = centers
        return centers, assignments, max_iter


def plot_results(
    data: np.ndarray,
    semantic_vals: np.ndarray,
    centers: np.ndarray,
    assignments: np.ndarray,
    figsize=(10, 4),
) -> None:
    """
    Scatter of 2D data colored by cluster + 1D semantic axis with centers.
    If matplotlib is not available, logs a message and returns.
    """
    global _plt
    try:
        if _plt is None:
            import matplotlib.pyplot as plt  # type: ignore

            _plt = plt
    except Exception:  # pragma: no cover - matplotlib optional
        logging.info("matplotlib not available; skipping plot.")
        return

    plt = _plt
    data = np.asarray(data)
    N = len(centers)
    cmap = plt.cm.get_cmap("tab10", max(3, N))
    plt.figure(figsize=figsize)

    plt.subplot(1, 2, 1)
    for k in range(N):
        mask = assignments == k
        plt.scatter(
            data[mask, 0],
            data[mask, 1],
            color=cmap(k),
            label=f"Cluster {k+1}",
            s=80,
            edgecolors="k",
        )
    plt.xlabel("Height")
    plt.ylabel("Weight")
    plt.title("Data points by cluster")
    plt.legend(frameon=False)

    plt.subplot(1, 2, 2)
    plt.scatter(
        semantic_vals,
        np.zeros_like(semantic_vals),
        c=[cmap(k) for k in assignments],
        s=80,
        edgecolors="k",
    )
    for k, c in enumerate(centers):
        plt.axvline(c, color=cmap(k), linestyle="--", label=f"Center {k+1}")
    plt.xlabel("Semantic value (normalized)")
    plt.yticks([])
    plt.title("Semantic values and cluster centers")
    plt.legend(frameon=False)

    plt.tight_layout()
    plt.show()


def run(cfg: Config) -> int:
    setup_logging(cfg.verbose)
    logging.debug("Configuration: %s", cfg)

    try:
        data = load_height_weight(cfg.csv_path)
        logging.info("Loaded %d data points", len(data))
    except Exception as exc:
        logging.error("Failed to load dataset: %s", exc)
        return 2

    norm_data = minmax_norm(data, axis=0)
    semantic_vals = np.array([semantic_value(row) for row in norm_data])

    clusterer = KMeansSemantic(n_clusters=cfg.n_clusters, init_method=cfg.init_method, seed=cfg.seed)
    centers, assignments, n_iter = clusterer.fit(semantic_vals, max_iter=cfg.max_iter, tol=cfg.tol)

    logging.info("Converged in %d iterations", n_iter)
    for i, row in enumerate(data):
        logging.debug("Data %s -> cluster %d", row.tolist(), int(assignments[i]) + 1)
    # Print summary to stdout (concise)
    print(f"Converged in {n_iter} iterations")
    for i, row in enumerate(data):
        print(f"Data point {row.tolist()}: Cluster {int(assignments[i]) + 1}")

    if not cfg.no_plot:
        try:
            plot_results(data, semantic_vals, centers, assignments)
        except Exception as exc:  # plotting should not crash the run
            logging.warning("Plotting failed: %s", exc)

    return 0


def _parse_args(argv=None) -> Config:
    parser = argparse.ArgumentParser(description="1D semantic-value clustering (refactored)")
    parser.add_argument("--n-clusters", "-k", type=int, default=2, help="Number of clusters")
    parser.add_argument("--init", "-i", choices=["linear", "sample", "uniform"], default="linear", help="Initialization method")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed")
    parser.add_argument("--csv", type=str, default=None, help="Path to CSV with height/weight")
    parser.add_argument("--max-iter", type=int, default=100, help="Maximum iterations")
    parser.add_argument("--tol", type=float, default=1e-6, help="Convergence tolerance")
    parser.add_argument("--no-plot", action="store_true", help="Do not show plots")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose debug logging")
    args = parser.parse_args(argv)
    return Config(
        n_clusters=args.n_clusters,
        init_method=args.init,
        seed=args.seed,
        max_iter=args.max_iter,
        tol=args.tol,
        csv_path=args.csv,
        no_plot=args.no_plot,
        verbose=args.verbose,
    )


def main() -> None:
    cfg = _parse_args()
    rc = run(cfg)
    if rc != 0:
        raise SystemExit(rc)


if __name__ == "__main__":
    main()
    
# Sửa đổi mã nguồn