"""CO-LTS: Co-Optimization của tham số HA và tập từ vựng bằng PSO lồng nhau.

Nguyen Duy Hieu (2023).
"Co-Optimization of Hedge Algebra Parameters and Word-Set Selection
for Linguistic Time Series Forecasting."

Kiến trúc:
- Outer PSO: tối ưu (theta, alpha) — N=20 particles, G_max=30 cycles.
- Inner PSO (UWO): tối ưu chọn d_w từ từ tập W_all — M=30 particles, G_wmax=100 cycles.
- Word encoding: continuous w_j ∈ [0,1] → floor(w_j × |W_all|) → index trong W_all.
- Chạy n_runs lần, lấy MSE nhỏ nhất.

Variants:
- COLTS3: k_max=3, d_w=7
- COLTS4: k_max=4, d_w=14
- COLTS5: k_max=5, d_w=16
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from lts.core.hedge_algebras import HAParams, HedgeAlgebra
from lts.metrics.measures import ForecastMetrics
from lts.models.base import BaseForecaster, ForecastResult
from lts.optimization.pso import PSO, PSOConfig


@dataclass
class COLTSConfig:
    """Cấu hình cho CO-LTS model.

    Parameters
    ----------
    k_max : int
        Độ sâu hedge tối đa để sinh tập từ W_all.
        3 → COLTS3, 4 → COLTS4, 5 → COLTS5.
    d_w : int
        Số lượng từ được chọn từ W_all.
        7 (COLTS3), 14 (COLTS4), 16 (COLTS5) theo bài báo.
    outer_n : int
        Số particles outer PSO (N=20).
    outer_max_iter : int
        Số vòng lặp outer PSO (G_max=30).
    inner_m : int
        Số particles inner PSO (M=30).
    inner_max_iter : int
        Số vòng lặp inner PSO (G_wmax=100).
    omega, c1, c2 : float
        Tham số PSO (ω=0.4, c1=c2=2.0).
    n_runs : int
        Số lần chạy toàn bộ co-optimization; lấy MSE nhỏ nhất.
    order : int
        Bậc mô hình LTS (mặc định 1).
    """

    k_max: int = 3
    d_w: int = 7
    outer_n: int = 20
    outer_max_iter: int = 30
    inner_m: int = 30
    inner_max_iter: int = 100
    omega: float = 0.4
    c1: float = 2.0
    c2: float = 2.0
    n_runs: int = 5
    order: int = 1

    @classmethod
    def colts3(cls) -> "COLTSConfig":
        """COLTS3: k_max=3, d_w=7."""
        return cls(k_max=3, d_w=7)

    @classmethod
    def colts4(cls) -> "COLTSConfig":
        """COLTS4: k_max=4, d_w=14."""
        return cls(k_max=4, d_w=14)

    @classmethod
    def colts5(cls) -> "COLTSConfig":
        """COLTS5: k_max=5, d_w=16."""
        return cls(k_max=5, d_w=16)


def _all_words_up_to_depth(params: HAParams, k_max: int) -> list[str]:
    """Sinh tất cả từ với hedge depth tối đa k_max.

    Tương đương get_words(specificity) nhưng giới hạn bởi k_max thay vì specificity.
    """
    ha = HedgeAlgebra(params)
    # specificity k_max tương đương hedge depth k_max
    # get_words(s) sinh từ với s lớp hedge. Mỗi lớp nhân 2 từ + 1 (W)
    # specificity = k_max là mapping trực tiếp
    return ha.get_words(k_max)


def _decode_words(
    particle: list[float], word_pool: list[str], d_w: int
) -> list[str]:
    """Chuyển particle (d_w giá trị [0,1]) → d_w từ duy nhất từ word_pool.

    Encoding: index_j = floor(p_j × |word_pool|), clamp to [0, |word_pool|-1].
    Nếu có trùng, lấy unique theo thứ tự xuất hiện.
    Nếu thiếu (unique < d_w), bổ sung từ đầu word_pool chưa được chọn.
    """
    n = len(word_pool)
    indices = [min(int(math.floor(p * n)), n - 1) for p in particle]
    selected: list[str] = []
    seen: set[int] = set()
    for idx in indices:
        if idx not in seen:
            selected.append(word_pool[idx])
            seen.add(idx)

    if len(selected) < d_w:
        for i, w in enumerate(word_pool):
            if i not in seen:
                selected.append(w)
                seen.add(i)
            if len(selected) == d_w:
                break

    return selected[:d_w]


def _fit_and_score(
    params: HAParams,
    words: list[str],
    data: list[float],
    lb: float,
    ub: float,
    order: int,
) -> float:
    """Fit LTS với params và word subset, trả về MSE."""
    from lts.models.lts_model import LTSModel

    try:
        m = LTSModel(params, order=order, use_repeat=False, words=words)
        m.fit(data, lb, ub)
        forecasted = m.predict()
        actual = data[order:]
        return ForecastMetrics.compute(actual, forecasted).mse
    except Exception:
        return float("inf")


class COLTSModel(BaseForecaster):
    """CO-LTS: Co-Optimization mô hình dự báo LTS.

    Dùng PSO lồng nhau để đồng thời tối ưu:
    1. Tham số HA (theta, alpha) — outer PSO.
    2. Tập từ vựng tối ưu từ W_all — inner PSO (UWO).
    """

    def __init__(self, config: COLTSConfig | None = None) -> None:
        self.config = config or COLTSConfig()

        self._data: list[float] = []
        self._lb: float = 0.0
        self._ub: float = 1.0
        self._best_params: HAParams | None = None
        self._best_words: list[str] = []
        self._semantic_points: dict[str, float] = {}
        self._labels: list[str] = []
        self._rules: dict[tuple[str, ...], list[str]] = {}
        self._forecasted: list[float] = []
        self._best_mse: float = float("inf")

    def _run_once(self, data: list[float], lb: float, ub: float, seed: int) -> float:
        """Một lần chạy co-optimization. Trả về MSE tốt nhất."""
        cfg = self.config

        def outer_objective(outer_pos: list[float]) -> float:
            theta, alpha = outer_pos
            try:
                params = HAParams(theta=theta, alpha=alpha)
            except Exception:
                return float("inf")

            word_pool = _all_words_up_to_depth(params, cfg.k_max)
            if len(word_pool) < cfg.d_w:
                return float("inf")

            # Inner PSO: tìm d_w từ tốt nhất từ word_pool
            def inner_objective(inner_pos: list[float]) -> float:
                words = _decode_words(inner_pos, word_pool, cfg.d_w)
                return _fit_and_score(params, words, data, lb, ub, cfg.order)

            inner_cfg = PSOConfig(
                n_particles=cfg.inner_m,
                max_iter=cfg.inner_max_iter,
                omega=cfg.omega,
                c1=cfg.c1,
                c2=cfg.c2,
                bounds=[(0.0, 1.0)] * cfg.d_w,
                seed=seed,
            )
            _, inner_best_val = PSO(inner_objective, inner_cfg).run()
            return inner_best_val

        outer_cfg = PSOConfig(
            n_particles=cfg.outer_n,
            max_iter=cfg.outer_max_iter,
            omega=cfg.omega,
            c1=cfg.c1,
            c2=cfg.c2,
            bounds=[(0.3, 0.7), (0.3, 0.7)],
            seed=seed,
        )
        outer_best_pos, outer_best_val = PSO(outer_objective, outer_cfg).run()

        # Lưu lại best solution từ lần chạy này nếu tốt hơn global best
        if outer_best_val < self._best_mse:
            theta, alpha = outer_best_pos
            try:
                best_params = HAParams(theta=theta, alpha=alpha)
                word_pool = _all_words_up_to_depth(best_params, cfg.k_max)

                # Re-run inner PSO để lấy best words
                def inner_obj_final(inner_pos: list[float]) -> float:
                    words = _decode_words(inner_pos, word_pool, cfg.d_w)
                    return _fit_and_score(
                        best_params, words, data, lb, ub, cfg.order
                    )

                inner_cfg_final = PSOConfig(
                    n_particles=cfg.inner_m,
                    max_iter=cfg.inner_max_iter,
                    omega=cfg.omega,
                    c1=cfg.c1,
                    c2=cfg.c2,
                    bounds=[(0.0, 1.0)] * cfg.d_w,
                    seed=seed + 1000,
                )
                inner_best_inner, _ = PSO(inner_obj_final, inner_cfg_final).run()
                best_words = _decode_words(inner_best_inner, word_pool, cfg.d_w)

                self._best_mse = outer_best_val
                self._best_params = best_params
                self._best_words = best_words
            except Exception:
                pass

        return outer_best_val

    # ------------------------------------------------------------------
    # BaseForecaster interface
    # ------------------------------------------------------------------
    def fit(self, data: list[float], lb: float, ub: float) -> None:
        """Chạy co-optimization n_runs lần, lấy kết quả MSE nhỏ nhất."""
        self._data = list(data)
        self._lb = lb
        self._ub = ub
        self._best_mse = float("inf")
        self._best_params = None
        self._best_words = []

        for run in range(self.config.n_runs):
            self._run_once(data, lb, ub, seed=run * 37)

        if self._best_params is None or not self._best_words:
            # Fallback nếu PSO thất bại hoàn toàn
            self._best_params = HAParams(theta=0.5, alpha=0.5)
            ha = HedgeAlgebra(self._best_params)
            self._best_words = ha.get_words(self.config.k_max)[: self.config.d_w]

        # Fit final model
        from lts.models.lts_model import LTSModel

        final = LTSModel(
            self._best_params,
            order=self.config.order,
            use_repeat=False,
            words=self._best_words,
        )
        final.fit(data, lb, ub)

        self._semantic_points = final.semantic_points
        self._labels = final.labels
        self._rules = final.rules
        self._forecasted = final.predict()

    def predict(self) -> list[float]:
        return list(self._forecasted)

    def get_result(self) -> ForecastResult:
        return ForecastResult(
            data=list(self._data),
            labels=list(self._labels),
            semantic_points=dict(self._semantic_points),
            rules={lhs: list(rhs) for lhs, rhs in self._rules.items()},
            forecasted=list(self._forecasted),
            order=self.config.order,
            lb=self._lb,
            ub=self._ub,
        )

    @property
    def best_params(self) -> HAParams | None:
        """Tham số HA tối ưu sau khi fit."""
        return self._best_params

    @property
    def best_words(self) -> list[str]:
        """Tập từ vựng tối ưu sau khi fit."""
        return list(self._best_words)

    @property
    def best_mse(self) -> float:
        """MSE tốt nhất đạt được qua PSO."""
        return self._best_mse
