"""LTS-PSO: LTS với tối ưu tham số (theta, alpha) bằng PSO.

Nguyen Duy Hieu (2022).
"Linguistic Time Series Forecasting Based on Hedge Algebras and PSO."

Key differences from LTSModel (2020):
- Công thức dự báo mới:
    Có rule: forecast = 0.5 × (s_lhs_last + mean(s_RHS))
    Không có rule: forecast = s_lhs_last
- Tối ưu (theta, alpha) bằng PSO để minimize MSE trên training data.
- Mặc định dùng 17 từ (specificity=2), order=1.
"""

from __future__ import annotations

from lts.core.hedge_algebras import HAParams, HedgeAlgebra
from lts.metrics.measures import ForecastMetrics
from lts.models.base import BaseForecaster, ForecastResult
from lts.optimization.pso import PSO, PSOConfig


class LTSPSOModel(BaseForecaster):
    """LTS-PSO: mô hình LTS với công thức dự báo cải tiến và PSO tối ưu tham số.

    Parameters
    ----------
    params : HAParams
        Tham số fuzziness khởi đầu (theta, alpha).
    specificity : int
        Mức từ vựng. 2 → 17 từ (mặc định theo bài báo).
    order : int
        Bậc mô hình (mặc định 1 theo bài báo).
    words : list[str] | None
        Danh sách từ tường minh; nếu None thì sinh từ specificity.
    """

    def __init__(
        self,
        params: HAParams,
        specificity: int = 2,
        order: int = 1,
        words: list[str] | None = None,
    ) -> None:
        self.params = params
        self.specificity = specificity
        self.order = order

        self._words_explicit = words

        self._data: list[float] = []
        self._lb: float = 0.0
        self._ub: float = 1.0
        self._semantic_points: dict[str, float] = {}
        self._labels: list[str] = []
        self._rules: dict[tuple[str, ...], list[str]] = {}
        self._forecasted: list[float] = []
        self._vocab: list[str] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_vocab(self, params: HAParams) -> list[str]:
        ha = HedgeAlgebra(params)
        words = (
            self._words_explicit
            if self._words_explicit is not None
            else ha.get_words(self.specificity)
        )
        return [w for w in words if w not in {"0", "1"}]

    def _compute_semantic_points(
        self, params: HAParams, vocab: list[str]
    ) -> dict[str, float]:
        ha = HedgeAlgebra(params)
        span = self._ub - self._lb
        return {w: self._lb + span * ha.sqm(w) for w in vocab}

    def _fuzzify(self, value: float, semantic_points: dict[str, float]) -> str:
        return min(semantic_points, key=lambda w: abs(semantic_points[w] - value))

    def _label_data(
        self, data: list[float], semantic_points: dict[str, float]
    ) -> list[str]:
        return [self._fuzzify(x, semantic_points) for x in data]

    def _build_rules(
        self, labels: list[str]
    ) -> dict[tuple[str, ...], list[str]]:
        """Unique-RHS LLRGs (không lặp RHS giống nhau)."""
        rules: dict[tuple[str, ...], list[str]] = {}
        for i in range(self.order, len(labels)):
            lhs = tuple(labels[i - self.order : i])
            rhs = labels[i]
            if lhs not in rules:
                rules[lhs] = [rhs]
            elif rhs not in rules[lhs]:
                rules[lhs].append(rhs)
        return rules

    def _forecast_one(
        self,
        lhs: tuple[str, ...],
        rules: dict[tuple[str, ...], list[str]],
        semantic_points: dict[str, float],
    ) -> float:
        """Công thức dự báo LTS-PSO (Section 3, bài báo 2022).

        Có rule: 0.5 × (s_lhs_last + mean(s_RHS))
        Không có rule: s_lhs_last
        """
        s_current = semantic_points[lhs[-1]]
        if lhs in rules:
            consequents = rules[lhs]
            s_rhs_mean = sum(semantic_points[w] for w in consequents) / len(consequents)
            return 0.5 * (s_current + s_rhs_mean)
        return s_current

    def _run_forecast(
        self, params: HAParams, data: list[float], lb: float, ub: float
    ) -> tuple[dict[str, float], list[str], dict[tuple[str, ...], list[str]], list[float]]:
        """Thực hiện toàn bộ pipeline, trả về (semantic_points, labels, rules, forecasted)."""
        vocab = self._build_vocab(params)
        sp = self._compute_semantic_points(params, vocab)
        labels = self._label_data(data, sp)
        rules = self._build_rules(labels)
        forecasted = [
            self._forecast_one(tuple(labels[i - self.order : i]), rules, sp)
            for i in range(self.order, len(data))
        ]
        return sp, labels, rules, forecasted

    # ------------------------------------------------------------------
    # BaseForecaster interface
    # ------------------------------------------------------------------
    def fit(self, data: list[float], lb: float, ub: float) -> None:
        """Fit với params hiện tại (không PSO)."""
        self._data = list(data)
        self._lb = lb
        self._ub = ub
        self._vocab = self._build_vocab(self.params)
        sp, labels, rules, forecasted = self._run_forecast(
            self.params, data, lb, ub
        )
        self._semantic_points = sp
        self._labels = labels
        self._rules = rules
        self._forecasted = forecasted

    def fit_optimize(
        self,
        data: list[float],
        lb: float,
        ub: float,
        pso_config: PSOConfig | None = None,
        n_runs: int = 3,
    ) -> HAParams:
        """Chạy PSO để tìm (theta, alpha) tối ưu, sau đó fit với params đó.

        Parameters
        ----------
        data : list[float]
            Training data.
        lb, ub : float
            Universe of discourse.
        pso_config : PSOConfig | None
            Cấu hình PSO. Mặc định: N=300, G_max=1000, ω=0.4, c1=c2=2.0.
        n_runs : int
            Số lần chạy PSO; lấy kết quả MSE nhỏ nhất.

        Returns
        -------
        HAParams
            Tham số tối ưu đã dùng để fit.
        """
        self._data = list(data)
        self._lb = lb
        self._ub = ub

        if pso_config is None:
            pso_config = PSOConfig(
                n_particles=300,
                max_iter=1000,
                omega=0.4,
                c1=2.0,
                c2=2.0,
                bounds=[(0.3, 0.7), (0.3, 0.7)],
            )
        else:
            pso_config = PSOConfig(
                n_particles=pso_config.n_particles,
                max_iter=pso_config.max_iter,
                omega=pso_config.omega,
                c1=pso_config.c1,
                c2=pso_config.c2,
                bounds=[(0.3, 0.7), (0.3, 0.7)],
            )

        def objective(position: list[float]) -> float:
            theta, alpha = position
            try:
                params = HAParams(theta=theta, alpha=alpha)
                _, _, _, forecasted = self._run_forecast(params, data, lb, ub)
                actual = data[self.order :]
                return ForecastMetrics.compute(actual, forecasted).mse
            except Exception:
                return float("inf")

        best_pos: list[float] = [0.5, 0.5]
        best_val = float("inf")
        for run in range(n_runs):
            cfg = PSOConfig(
                n_particles=pso_config.n_particles,
                max_iter=pso_config.max_iter,
                omega=pso_config.omega,
                c1=pso_config.c1,
                c2=pso_config.c2,
                bounds=pso_config.bounds,
                seed=run,
            )
            pos, val = PSO(objective, cfg).run()
            if val < best_val:
                best_val = val
                best_pos = pos

        best_params = HAParams(theta=best_pos[0], alpha=best_pos[1])
        self.params = best_params
        self._vocab = self._build_vocab(best_params)
        sp, labels, rules, forecasted = self._run_forecast(best_params, data, lb, ub)
        self._semantic_points = sp
        self._labels = labels
        self._rules = rules
        self._forecasted = forecasted
        return best_params

    def predict(self) -> list[float]:
        return list(self._forecasted)

    def get_result(self) -> ForecastResult:
        return ForecastResult(
            data=list(self._data),
            labels=list(self._labels),
            semantic_points=dict(self._semantic_points),
            rules={lhs: list(rhs) for lhs, rhs in self._rules.items()},
            forecasted=list(self._forecasted),
            order=self.order,
            lb=self._lb,
            ub=self._ub,
        )

    @property
    def words(self) -> list[str]:
        return list(self._vocab)
