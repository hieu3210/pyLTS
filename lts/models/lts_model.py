"""Mô hình Linguistic Time Series (LTS) — thuật toán 6 bước từ bài báo.

Nguyen Duy Hieu, Nguyen Cat Ho, Vu Nhu Lan (2020).
"Enrollment Forecasting Based on Linguistic Time Series."
Journal of Computer Science and Cybernetics, V.36, N.2.
"""

from __future__ import annotations

from lts.core.hedge_algebras import HAParams, HedgeAlgebra
from lts.models.base import BaseForecaster, ForecastResult


class LTSModel(BaseForecaster):
    """Mô hình dự báo Linguistic Time Series dựa trên Hedge Algebras.

    Implements chính xác thuật toán 6 bước từ Section 4.1 của bài báo.

    Parameters
    ----------
    params : HAParams
        Tham số fuzziness (theta, alpha).
    specificity : int
        Mức độ từ vựng. 1 → 7 từ, 2 → 17 từ, 3 → 33 từ, ...
    order : int
        Bậc mô hình LTS (1 = first-order, 2 = second-order, ...).
    use_repeat : bool
        True: cho phép các RHS trùng lặp trong cùng LLRG (weighted).
        False: mỗi cặp LHS-RHS chỉ xuất hiện một lần.
    words : list[str] | None
        Danh sách từ tường minh; nếu None thì tự sinh từ specificity.
    """

    def __init__(
        self,
        params: HAParams,
        specificity: int = 1,
        order: int = 1,
        use_repeat: bool = True,
        words: list[str] | None = None,
    ) -> None:
        self.params = params
        self.specificity = specificity
        self.order = order
        self.use_repeat = use_repeat

        self.ha = HedgeAlgebra(params)
        self._words: list[str] = (
            words if words is not None else self.ha.get_words(specificity)
        )
        # Chỉ loại bỏ các boundary element tuyệt đối ("0", "1")
        # "W" (Middle/neutral) được giữ lại như một từ vựng hợp lệ
        self._vocab: list[str] = [w for w in self._words if w not in {"0", "1"}]

        # State được điền bởi fit()
        self._data: list[float] = []
        self._lb: float = 0.0
        self._ub: float = 1.0
        self._semantic_points: dict[str, float] = {}
        self._labels: list[str] = []
        self._rules: dict[tuple[str, ...], list[str]] = {}
        self._forecasted: list[float] = []

    # ------------------------------------------------------------------
    # Bước 2: Tính SQM → [0, 1]
    # ------------------------------------------------------------------
    def _compute_normalized_semantics(self) -> dict[str, float]:
        return {w: self.ha.sqm(w) for w in self._vocab}

    # ------------------------------------------------------------------
    # Bước 3: Ánh xạ [0, 1] → [lb, ub]
    # ------------------------------------------------------------------
    def _compute_semantic_points(self) -> dict[str, float]:
        norm = self._compute_normalized_semantics()
        span = self._ub - self._lb
        return {w: self._lb + span * v for w, v in norm.items()}

    # ------------------------------------------------------------------
    # Bước 4: Fuzzification — gán nhãn ngôn ngữ cho từng điểm dữ liệu
    # ------------------------------------------------------------------
    def _fuzzify(self, value: float, semantic_points: dict[str, float]) -> str:
        """Trả về word có semantic point gần value nhất."""
        return min(semantic_points, key=lambda w: abs(semantic_points[w] - value))

    def _label_data(
        self, data: list[float], semantic_points: dict[str, float]
    ) -> list[str]:
        return [self._fuzzify(x, semantic_points) for x in data]

    # ------------------------------------------------------------------
    # Bước 5: Xây dựng LLRs và gom thành LLRGs
    # ------------------------------------------------------------------
    def _build_rules(
        self,
        labels: list[str],
        order: int,
        use_repeat: bool,
    ) -> dict[tuple[str, ...], list[str]]:
        """Sinh các Linguistic Logical Relationship Groups.

        Returns
        -------
        dict[tuple[str, ...], list[str]]
            Ánh xạ lhs_tuple → [rhs_word, ...]
        """
        rules: dict[tuple[str, ...], list[str]] = {}
        for i in range(order, len(labels)):
            lhs = tuple(labels[i - order : i])
            rhs = labels[i]
            if lhs not in rules:
                rules[lhs] = [rhs]
            elif use_repeat:
                rules[lhs].append(rhs)
            elif rhs not in rules[lhs]:
                rules[lhs].append(rhs)
        return rules

    # ------------------------------------------------------------------
    # Bước 6: Dự báo
    # ------------------------------------------------------------------
    def _forecast_one(
        self,
        lhs: tuple[str, ...],
        rules: dict[tuple[str, ...], list[str]],
        semantic_points: dict[str, float],
    ) -> float:
        """Dự báo cho một trạng thái LHS.

        - Nếu có LLRG: trả về trung bình các semantic point của RHS.
        - Nếu không có rule: trả về semantic point của từ cuối trong LHS.
        """
        if lhs in rules:
            consequents = rules[lhs]
            return sum(semantic_points[w] for w in consequents) / len(consequents)
        return semantic_points[lhs[-1]]

    # ------------------------------------------------------------------
    # BaseForecaster interface
    # ------------------------------------------------------------------
    def fit(self, data: list[float], lb: float, ub: float) -> None:
        """Chạy toàn bộ pipeline 6 bước và lưu kết quả."""
        self._data = list(data)
        self._lb = lb
        self._ub = ub

        self._semantic_points = self._compute_semantic_points()          # Bước 2+3
        self._labels = self._label_data(self._data, self._semantic_points)  # Bước 4
        self._rules = self._build_rules(self._labels, self.order, self.use_repeat)  # Bước 5

        self._forecasted = [
            self._forecast_one(
                tuple(self._labels[i - self.order : i]),
                self._rules,
                self._semantic_points,
            )
            for i in range(self.order, len(self._data))
        ]  # Bước 6

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

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------
    @property
    def words(self) -> list[str]:
        """Danh sách từ vựng đang dùng (đã sắp xếp theo SQM)."""
        return list(self._vocab)

    @property
    def labels(self) -> list[str]:
        """Nhãn ngôn ngữ của chuỗi dữ liệu (sau khi fit)."""
        return list(self._labels)

    @property
    def semantic_points(self) -> dict[str, float]:
        """Ánh xạ word → semantic point trong [lb, ub] (sau khi fit)."""
        return dict(self._semantic_points)

    @property
    def rules(self) -> dict[tuple[str, ...], list[str]]:
        """LLRGs (sau khi fit)."""
        return {lhs: list(rhs) for lhs, rhs in self._rules.items()}
