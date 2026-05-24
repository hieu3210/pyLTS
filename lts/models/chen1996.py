"""Chen [1996] — Phương pháp Fuzzy Time Series cơ sở.

Tham khảo:
  Chen, S.-M. (1996). Forecasting enrollments based on fuzzy time series.
  Fuzzy Sets and Systems, 81(3), 311-319.

Thuật toán:
1. Phân chia [lb, ub] thành n khoảng đều nhau.
2. Midpoint của mỗi khoảng là giá trị đại diện.
3. Fuzzification: ánh xạ mỗi điểm → khoảng gần nhất.
4. Xây dựng Fuzzy Logical Relationship Groups (FLRGs).
5. Dự báo: trung bình các midpoint của RHS; nếu không có rule → self-midpoint.
"""

from __future__ import annotations

from lts.models.base import BaseForecaster, ForecastResult


class Chen1996(BaseForecaster):
    """Phương pháp dự báo FTS của Chen [1996] để so sánh baseline.

    Parameters
    ----------
    n_intervals : int
        Số khoảng bằng nhau để phân vùng universe of discourse.
        Mặc định 7 để so sánh trực tiếp với LTSModel có 7 từ.
    order : int
        Bậc mô hình (first-order, second-order, ...).
    """

    def __init__(self, n_intervals: int = 7, order: int = 1) -> None:
        self.n_intervals = n_intervals
        self.order = order

        self._midpoints: list[float] = []
        self._labels: list[int] = []
        self._rules: dict[tuple[int, ...], list[int]] = {}
        self._forecasted: list[float] = []
        self._data: list[float] = []
        self._lb: float = 0.0
        self._ub: float = 1.0

    def _build_midpoints(self) -> list[float]:
        width = (self._ub - self._lb) / self.n_intervals
        return [self._lb + (i + 0.5) * width for i in range(self.n_intervals)]

    def _fuzzify_value(self, value: float) -> int:
        return min(range(self.n_intervals), key=lambda i: abs(self._midpoints[i] - value))

    def fit(self, data: list[float], lb: float, ub: float) -> None:
        self._data = list(data)
        self._lb = lb
        self._ub = ub
        self._midpoints = self._build_midpoints()
        self._labels = [self._fuzzify_value(x) for x in data]

        self._rules = {}
        for i in range(self.order, len(data)):
            lhs = tuple(self._labels[i - self.order : i])
            rhs = self._labels[i]
            if lhs not in self._rules:
                self._rules[lhs] = [rhs]
            elif rhs not in self._rules[lhs]:
                self._rules[lhs].append(rhs)

        self._forecasted = []
        for i in range(self.order, len(data)):
            lhs = tuple(self._labels[i - self.order : i])
            if lhs in self._rules:
                consequents = self._rules[lhs]
                self._forecasted.append(
                    sum(self._midpoints[j] for j in consequents) / len(consequents)
                )
            else:
                self._forecasted.append(self._midpoints[lhs[-1]])

    def predict(self) -> list[float]:
        return list(self._forecasted)

    def get_result(self) -> ForecastResult:
        return ForecastResult(
            data=list(self._data),
            labels=[str(l) for l in self._labels],
            semantic_points={str(i): m for i, m in enumerate(self._midpoints)},
            rules={lhs: [str(r) for r in rhs] for lhs, rhs in self._rules.items()},
            forecasted=list(self._forecasted),
            order=self.order,
            lb=self._lb,
            ub=self._ub,
        )
