"""Song & Chissom [1993] — Phương pháp Fuzzy Time Series gốc.

Tham khảo:
  Song, Q., & Chissom, B. S. (1993). Forecasting enrollments with fuzzy time series
  — Part I. Fuzzy Sets and Systems, 54(1), 1-9.

Thuật toán:
1. Phân chia [lb, ub] thành n khoảng đều nhau với membership function tam giác.
2. Fuzzification: tính vector membership cho mỗi điểm dữ liệu.
3. Xây dựng ma trận quan hệ mờ R qua max-min composition.
4. Dự báo: F(t) = F(t-1) ◦ R (max-min composition).
5. Defuzzification: lấy midpoint của khoảng có membership cao nhất.
"""

from __future__ import annotations

from lts.models.base import BaseForecaster, ForecastResult


class SongChissom1993(BaseForecaster):
    """Phương pháp FTS gốc của Song & Chissom [1993] để so sánh baseline.

    Parameters
    ----------
    n_intervals : int
        Số khoảng đều nhau. Mặc định 7.
    """

    def __init__(self, n_intervals: int = 7) -> None:
        self.n_intervals = n_intervals
        self._data: list[float] = []
        self._lb: float = 0.0
        self._ub: float = 1.0
        self._midpoints: list[float] = []
        self._width: float = 0.0
        self._R: list[list[float]] = []
        self._labels: list[int] = []
        self._forecasted: list[float] = []

    def _build_midpoints(self) -> list[float]:
        self._width = (self._ub - self._lb) / self.n_intervals
        return [self._lb + (i + 0.5) * self._width for i in range(self.n_intervals)]

    def _membership(self, value: float, idx: int) -> float:
        """Membership function tam giác centered tại midpoints[idx]."""
        center = self._midpoints[idx]
        w = self._width
        dist = abs(value - center)
        if dist >= w:
            return 0.0
        return 1.0 - dist / w

    def _fuzzify_vector(self, value: float) -> list[float]:
        """Trả về vector membership n chiều cho value."""
        vec = [self._membership(value, i) for i in range(self.n_intervals)]
        # Chuẩn hóa: đảm bảo ít nhất 1 phần tử = 1.0 (khoảng gần nhất)
        max_m = max(vec)
        if max_m == 0.0:
            best = min(range(self.n_intervals), key=lambda i: abs(self._midpoints[i] - value))
            vec[best] = 1.0
        return vec

    def _argmax(self, vec: list[float]) -> int:
        return max(range(len(vec)), key=lambda i: vec[i])

    def _defuzzify(self, fuzz_vec: list[float]) -> float:
        """Lấy midpoint của khoảng có membership cao nhất."""
        return self._midpoints[self._argmax(fuzz_vec)]

    def _max_min_compose(
        self, fuzz_vec: list[float], R: list[list[float]]
    ) -> list[float]:
        """F(t) = F(t-1) ◦ R dùng toán tử max-min."""
        n = self.n_intervals
        result = [0.0] * n
        for j in range(n):
            result[j] = max(min(fuzz_vec[i], R[i][j]) for i in range(n))
        return result

    def _build_relation_matrix(
        self, fuzz_vecs: list[list[float]]
    ) -> list[list[float]]:
        """Xây dựng ma trận quan hệ mờ R bằng max-union của outer products."""
        n = self.n_intervals
        R = [[0.0] * n for _ in range(n)]
        for t in range(len(fuzz_vecs) - 1):
            A = fuzz_vecs[t]
            B = fuzz_vecs[t + 1]
            # Outer product dùng min; union dùng max
            for i in range(n):
                for j in range(n):
                    R[i][j] = max(R[i][j], min(A[i], B[j]))
        return R

    def fit(self, data: list[float], lb: float, ub: float) -> None:
        self._data = list(data)
        self._lb = lb
        self._ub = ub
        self._midpoints = self._build_midpoints()

        fuzz_vecs = [self._fuzzify_vector(x) for x in data]
        self._labels = [self._argmax(v) for v in fuzz_vecs]
        self._R = self._build_relation_matrix(fuzz_vecs)

        self._forecasted = []
        for t in range(1, len(data)):
            composed = self._max_min_compose(fuzz_vecs[t - 1], self._R)
            self._forecasted.append(self._defuzzify(composed))

    def predict(self) -> list[float]:
        return list(self._forecasted)

    def get_result(self) -> ForecastResult:
        return ForecastResult(
            data=list(self._data),
            labels=[str(l) for l in self._labels],
            semantic_points={str(i): m for i, m in enumerate(self._midpoints)},
            rules={},
            forecasted=list(self._forecasted),
            order=1,
            lb=self._lb,
            ub=self._ub,
        )
