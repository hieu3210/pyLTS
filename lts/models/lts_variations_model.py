"""Mô hình LTS trên chuỗi sai phân (variation series) — Section 4.2 bài báo."""

from __future__ import annotations

from lts.core.hedge_algebras import HAParams
from lts.models.base import ForecastResult
from lts.models.lts_model import LTSModel


class LTSVariationsModel:
    """Dự báo LTS trên chuỗi sai phân bậc 1 (Hwang-style variant).

    Mô hình hoạt động theo 3 giai đoạn:
    1. Tính chuỗi variation từ dữ liệu gốc.
    2. Fit LTSModel trên chuỗi variation với universe of discourse riêng.
    3. Tái tạo giá trị dự báo gốc bằng cách cộng variation dự báo
       vào giá trị thực tế trước đó.

    Parameters
    ----------
    params : HAParams
        theta=0.55, alpha=0.52 theo bài báo Section 4.2.
    lb_variation : float
        Cận dưới universe of variation (vd: -1000).
    ub_variation : float
        Cận trên universe of variation (vd: +1400).
    specificity : int
    order : int
    use_repeat : bool
    """

    def __init__(
        self,
        params: HAParams,
        lb_variation: float,
        ub_variation: float,
        specificity: int = 1,
        order: int = 1,
        use_repeat: bool = True,
    ) -> None:
        self.params = params
        self.lb_variation = lb_variation
        self.ub_variation = ub_variation
        self.order = order

        self._inner = LTSModel(
            params=params,
            specificity=specificity,
            order=order,
            use_repeat=use_repeat,
        )

        self._original_data: list[float] = []
        self._variations: list[float] = []
        self._forecasted_enrollments: list[float] = []

    def fit(self, data: list[float]) -> None:
        """Tính variation series và fit mô hình bên trong.

        Parameters
        ----------
        data : list[float]
            Chuỗi dữ liệu gốc (vd: enrollment 1971-1992).
        """
        self._original_data = list(data)
        self._variations = [
            data[i] - data[i - 1] for i in range(1, len(data))
        ]
        self._inner.fit(self._variations, self.lb_variation, self.ub_variation)

        # Tái tạo enrollment từ variation dự báo + actual trước đó
        forecasted_variations = self._inner.predict()
        order = self.order
        self._forecasted_enrollments = [
            data[i] + forecasted_variations[i - order]
            for i in range(order, len(data) - 1)
        ]

    def predict(self) -> list[float]:
        """Trả về giá trị enrollment dự báo."""
        return list(self._forecasted_enrollments)

    def get_variation_result(self) -> ForecastResult:
        """Trả về ForecastResult của mô hình bên trong (trên chuỗi variation)."""
        return self._inner.get_result()

    @property
    def variations(self) -> list[float]:
        """Chuỗi variation được tính từ dữ liệu gốc."""
        return list(self._variations)

    @property
    def actual_for_comparison(self) -> list[float]:
        """Giá trị thực tế tương ứng với khoảng dự báo."""
        return self._original_data[self.order + 1 :]
