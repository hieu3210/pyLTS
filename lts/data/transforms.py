"""Biến đổi dataset cho các chế độ mô hình khác nhau."""

from __future__ import annotations

from lts.data.loader import Dataset


class DataTransformer:
    """Các phép biến đổi chuỗi thời gian."""

    @staticmethod
    def difference_series(dataset: Dataset) -> tuple[list[float], list[float]]:
        """Tính chuỗi sai phân bậc 1 (variation series).

        Returns
        -------
        tuple[list[float], list[float]]
            (original_values, variations) với:
            variations[i] = values[i+1] - values[i]
        """
        values = dataset.values
        variations = [values[i + 1] - values[i] for i in range(len(values) - 1)]
        return values, variations

    @staticmethod
    def auto_bounds(values: list[float], margin: float = 0.0) -> tuple[float, float]:
        """Tính lb/ub từ data với margin tuỳ chọn."""
        return min(values) - margin, max(values) + margin

    @staticmethod
    def normalize(
        values: list[float], lb: float, ub: float
    ) -> list[float]:
        """Chuẩn hóa values về [0, 1] theo [lb, ub]."""
        span = ub - lb
        if span == 0:
            raise ValueError("lb == ub, không thể chuẩn hóa.")
        return [(v - lb) / span for v in values]
