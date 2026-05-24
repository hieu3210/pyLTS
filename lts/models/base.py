"""Abstractions cơ sở cho các mô hình dự báo chuỗi thời gian ngôn ngữ."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ForecastResult:
    """Container chứa toàn bộ kết quả của một lần chạy mô hình.

    Attributes
    ----------
    data : list[float]
        Chuỗi dữ liệu gốc.
    labels : list[str]
        Nhãn ngôn ngữ tương ứng với mỗi điểm dữ liệu.
    semantic_points : dict[str, float]
        Ánh xạ word → giá trị ngữ nghĩa trong [lb, ub].
    rules : dict[tuple[str, ...], list[str]]
        Các Linguistic Logical Relationship Groups:
        lhs_tuple → [rhs_word, ...].
    forecasted : list[float]
        Giá trị dự báo (căn chỉnh với data[order:]).
    order : int
        Bậc của mô hình.
    lb : float
        Cận dưới universe of discourse.
    ub : float
        Cận trên universe of discourse.
    """

    data: list[float]
    labels: list[str]
    semantic_points: dict[str, float]
    rules: dict[tuple[str, ...], list[str]]
    forecasted: list[float]
    order: int
    lb: float = 0.0
    ub: float = 1.0

    @property
    def actual(self) -> list[float]:
        """Giá trị thực tế tương ứng với khoảng dự báo (data[order:])."""
        return self.data[self.order:]

    @property
    def n_rules(self) -> int:
        """Số lượng LLRG (linguistic logical relationship groups)."""
        return len(self.rules)

    def llrg_summary(self) -> list[str]:
        """Trả về danh sách mô tả các LLRG dạng 'X → X1, X2, ...'."""
        lines = []
        for lhs, rhs_list in sorted(self.rules.items()):
            lhs_str = ", ".join(lhs)
            rhs_str = ", ".join(rhs_list)
            lines.append(f"({lhs_str}) → {rhs_str}")
        return lines


class BaseForecaster(ABC):
    """Lớp trừu tượng cơ sở cho tất cả mô hình dự báo LTS.

    Mọi mô hình đều phải implement 3 phương thức:
    - fit(): nhận dữ liệu và chạy toàn bộ pipeline
    - predict(): trả về giá trị dự báo
    - get_result(): trả về ForecastResult đầy đủ
    """

    @abstractmethod
    def fit(self, data: list[float], lb: float, ub: float) -> None:
        """Nhập dữ liệu và chạy pipeline đầy đủ đến bước rule induction."""
        ...

    @abstractmethod
    def predict(self) -> list[float]:
        """Trả về danh sách giá trị dự báo."""
        ...

    @abstractmethod
    def get_result(self) -> ForecastResult:
        """Trả về ForecastResult đầy đủ sau khi đã fit()."""
        ...
