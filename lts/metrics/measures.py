"""Các độ đo đánh giá sai số dự báo chuỗi thời gian."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ForecastMetrics:
    """Container bất biến cho toàn bộ các độ đo sai số dự báo.

    Attributes
    ----------
    mae : float
        Mean Absolute Error.
    mse : float
        Mean Squared Error.
    rmse : float
        Root Mean Squared Error.
    mape : float
        Mean Absolute Percentage Error (%, ví dụ 1.27 tức là 1.27%).
    smape : float
        Symmetric MAPE (%).
    n : int
        Số điểm dự báo.
    """

    mae: float
    mse: float
    rmse: float
    mape: float
    smape: float
    n: int

    @classmethod
    def compute(
        cls,
        actual: list[float],
        forecasted: list[float],
        mape_digits: int = 2,
    ) -> "ForecastMetrics":
        """Tính toàn bộ các độ đo từ hai chuỗi actual và forecasted.

        Parameters
        ----------
        actual : list[float]
            Giá trị thực tế.
        forecasted : list[float]
            Giá trị dự báo (phải cùng độ dài với actual).
        mape_digits : int
            Số chữ số làm tròn cho MAPE, SMAPE.

        Raises
        ------
        ValueError
            Nếu hai danh sách không cùng độ dài, hoặc rỗng.
        """
        if len(actual) != len(forecasted):
            raise ValueError(
                f"Độ dài không khớp: actual={len(actual)}, "
                f"forecasted={len(forecasted)}"
            )
        n = len(actual)
        if n == 0:
            raise ValueError("Danh sách không được rỗng.")

        errors = [abs(f - a) for f, a in zip(forecasted, actual)]

        mae = sum(errors) / n
        mse = sum(e ** 2 for e in errors) / n
        rmse = math.sqrt(mse)

        mape = round(
            sum(e / abs(a) for e, a in zip(errors, actual)) / n * 100,
            mape_digits,
        )
        smape = round(
            sum(
                abs(f - a) / ((abs(a) + abs(f)) / 2)
                for f, a in zip(forecasted, actual)
            )
            / n
            * 100,
            mape_digits,
        )

        return cls(mae=mae, mse=mse, rmse=rmse, mape=mape, smape=smape, n=n)

    def __str__(self) -> str:
        return (
            f"ForecastMetrics(n={self.n}, "
            f"MAE={self.mae:.4f}, MSE={self.mse:.4f}, "
            f"RMSE={self.rmse:.4f}, MAPE={self.mape}%, SMAPE={self.smape}%)"
        )
