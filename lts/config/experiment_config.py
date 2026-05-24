"""Cấu hình thực nghiệm — mọi tham số ảnh hưởng đến kết quả đều ở đây."""

from __future__ import annotations

from dataclasses import dataclass, field

from lts.core.hedge_algebras import HAParams


@dataclass
class ExperimentConfig:
    """Cấu hình đầy đủ cho một thực nghiệm dự báo.

    Mỗi instance của class này đại diện cho một thực nghiệm hoàn toàn
    có thể tái hiện (reproducible). Mọi tham số ảnh hưởng đến kết quả
    đều được ghi nhận tại đây.

    Parameters
    ----------
    params : HAParams
        Tham số Hedge Algebra (theta, alpha).
    lb : float
        Cận dưới universe of discourse.
    ub : float
        Cận trên universe of discourse.
    order : int
        Bậc mô hình LTS.
    use_repeat : bool
        Cho phép RHS trùng lặp trong LLRG.
    specificity : int
        Mức từ vựng HA (1=7 từ, 2=17 từ, ...).
    words : list[str] | None
        Danh sách từ tường minh; nếu None thì sinh từ specificity.
    dataset_name : str
        Tên dataset bundled ('alabama', ...).
    dataset_path : str | None
        Đường dẫn tường minh tới file .txt; nếu None thì dùng bundled.
    use_variations : bool
        True → chạy LTSVariationsModel thay vì LTSModel.
    lb_variation : float | None
        Cận dưới universe of variation (khi use_variations=True).
    ub_variation : float | None
        Cận trên universe of variation (khi use_variations=True).
    mape_digits : int
        Số chữ số làm tròn cho MAPE.
    label : str
        Nhãn mô tả thực nghiệm (tuỳ chọn).
    """

    params: HAParams
    lb: float
    ub: float

    order: int = 1
    use_repeat: bool = True
    specificity: int = 1
    words: list[str] | None = None

    dataset_name: str = "alabama"
    dataset_path: str | None = None

    use_variations: bool = False
    lb_variation: float | None = None
    ub_variation: float | None = None

    mape_digits: int = 2
    label: str = ""

    # ------------------------------------------------------------------
    # Named presets tái hiện đúng các thực nghiệm trong bài báo
    # ------------------------------------------------------------------

    @classmethod
    def paper_table4(cls) -> "ExperimentConfig":
        """Tái hiện Table 4: enrollment, θ=0.57, α=0.49, use_repeat=False → MSE≈262.326×1000."""
        return cls(
            params=HAParams.enrollment(),
            lb=13000.0,
            ub=20000.0,
            order=1,
            use_repeat=False,
            specificity=1,
            dataset_name="alabama",
            label="Table 4 — Enrollment Forecasting (LTS)",
        )

    @classmethod
    def paper_table8(cls) -> "ExperimentConfig":
        """Tái hiện Table 8: variation series, θ=0.55, α=0.52 → MAPE=1.27%."""
        return cls(
            params=HAParams.variations(),
            lb=13000.0,
            ub=20000.0,
            order=1,
            use_repeat=True,
            specificity=1,
            use_variations=True,
            lb_variation=-1000.0,
            ub_variation=1400.0,
            dataset_name="alabama",
            label="Table 8 — Variation-Based Forecasting (LTS)",
        )

    @classmethod
    def chen_baseline(cls) -> "ExperimentConfig":
        """Cấu hình Chen [1996] để so sánh Table 9."""
        return cls(
            params=HAParams.enrollment(),
            lb=13000.0,
            ub=20000.0,
            order=1,
            specificity=1,
            dataset_name="alabama",
            label="Chen [1996] Baseline",
        )

    @classmethod
    def song_baseline(cls) -> "ExperimentConfig":
        """Cấu hình Song & Chissom [1993] để so sánh Table 9."""
        return cls(
            params=HAParams.enrollment(),
            lb=13000.0,
            ub=20000.0,
            order=1,
            specificity=1,
            dataset_name="alabama",
            label="Song & Chissom [1993] Baseline",
        )
