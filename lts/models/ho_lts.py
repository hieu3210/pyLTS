"""High-Order LTS Model (HO-LTS).

Nguyen Duy Hieu (2021).
"High-Order Linguistic Time Series Forecasting Based on Hedge Algebras."

Key difference from LTSModel (2020):
- Supports order λ = 1..9
- Fallback rule: mean of ALL LHS words' semantic points (not just the last one)
"""

from __future__ import annotations

from lts.core.hedge_algebras import HAParams
from lts.models.lts_model import LTSModel


class HOLTSModel(LTSModel):
    """HO-LTS: High-Order LTS with mean-of-LHS fallback.

    Parameters
    ----------
    params : HAParams
        Tham số fuzziness (theta, alpha).
    order : int
        Bậc của mô hình, λ = 1..9.
    specificity : int
        Mức từ vựng: 1→7, 2→17, 3→33, 4→65 từ.
    use_repeat : bool
        Cho phép RHS trùng lặp trong LLRG.
    words : list[str] | None
        Danh sách từ tường minh; nếu None thì sinh từ specificity.
    """

    def __init__(
        self,
        params: HAParams,
        order: int = 2,
        specificity: int = 2,
        use_repeat: bool = False,
        words: list[str] | None = None,
    ) -> None:
        super().__init__(
            params=params,
            specificity=specificity,
            order=order,
            use_repeat=use_repeat,
            words=words,
        )

    def _forecast_one(
        self,
        lhs: tuple[str, ...],
        rules: dict[tuple[str, ...], list[str]],
        semantic_points: dict[str, float],
    ) -> float:
        """Dự báo cho một trạng thái LHS.

        - Nếu có LLRG: trả về trung bình semantic points của các RHS.
        - Fallback (HO-LTS): trung bình semantic points của TẤT CẢ từ trong LHS.
        """
        if lhs in rules:
            consequents = rules[lhs]
            return sum(semantic_points[w] for w in consequents) / len(consequents)
        # HO-LTS fallback: mean of all LHS words
        return sum(semantic_points[w] for w in lhs) / len(lhs)

    # Named parameter presets matching paper (Table 3 — hieund_2021)
    @staticmethod
    def params_for_specificity(specificity: int) -> HAParams:
        """Trả về HAParams từ bài báo 2021 theo specificity.

        specificity=1 (7 từ)  → theta=0.437, alpha=0.511
        specificity=2 (15 từ) → theta=0.527, alpha=0.412
        specificity=3 (31 từ) → theta=0.65,  alpha=0.35
        specificity=4 (63 từ) → theta=0.65,  alpha=0.35

        Lưu ý: bài báo dùng 9/17/33/65 từ do cấu trúc HA hơi khác;
        các tham số này là xấp xỉ tốt nhất cho cấu trúc HA hiện tại.
        """
        table = {
            1: HAParams(theta=0.437, alpha=0.511),
            2: HAParams(theta=0.527, alpha=0.412),
            3: HAParams(theta=0.65,  alpha=0.35),
            4: HAParams(theta=0.65,  alpha=0.35),
        }
        if specificity not in table:
            raise ValueError(
                f"No preset for specificity={specificity}; choose from {list(table)}"
            )
        return table[specificity]

    @staticmethod
    def params_for_word_count(n_words: int) -> HAParams:
        """Tương thích ngược: map số từ gần đúng → params_for_specificity.

        Bảng ánh xạ: 7→spec1, 15→spec2, 31→spec3, 63→spec4.
        """
        mapping = {7: 1, 15: 2, 31: 3, 63: 4}
        if n_words not in mapping:
            raise ValueError(
                f"No preset for {n_words} words; supported: {list(mapping)}"
            )
        return HOLTSModel.params_for_specificity(mapping[n_words])
