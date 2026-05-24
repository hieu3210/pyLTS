"""
Công thức SQM closed-form cho bộ từ vựng 7 từ (Specificity = 1).

Implements chính xác các phương trình (4.1)-(4.7) từ bài báo:
  "Enrollment Forecasting Based on Linguistic Time Series" (Nguyen Duy Hieu, 2020)

Bộ từ vựng 7 từ (sắp xếp theo thứ tự ngữ nghĩa tăng dần):
  "V-"  Very Small     (rất nhỏ)
  "-"   Small          (nhỏ)
  "L-"  Rather Small   (khá nhỏ)
  "W"   Middle         (trung bình)
  "L+"  Rather Large   (khá lớn)
  "+"   Large          (lớn)
  "V+"  Very Large     (rất lớn)
"""

from __future__ import annotations

from lts.core.hedge_algebras import HAParams

PAPER_WORDS_7 = ("V-", "-", "L-", "W", "L+", "+", "V+")

# Tên ngôn ngữ tự nhiên tương ứng (để hiển thị)
WORD_LABELS = {
    "V-": "Very Small",
    "-":  "Small",
    "L-": "Rather Small",
    "W":  "Middle",
    "L+": "Rather Large",
    "+":  "Large",
    "V+": "Very Large",
}


def sqm_closed_form_7(word: str, params: HAParams) -> float:
    """Công thức SQM closed-form cho bộ 7 từ — phương trình (4.1)-(4.7).

    Parameters
    ----------
    word : str
        Một trong PAPER_WORDS_7.
    params : HAParams
        Tham số theta, alpha.

    Returns
    -------
    float
        Giá trị SQM trong [0, 1].

    Raises
    ------
    ValueError
        Nếu word không thuộc PAPER_WORDS_7.
    """
    theta, alpha = params.theta, params.alpha

    formulas: dict[str, float] = {
        "V-": theta - 2 * theta * alpha + theta * alpha ** 2,           # (4.1)
        "-":  theta - theta * alpha,                                     # (4.2)
        "L-": theta - theta * alpha ** 2,                               # (4.3)
        "W":  theta,                                                     # (4.4)
        "L+": theta + alpha ** 2 - theta * alpha ** 2,                  # (4.5)
        "+":  theta - theta * alpha + alpha,                             # (4.6)
        "V+": theta + 2 * alpha - alpha ** 2 - 2 * theta * alpha + theta * alpha ** 2,  # (4.7)
    }

    if word not in formulas:
        raise ValueError(
            f"Word '{word}' không thuộc bộ 7 từ. "
            f"Các từ hợp lệ: {PAPER_WORDS_7}"
        )
    return formulas[word]


def all_sqm_values(params: HAParams) -> dict[str, float]:
    """Trả về dict {word: sqm_value} cho toàn bộ 7 từ."""
    return {w: sqm_closed_form_7(w, params) for w in PAPER_WORDS_7}


def validate_against_recursive(params: HAParams) -> dict[str, tuple[float, float]]:
    """So sánh closed-form với recursive SQM để kiểm tra tính đúng đắn.

    Returns
    -------
    dict[str, tuple[float, float]]
        {word: (closed_form_value, recursive_sqm_value)}
    """
    from lts.core.hedge_algebras import HedgeAlgebra

    ha = HedgeAlgebra(params)
    result = {}
    for word in PAPER_WORDS_7:
        cf = sqm_closed_form_7(word, params)
        rec = ha.sqm(word)
        result[word] = (cf, rec)
    return result
