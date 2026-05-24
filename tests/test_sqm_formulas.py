"""Tests cho closed-form SQM — kiểm tra khớp với recursive và với bài báo."""

import pytest
from lts.core.hedge_algebras import HAParams, HedgeAlgebra
from lts.core.sqm_formulas import (
    PAPER_WORDS_7,
    all_sqm_values,
    sqm_closed_form_7,
    validate_against_recursive,
)

PARAMS = HAParams.enrollment()  # theta=0.57, alpha=0.49


class TestClosedFormValues:
    """Kiểm tra giá trị closed-form khớp với bài báo."""

    def test_middle_equals_theta(self):
        """v(M) = θ (phương trình 4.4)."""
        assert sqm_closed_form_7("W", PARAMS) == pytest.approx(PARAMS.theta)

    def test_all_7_words_in_range(self):
        values = all_sqm_values(PARAMS)
        for word, v in values.items():
            assert 0.0 <= v <= 1.0, f"v({word}) = {v} ngoài [0,1]"

    def test_monotone_ordering(self):
        """v(V-) < v(-) < v(L-) < v(W) < v(L+) < v(+) < v(V+)."""
        values = all_sqm_values(PARAMS)
        ordered = [values[w] for w in PAPER_WORDS_7]
        assert ordered == sorted(ordered), "Thứ tự ngữ nghĩa không đúng"

    def test_invalid_word_raises(self):
        with pytest.raises(ValueError):
            sqm_closed_form_7("X", PARAMS)

    def test_very_small_formula(self):
        """v(VS) = θ - 2θα + θα² (phương trình 4.1)."""
        t, a = PARAMS.theta, PARAMS.alpha
        expected = t - 2 * t * a + t * a ** 2
        assert sqm_closed_form_7("V-", PARAMS) == pytest.approx(expected)

    def test_very_large_formula(self):
        """v(VL) = θ + 2α - α² - 2θα + θα² (phương trình 4.7)."""
        t, a = PARAMS.theta, PARAMS.alpha
        expected = t + 2 * a - a ** 2 - 2 * t * a + t * a ** 2
        assert sqm_closed_form_7("V+", PARAMS) == pytest.approx(expected)


class TestClosedFormVsRecursive:
    """Kiểm tra closed-form khớp với recursive SQM."""

    def test_all_7_words_match(self):
        comparison = validate_against_recursive(PARAMS)
        for word, (cf, rec) in comparison.items():
            assert cf == pytest.approx(rec, rel=1e-9), (
                f"Không khớp cho '{word}': closed={cf:.8f}, recursive={rec:.8f}"
            )

    def test_paper_values(self):
        """Kiểm tra các giá trị cụ thể đã công bố trong bài báo.

        Từ bài báo (theta=0.57, alpha=0.49):
          v(X1) ≈ 0.1483 (Very Small)
          v(X2) ≈ 0.2907 (Small)
          v(X3) ≈ 0.4331 (Rather Small)  -- xấp xỉ
          v(X4) = 0.57   (Middle)
          v(X5) ≈ 0.6683 (Rather Large)
          v(X6) ≈ 0.8093 (Large)
          v(X7) ≈ 0.8882 (Very Large)
        """
        t, a = PARAMS.theta, PARAMS.alpha

        # v(VS) = θ(1-α)² = 0.57 * (0.51)² ≈ 0.1483
        vs = sqm_closed_form_7("V-", PARAMS)
        assert vs == pytest.approx(t * (1 - a) ** 2, rel=1e-6)

        # v(S) = θ(1-α) = 0.57 * 0.51 ≈ 0.2907
        s = sqm_closed_form_7("-", PARAMS)
        assert s == pytest.approx(t * (1 - a), rel=1e-6)

        # v(M) = θ = 0.57
        m = sqm_closed_form_7("W", PARAMS)
        assert m == pytest.approx(t, rel=1e-6)

        # v(L) = θ(1-α) + α = v(S) + α
        lg = sqm_closed_form_7("+", PARAMS)
        assert lg == pytest.approx(t * (1 - a) + a, rel=1e-6)
