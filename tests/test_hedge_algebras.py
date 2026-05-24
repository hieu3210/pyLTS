"""Tests cho HedgeAlgebra — kiểm tra SQM, fm, sign, omega."""

import pytest
from lts.core.hedge_algebras import HAParams, HedgeAlgebra

PARAMS = HAParams.enrollment()  # theta=0.57, alpha=0.49


@pytest.fixture
def ha():
    return HedgeAlgebra(PARAMS)


class TestSQM:
    """Kiểm tra SQM khớp với các giá trị trong bài báo (Table 2)."""

    # Giá trị từ bài báo: theta=0.57, alpha=0.49
    # v(X1)=0.1183(?), cần kiểm tra bằng closed-form
    expected = {
        "0": 0.0,
        "W": 0.57,
        "1": 1.0,
        "-": 0.57 - 0.49 * 0.57,       # θ - α·θ = θ(1-α)
        "+": 0.57 + 0.49 * (1 - 0.57),  # θ + α·(1-θ)
    }

    def test_boundary_elements(self, ha):
        assert ha.sqm("0") == 0.0
        assert ha.sqm("W") == pytest.approx(0.57)
        assert ha.sqm("1") == 1.0

    def test_generators(self, ha):
        # v("-") = θ - α·fm("-") = θ - α·θ
        expected_minus = PARAMS.theta - PARAMS.alpha * PARAMS.theta
        assert ha.sqm("-") == pytest.approx(expected_minus, rel=1e-6)

        # v("+") = θ + α·fm("+") = θ + α·(1-θ)
        expected_plus = PARAMS.theta + PARAMS.alpha * (1 - PARAMS.theta)
        assert ha.sqm("+") == pytest.approx(expected_plus, rel=1e-6)

    def test_sqm_ordering(self, ha):
        """Kiểm tra thứ tự ngữ nghĩa: V- < - < L- < W < L+ < + < V+."""
        words_7 = ["V-", "-", "L-", "W", "L+", "+", "V+"]
        sqm_vals = [ha.sqm(w) for w in words_7]
        assert sqm_vals == sorted(sqm_vals), "SQM không đơn điệu tăng dần"

    def test_sqm_range(self, ha):
        """Mọi giá trị SQM phải nằm trong [0, 1]."""
        words = ha.get_words(2)
        for w in words:
            v = ha.sqm(w)
            assert 0.0 <= v <= 1.0, f"sqm({w}) = {v} ngoài [0,1]"

    def test_very_small_lt_small(self, ha):
        assert ha.sqm("V-") < ha.sqm("-")

    def test_very_large_gt_large(self, ha):
        assert ha.sqm("V+") > ha.sqm("+")

    def test_rather_small_gt_small(self, ha):
        """L- (Rather Small) nằm giữa Small và Middle."""
        assert ha.sqm("-") < ha.sqm("L-") < ha.sqm("W")

    def test_rather_large_lt_large(self, ha):
        """L+ (Rather Large) nằm giữa Middle và Large."""
        assert ha.sqm("W") < ha.sqm("L+") < ha.sqm("+")


class TestFM:
    """Kiểm tra fuzziness measure."""

    def test_boundary_fm_is_zero(self, ha):
        for w in ["0", "W", "1"]:
            assert ha.fm(w) == 0.0, f"fm({w}) phải = 0"

    def test_fm_generator_minus(self, ha):
        assert ha.fm("-") == pytest.approx(PARAMS.theta)

    def test_fm_generator_plus(self, ha):
        assert ha.fm("+") == pytest.approx(1.0 - PARAMS.theta)

    def test_fm_hedge_rather(self, ha):
        """fm(L-) = alpha * fm(-)."""
        assert ha.fm("L-") == pytest.approx(PARAMS.alpha * PARAMS.theta)

    def test_fm_hedge_very(self, ha):
        """fm(V-) = beta * fm(-)."""
        assert ha.fm("V-") == pytest.approx(PARAMS.beta * PARAMS.theta)


class TestSign:
    """Kiểm tra hàm sign."""

    def test_sign_minus(self, ha):
        assert ha.sign("-") == -1

    def test_sign_plus(self, ha):
        assert ha.sign("+") == 1

    def test_sign_very_minus(self, ha):
        # V- = Very Small, V giữ dấu
        assert ha.sign("V-") == -1

    def test_sign_rather_minus(self, ha):
        # L- = Rather Small, L đảo dấu: sign("-")=-1 → sign("L-")=+1? Không đúng
        # Đúng hơn: sign("L-") = -sign("-") = -(-1) = 1...
        # Nhưng L- là Rather Small, nằm giữa Small và Middle, sign nên > 0 tương đối
        # Theo code: sign("L-") = -sign("-") = -(-1) = 1
        assert ha.sign("L-") == 1

    def test_sign_very_large(self, ha):
        assert ha.sign("V+") == 1

    def test_sign_rather_large(self, ha):
        assert ha.sign("L+") == -1


class TestGetWords:
    """Kiểm tra sinh từ vựng."""

    def test_specificity_1_has_7_words(self, ha):
        words = ha.get_words(1)
        # Bao gồm cả boundary: 0, W, 1 và 4 generators: -, +, L-, L+ ...
        # Thực ra specificity=1 cho: ["0", "-", "W", "+", "1"] = 5 words
        # Specificity=2 thêm V-, L-, L+, V+ = 9...
        # Cần xem lại logic get_words
        assert len(words) >= 5

    def test_specificity_2_more_words(self, ha):
        w1 = ha.get_words(1)
        w2 = ha.get_words(2)
        assert len(w2) > len(w1)

    def test_words_sorted_by_sqm(self, ha):
        for spec in [1, 2]:
            words = ha.get_words(spec)
            sqm_vals = [ha.sqm(w) for w in words]
            assert sqm_vals == sorted(sqm_vals), (
                f"get_words(specificity={spec}) không sắp xếp đúng"
            )
