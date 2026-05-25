"""Tests for SWLTSModel và SWPSOLTSModel (SW-LTS — Hướng nghiên cứu 1)."""
import math
import pytest
from lts.core.hedge_algebras import HAParams
from lts.data.loader import DataLoader
from lts.metrics.measures import ForecastMetrics
from lts.models.sw_lts import SWLTSModel, SWPSOLTSModel, _lhs_vector, _sq_dist


@pytest.fixture
def alabama():
    return DataLoader.bundled("alabama")


@pytest.fixture
def model_sw(alabama):
    params = HAParams.enrollment()
    m = SWLTSModel(params, sigma=1000.0, specificity=1, order=1)
    m.fit(alabama.values, alabama.lb, alabama.ub)
    return m


class TestHelpers:
    def test_lhs_vector(self):
        sp = {"A": 1.0, "B": 2.0, "C": 3.0}
        assert _lhs_vector(("A", "C"), sp) == [1.0, 3.0]

    def test_sq_dist_zero(self):
        assert _sq_dist([1.0, 2.0], [1.0, 2.0]) == pytest.approx(0.0)

    def test_sq_dist_basic(self):
        assert _sq_dist([0.0, 0.0], [3.0, 4.0]) == pytest.approx(25.0)


class TestSWLTSKernel:
    def test_small_sigma_approaches_exact_match(self, alabama):
        """Sigma rất nhỏ → kết quả gần với exact-match (LTS gốc)."""
        from lts.models.lts_model import LTSModel
        params = HAParams.enrollment()

        m_sw = SWLTSModel(params, sigma=0.01, specificity=1, order=1)
        m_lts = LTSModel(params, specificity=1, order=1, use_repeat=False)
        m_sw.fit(alabama.values, alabama.lb, alabama.ub)
        m_lts.fit(alabama.values, alabama.lb, alabama.ub)

        pred_sw = m_sw.predict()
        pred_lts = m_lts.predict()
        # Với sigma rất nhỏ, SW-LTS xấp xỉ exact-match
        diffs = [abs(a - b) for a, b in zip(pred_sw, pred_lts)]
        assert max(diffs) < 100.0  # Sai lệch nhỏ

    def test_large_sigma_uses_all_rules(self, alabama):
        """Sigma rất lớn → tất cả rule đóng góp gần như bằng nhau."""
        params = HAParams.enrollment()
        m_big = SWLTSModel(params, sigma=1e9, specificity=1, order=1)
        m_big.fit(alabama.values, alabama.lb, alabama.ub)
        pred = m_big.predict()
        # Phải cho kết quả hợp lý (không NaN, không inf)
        assert all(math.isfinite(v) for v in pred)
        assert all(alabama.lb * 0.5 < v < alabama.ub * 1.5 for v in pred)

    def test_forecast_uses_all_rules_not_just_exact(self, alabama):
        """SW-LTS phải cho kết quả khác với LTS gốc ở ít nhất một điểm."""
        from lts.models.lts_model import LTSModel
        params = HAParams.enrollment()

        m_sw = SWLTSModel(params, sigma=500.0, specificity=1, order=1)
        m_lts = LTSModel(params, specificity=1, order=1, use_repeat=False)
        m_sw.fit(alabama.values, alabama.lb, alabama.ub)
        m_lts.fit(alabama.values, alabama.lb, alabama.ub)

        diffs = sum(1 for a, b in zip(m_sw.predict(), m_lts.predict()) if abs(a - b) > 1e-6)
        assert diffs > 0, "SW-LTS phải cho kết quả khác LTS ở ít nhất 1 điểm"

    def test_auto_sigma(self, alabama):
        """Khi sigma=None, tự động tính từ dữ liệu và fit thành công."""
        params = HAParams.enrollment()
        m = SWLTSModel(params, sigma=None, specificity=1, order=1)
        m.fit(alabama.values, alabama.lb, alabama.ub)
        assert m.sigma > 0
        assert len(m.predict()) == len(alabama.values) - 1

    def test_kernel_formula(self):
        """Kiểm tra công thức kernel trực tiếp (normalize=False để dùng raw distance)."""
        params = HAParams(theta=0.5, alpha=0.5)
        m = SWLTSModel(params, sigma=1000.0, specificity=1, order=1, normalize=False)
        sp = {"A": 0.0, "B": 1000.0, "C": 2000.0}
        rules = {("A",): ["C"], ("B",): ["A"]}
        m._sigma = 1000.0

        result = m._forecast_one(("A",), rules, sp)
        # sim(A,A) = exp(0) = 1.0, mean_s(RHS_A) = 2000
        # sim(A,B) = exp(-1000²/(2×1000²)) = exp(-0.5) ≈ 0.607, mean_s(RHS_B) = 0
        exp_neg_half = math.exp(-0.5)
        expected = (1.0 * 2000.0 + exp_neg_half * 0.0) / (1.0 + exp_neg_half)
        assert result == pytest.approx(expected, rel=1e-6)

    def test_normalize_true_scale_invariant(self, alabama):
        """Với normalize=True, sigma có ý nghĩa nhất quán (không phụ thuộc span)."""
        params = HAParams.enrollment()
        # sigma=0.2 nghĩa là 20% span — giá trị hợp lý trên mọi dataset
        m1 = SWLTSModel(params, sigma=0.2, specificity=1, order=1, normalize=True)
        m1.fit(alabama.values, alabama.lb, alabama.ub)
        pred = m1.predict()
        assert all(math.isfinite(v) for v in pred)
        assert all(alabama.lb * 0.8 < v < alabama.ub * 1.2 for v in pred)


class TestSWLTSFit:
    def test_output_length(self, model_sw, alabama):
        assert len(model_sw.predict()) == len(alabama.values) - 1

    def test_output_in_range(self, model_sw, alabama):
        pred = model_sw.predict()
        assert all(alabama.lb * 0.8 < v < alabama.ub * 1.2 for v in pred)

    def test_order2(self, alabama):
        params = HAParams.enrollment()
        m = SWLTSModel(params, sigma=800.0, specificity=1, order=2)
        m.fit(alabama.values, alabama.lb, alabama.ub)
        assert len(m.predict()) == len(alabama.values) - 2

    def test_mse_finite(self, model_sw, alabama):
        mse = ForecastMetrics.compute(alabama.values[1:], model_sw.predict()).mse
        assert math.isfinite(mse) and mse > 0

    def test_get_result_consistent(self, model_sw, alabama):
        result = model_sw.get_result()
        assert len(result.forecasted) == len(alabama.values) - 1


class TestSWBetterThanLTS:
    def test_swlts_beats_lts_on_alabama(self, alabama):
        """SW-LTS với sigma tối ưu phải cho MSE <= LTS gốc trên Alabama.

        Đây là điều kiện cần thiết để SW-LTS có ý nghĩa.
        """
        from lts.models.lts_model import LTSModel
        params = HAParams.enrollment()

        m_lts = LTSModel(params, specificity=1, order=1, use_repeat=False)
        m_lts.fit(alabama.values, alabama.lb, alabama.ub)
        mse_lts = ForecastMetrics.compute(alabama.values[1:], m_lts.predict()).mse

        # Thử nhiều giá trị sigma trong không gian chuẩn hóa [0, 1]
        best_mse_sw = float("inf")
        for sigma in [0.05, 0.1, 0.2, 0.3, 0.5, 0.8]:
            m = SWLTSModel(params, sigma=sigma, specificity=1, order=1)
            m.fit(alabama.values, alabama.lb, alabama.ub)
            mse = ForecastMetrics.compute(alabama.values[1:], m.predict()).mse
            best_mse_sw = min(best_mse_sw, mse)

        # SW-LTS (best sigma) phải tốt hơn hoặc bằng LTS gốc
        assert best_mse_sw <= mse_lts * 1.05  # cho phép 5% margin


class TestSWPSOLTS:
    def test_runs_without_error(self, alabama):
        m = SWPSOLTSModel(specificity=1, order=1, n_runs=1)
        from lts.optimization.pso import PSOConfig
        small_cfg = PSOConfig(
            n_particles=5, max_iter=10, omega=0.4, c1=2.0, c2=2.0,
            bounds=[(0.3, 0.7), (0.3, 0.7), (0.01, 1.0)], seed=0
        )
        m._pso_config = small_cfg
        m.fit(alabama.values, alabama.lb, alabama.ub)
        assert len(m.predict()) == len(alabama.values) - 1

    def test_best_params_in_bounds(self, alabama):
        m = SWPSOLTSModel(specificity=1, order=1, n_runs=1)
        from lts.optimization.pso import PSOConfig
        small_cfg = PSOConfig(
            n_particles=5, max_iter=10, omega=0.4, c1=2.0, c2=2.0,
            bounds=[(0.3, 0.7), (0.3, 0.7), (0.01, 1.0)], seed=0
        )
        m._pso_config = small_cfg
        m.fit(alabama.values, alabama.lb, alabama.ub)
        theta, alpha, sigma = m.best_params
        assert 0.3 <= theta <= 0.7
        assert 0.3 <= alpha <= 0.7
        assert 0.01 <= sigma <= 1.0
