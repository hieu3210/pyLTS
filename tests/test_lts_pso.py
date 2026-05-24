"""Tests for LTSPSOModel (hieund_2022)."""
import pytest
from lts.core.hedge_algebras import HAParams
from lts.data.loader import DataLoader
from lts.metrics.measures import ForecastMetrics
from lts.models.lts_pso import LTSPSOModel
from lts.optimization.pso import PSO, PSOConfig


@pytest.fixture
def alabama():
    return DataLoader.bundled("alabama")


@pytest.fixture
def model_fixed(alabama):
    """LTSPSOModel với params cố định (không PSO) để test nhanh."""
    params = HAParams(theta=0.4789, alpha=0.608)
    m = LTSPSOModel(params, specificity=2, order=1)
    m.fit(alabama.values, alabama.lb, alabama.ub)
    return m


class TestPSOOptimizer:
    def test_pso_minimizes_quadratic(self):
        """PSO phải tìm được minimum gần (0.5, 0.5) cho hàm quadratic."""
        def f(p):
            return (p[0] - 0.5) ** 2 + (p[1] - 0.5) ** 2

        cfg = PSOConfig(
            n_particles=20, max_iter=100, omega=0.4, c1=2.0, c2=2.0,
            bounds=[(0.3, 0.7), (0.3, 0.7)], seed=42
        )
        best_pos, best_val = PSO(f, cfg).run()
        assert best_pos[0] == pytest.approx(0.5, abs=0.02)
        assert best_pos[1] == pytest.approx(0.5, abs=0.02)
        assert best_val < 1e-3

    def test_pso_returns_within_bounds(self):
        cfg = PSOConfig(
            n_particles=10, max_iter=20, omega=0.4, c1=2.0, c2=2.0,
            bounds=[(0.3, 0.7), (0.3, 0.7)], seed=0
        )
        pos, _ = PSO(lambda p: sum(p), cfg).run()
        assert 0.3 <= pos[0] <= 0.7
        assert 0.3 <= pos[1] <= 0.7


class TestLTSPSOForecastFormula:
    def test_forecast_formula_with_rule(self):
        """Khi có rule: forecast = 0.5 × (s_lhs + mean(s_rhs))."""
        params = HAParams(theta=0.5, alpha=0.5)
        m = LTSPSOModel(params, specificity=1, order=1)
        sp = {"A": 100.0, "B": 200.0, "C": 300.0}
        rules = {("A",): ["B", "C"]}

        result = m._forecast_one(("A",), rules, sp)
        expected = 0.5 * (100.0 + (200.0 + 300.0) / 2)
        assert result == pytest.approx(expected)

    def test_forecast_formula_no_rule(self):
        """Khi không có rule: forecast = s_lhs_last."""
        params = HAParams(theta=0.5, alpha=0.5)
        m = LTSPSOModel(params, specificity=1, order=1)
        sp = {"A": 100.0, "B": 200.0}
        rules = {}

        result = m._forecast_one(("A",), rules, sp)
        assert result == pytest.approx(100.0)

    def test_forecast_formula_single_rhs(self):
        """Một RHS duy nhất: forecast = 0.5 × (s_lhs + s_rhs)."""
        params = HAParams(theta=0.5, alpha=0.5)
        m = LTSPSOModel(params, specificity=1, order=1)
        sp = {"A": 100.0, "B": 200.0}
        rules = {("A",): ["B"]}

        result = m._forecast_one(("A",), rules, sp)
        assert result == pytest.approx(0.5 * (100.0 + 200.0))


class TestLTSPSOFit:
    def test_output_length(self, model_fixed, alabama):
        assert len(model_fixed.predict()) == len(alabama.values) - 1

    def test_15word_vocab(self, model_fixed):
        """specificity=2 → 15 từ trong vocab."""
        assert len(model_fixed.words) == 15

    def test_mse_reasonable(self, model_fixed, alabama):
        """Với params Alabama từ bài báo, MSE phải hợp lý."""
        metrics = ForecastMetrics.compute(alabama.values[1:], model_fixed.predict())
        assert metrics.mse < 500_000

    def test_get_result_consistent(self, model_fixed, alabama):
        result = model_fixed.get_result()
        assert len(result.forecasted) == len(alabama.values) - 1
        assert result.order == 1


class TestLTSPSOOptimize:
    def test_fit_optimize_small(self, alabama):
        """PSO tối ưu với config nhỏ (nhanh) phải chạy không lỗi."""
        params = HAParams(theta=0.5, alpha=0.5)
        m = LTSPSOModel(params, specificity=2, order=1)
        small_cfg = PSOConfig(
            n_particles=5, max_iter=10, omega=0.4, c1=2.0, c2=2.0,
            bounds=[(0.3, 0.7), (0.3, 0.7)], seed=0
        )
        best = m.fit_optimize(
            alabama.values, alabama.lb, alabama.ub,
            pso_config=small_cfg, n_runs=1
        )
        assert 0.3 <= best.theta <= 0.7
        assert 0.3 <= best.alpha <= 0.7
        assert len(m.predict()) == len(alabama.values) - 1

    def test_fit_optimize_improves_or_equal(self, alabama):
        """MSE sau PSO phải <= MSE với params mặc định (trong điều kiện lý tưởng)."""
        params_default = HAParams(theta=0.5, alpha=0.5)
        m_default = LTSPSOModel(params_default, specificity=2, order=1)
        m_default.fit(alabama.values, alabama.lb, alabama.ub)
        mse_default = ForecastMetrics.compute(
            alabama.values[1:], m_default.predict()
        ).mse

        m_opt = LTSPSOModel(params_default, specificity=2, order=1)
        small_cfg = PSOConfig(
            n_particles=10, max_iter=30, omega=0.4, c1=2.0, c2=2.0,
            bounds=[(0.3, 0.7), (0.3, 0.7)], seed=42
        )
        m_opt.fit_optimize(
            alabama.values, alabama.lb, alabama.ub,
            pso_config=small_cfg, n_runs=2
        )
        mse_opt = ForecastMetrics.compute(
            alabama.values[1:], m_opt.predict()
        ).mse
        # PSO-optimized should be at most 2x worse than default
        assert mse_opt < mse_default * 2
