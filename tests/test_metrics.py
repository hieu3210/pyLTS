"""Tests cho ForecastMetrics."""

import math
import pytest
from lts.metrics.measures import ForecastMetrics


class TestForecastMetrics:
    def test_perfect_forecast_mse_zero(self):
        actual = [1.0, 2.0, 3.0]
        m = ForecastMetrics.compute(actual, actual)
        assert m.mse == 0.0
        assert m.mae == 0.0
        assert m.rmse == 0.0
        assert m.mape == 0.0

    def test_simple_mse(self):
        actual = [100.0, 200.0]
        forecasted = [110.0, 190.0]
        m = ForecastMetrics.compute(actual, forecasted)
        # errors = [10, 10], mse = (100+100)/2 = 100
        assert m.mse == pytest.approx(100.0)
        assert m.mae == pytest.approx(10.0)
        assert m.rmse == pytest.approx(10.0)

    def test_mape_calculation(self):
        actual = [100.0, 200.0]
        forecasted = [110.0, 210.0]
        m = ForecastMetrics.compute(actual, forecasted, mape_digits=4)
        # mape = (10/100 + 10/200) / 2 * 100 = (0.1 + 0.05) / 2 * 100 = 7.5
        assert m.mape == pytest.approx(7.5, rel=1e-4)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="Độ dài"):
            ForecastMetrics.compute([1.0, 2.0], [1.0])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            ForecastMetrics.compute([], [])

    def test_n_correct(self):
        actual = [1.0, 2.0, 3.0, 4.0]
        m = ForecastMetrics.compute(actual, actual)
        assert m.n == 4

    def test_smape_symmetric(self):
        """SMAPE phải đối xứng: swap actual/forecasted cho cùng kết quả."""
        a = [100.0, 200.0, 150.0]
        f = [120.0, 180.0, 160.0]
        m1 = ForecastMetrics.compute(a, f)
        m2 = ForecastMetrics.compute(f, a)
        assert m1.smape == pytest.approx(m2.smape, rel=1e-6)

    def test_rmse_is_sqrt_mse(self):
        actual = [1.0, 2.0, 3.0]
        forecasted = [1.5, 2.5, 3.5]
        m = ForecastMetrics.compute(actual, forecasted)
        assert m.rmse == pytest.approx(math.sqrt(m.mse))

    def test_immutable(self):
        m = ForecastMetrics.compute([1.0], [1.1])
        with pytest.raises(Exception):
            m.mse = 0.0  # frozen dataclass không cho phép gán
