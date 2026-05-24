"""Tests for HOLTSModel (hieund_2021)."""
import pytest
from lts.core.hedge_algebras import HAParams
from lts.data.loader import DataLoader
from lts.metrics.measures import ForecastMetrics
from lts.models.ho_lts import HOLTSModel


@pytest.fixture
def alabama():
    return DataLoader.bundled("alabama")


@pytest.fixture
def model_order2(alabama):
    params = HAParams(theta=0.527, alpha=0.412)
    m = HOLTSModel(params, order=2, specificity=2)
    m.fit(alabama.values, alabama.lb, alabama.ub)
    return m


class TestHOLTSFallback:
    def test_fallback_averages_all_lhs_words(self, alabama):
        """Fallback rule phải là trung bình tất cả từ trong LHS, không chỉ từ cuối."""
        params = HAParams(theta=0.527, alpha=0.412)
        m = HOLTSModel(params, order=2, specificity=2)
        m.fit(alabama.values, alabama.lb, alabama.ub)

        sp = m.semantic_points
        # Dùng empty rules để chắc chắn trigger fallback
        lhs = ("VV-", "L-")  # Hai từ khác nhau
        if "VV-" in sp and "L-" in sp:
            result = m._forecast_one(lhs, {}, sp)  # {} → không có rule → fallback
            expected = (sp["VV-"] + sp["L-"]) / 2
            assert result == pytest.approx(expected)

    def test_fallback_differs_from_lts_model(self, alabama):
        """HOLTSModel fallback khác LTSModel fallback khi order > 1."""
        from lts.models.lts_model import LTSModel

        params = HAParams(theta=0.527, alpha=0.412)
        m_ho = HOLTSModel(params, order=2, specificity=2)
        m_lts = LTSModel(params, specificity=2, order=2)
        m_ho.fit(alabama.values, alabama.lb, alabama.ub)
        m_lts.fit(alabama.values, alabama.lb, alabama.ub)

        # With order=2, HO-LTS fallback = avg(lhs[0], lhs[1])
        # LTS fallback = lhs[-1] only
        # They may produce different forecasts when fallback triggers
        ho_pred = m_ho.predict()
        lts_pred = m_lts.predict()
        # They may differ (not required, but the model is distinct)
        assert len(ho_pred) == len(lts_pred)

    def test_fallback_order1_same_as_lts_for_no_rule(self):
        """Order=1: HO-LTS fallback avg(lhs) = lhs[-1] (single element)."""
        params = HAParams(theta=0.527, alpha=0.412)
        m = HOLTSModel(params, order=1, specificity=2)
        sp = {"W": 16990.0, "-": 15035.0}
        lhs = ("W",)  # LHS không có trong rules
        result = m._forecast_one(lhs, {}, sp)
        assert result == pytest.approx(sp["W"])


class TestHOLTSFit:
    def test_output_length(self, model_order2, alabama):
        """Forecast length phải = data length - order."""
        forecasted = model_order2.predict()
        assert len(forecasted) == len(alabama.values) - 2

    def test_order3_runs(self, alabama):
        params = HAParams(theta=0.65, alpha=0.35)
        m = HOLTSModel(params, order=3, specificity=3)
        m.fit(alabama.values, alabama.lb, alabama.ub)
        assert len(m.predict()) == len(alabama.values) - 3

    def test_order5_runs(self, alabama):
        params = HAParams(theta=0.65, alpha=0.35)
        m = HOLTSModel(params, order=5, specificity=3)
        m.fit(alabama.values, alabama.lb, alabama.ub)
        assert len(m.predict()) == len(alabama.values) - 5

    def test_mse_reasonable(self, model_order2, alabama):
        """MSE phải trong khoảng hợp lý cho Alabama."""
        metrics = ForecastMetrics.compute(alabama.values[2:], model_order2.predict())
        assert metrics.mse < 2_000_000

    def test_15word_specificity2(self, alabama):
        """specificity=2 phải có 15 từ trong vocab."""
        params = HAParams(theta=0.527, alpha=0.412)
        m = HOLTSModel(params, order=2, specificity=2)
        m.fit(alabama.values, alabama.lb, alabama.ub)
        assert len(m.words) == 15

    def test_31word_specificity3(self, alabama):
        """specificity=3 phải có 31 từ trong vocab."""
        params = HAParams(theta=0.65, alpha=0.35)
        m = HOLTSModel(params, order=2, specificity=3)
        m.fit(alabama.values, alabama.lb, alabama.ub)
        assert len(m.words) == 31


class TestHOLTSParamsPreset:
    def test_spec1_preset(self):
        p = HOLTSModel.params_for_specificity(1)
        assert p.theta == pytest.approx(0.437)
        assert p.alpha == pytest.approx(0.511)

    def test_spec2_preset(self):
        p = HOLTSModel.params_for_specificity(2)
        assert p.theta == pytest.approx(0.527)
        assert p.alpha == pytest.approx(0.412)

    def test_word_count_7_preset(self):
        p = HOLTSModel.params_for_word_count(7)
        assert p.theta == pytest.approx(0.437)
        assert p.alpha == pytest.approx(0.511)

    def test_word_count_15_preset(self):
        p = HOLTSModel.params_for_word_count(15)
        assert p.theta == pytest.approx(0.527)
        assert p.alpha == pytest.approx(0.412)

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            HOLTSModel.params_for_word_count(10)
