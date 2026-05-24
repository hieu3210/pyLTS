"""Tests cho LTSModel — kiểm tra tái hiện Table 1-4 của bài báo."""

import pytest
from lts.core.hedge_algebras import HAParams
from lts.data.loader import DataLoader
from lts.metrics.measures import ForecastMetrics
from lts.models.lts_model import LTSModel

PARAMS = HAParams.enrollment()  # theta=0.57, alpha=0.49
ALABAMA_DATA = [
    13055, 13563, 13867, 14696, 15460, 15311, 15603, 15861,
    16807, 16919, 16388, 15433, 15497, 15145, 15163, 15984,
    16859, 18150, 18970, 19328, 19337, 18876,
]
LB, UB = 13000.0, 20000.0


@pytest.fixture
def model():
    m = LTSModel(params=PARAMS, specificity=1, order=1, use_repeat=False)
    m.fit(ALABAMA_DATA, LB, UB)
    return m


class TestDataLabeling:
    """Kiểm tra Table 1: nhãn ngôn ngữ của từng năm."""

    def test_has_correct_number_of_labels(self, model):
        assert len(model.labels) == len(ALABAMA_DATA)

    def test_labels_from_vocabulary(self, model):
        """Mọi nhãn phải thuộc từ vựng đã định nghĩa."""
        for label in model.labels:
            assert label in model.words, f"Nhãn '{label}' không thuộc từ vựng"

    def test_semantic_points_count(self, model):
        """Số semantic points = số từ trong từ vựng."""
        assert len(model.semantic_points) == len(model.words)

    def test_semantic_points_in_range(self, model):
        """Mọi semantic point phải nằm trong [lb, ub]."""
        for w, v in model.semantic_points.items():
            assert LB <= v <= UB, f"semantic_point({w}) = {v} ngoài [{LB}, {UB}]"


class TestRuleInduction:
    """Kiểm tra Table 2-3: LLRs và LLRGs."""

    def test_rules_built(self, model):
        assert len(model.rules) > 0, "Phải có ít nhất 1 LLRG"

    def test_rules_lhs_in_vocab(self, model):
        for lhs in model.rules:
            for w in lhs:
                assert w in model.words

    def test_rules_rhs_in_vocab(self, model):
        for rhs_list in model.rules.values():
            for w in rhs_list:
                assert w in model.words

    def test_first_order_lhs_length(self, model):
        """Với order=1, mỗi LHS có đúng 1 phần tử."""
        for lhs in model.rules:
            assert len(lhs) == 1


class TestForecasting:
    """Kiểm tra Table 4: giá trị dự báo và độ đo lỗi."""

    def test_forecast_count(self, model):
        """Số giá trị dự báo = len(data) - order."""
        assert len(model.predict()) == len(ALABAMA_DATA) - 1

    def test_forecast_in_range(self, model):
        """Mọi giá trị dự báo phải trong khoảng hợp lý."""
        for f in model.predict():
            assert LB * 0.5 <= f <= UB * 1.5, f"Dự báo {f} ra ngoài range hợp lý"

    def test_mse_matches_paper(self, model):
        """MSE phải ≈ 262,211 (≈ paper Table 4: 262.326 × 1000 units, use_repeat=False)."""
        actual = ALABAMA_DATA[1:]
        metrics = ForecastMetrics.compute(actual, model.predict())
        assert metrics.mse == pytest.approx(262326, rel=0.01), (
            f"MSE = {metrics.mse:.3f}, kỳ vọng ≈ 262326"
        )

    def test_mape_matches_paper(self, model):
        """MAPE phải ≈ 2.57% cho mô hình enrollment trực tiếp (Table 4).

        Lưu ý: MAPE=1.27% trong bài báo là từ mô hình variation (Table 8), không phải Table 4.
        """
        actual = ALABAMA_DATA[1:]
        metrics = ForecastMetrics.compute(actual, model.predict(), mape_digits=2)
        assert metrics.mape == pytest.approx(2.57, abs=0.3), (
            f"MAPE = {metrics.mape}%, kỳ vọng ≈ 2.57%"
        )


class TestForecastResult:
    """Kiểm tra ForecastResult structure."""

    def test_result_data_matches_input(self, model):
        result = model.get_result()
        assert result.data == ALABAMA_DATA

    def test_result_actual_is_data_from_order(self, model):
        result = model.get_result()
        assert result.actual == ALABAMA_DATA[1:]

    def test_result_forecasted_length(self, model):
        result = model.get_result()
        assert len(result.forecasted) == len(ALABAMA_DATA) - 1

    def test_llrg_summary(self, model):
        result = model.get_result()
        summary = result.llrg_summary()
        assert len(summary) == result.n_rules


class TestDataLoaderIntegration:
    """Kiểm tra tải dataset bundled và chạy mô hình."""

    def test_bundled_alabama(self):
        dataset = DataLoader.bundled("alabama")
        assert len(dataset.values) == 22
        assert dataset.lb == 13000.0
        assert dataset.ub == 20000.0

    def test_full_pipeline_with_loader(self):
        dataset = DataLoader.bundled("alabama")
        model = LTSModel(params=PARAMS, specificity=1)
        model.fit(dataset.values, dataset.lb, dataset.ub)
        assert len(model.predict()) == len(dataset.values) - 1
