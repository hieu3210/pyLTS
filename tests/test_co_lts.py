"""Tests for COLTSModel (hieund_2023)."""
import math
import pytest
from lts.core.hedge_algebras import HAParams
from lts.data.loader import DataLoader
from lts.metrics.measures import ForecastMetrics
from lts.models.co_lts import COLTSConfig, COLTSModel, _all_words_up_to_depth, _decode_words


@pytest.fixture
def alabama():
    return DataLoader.bundled("alabama")


class TestDecodeWords:
    def test_returns_d_w_words(self):
        word_pool = ["A", "B", "C", "D", "E", "F", "G"]
        particle = [0.1, 0.3, 0.5, 0.7, 0.9]
        result = _decode_words(particle, word_pool, 5)
        assert len(result) == 5

    def test_all_from_pool(self):
        word_pool = ["A", "B", "C", "D", "E"]
        particle = [0.0, 0.2, 0.4, 0.6, 0.8]
        result = _decode_words(particle, word_pool, 5)
        assert all(w in word_pool for w in result)

    def test_unique_words(self):
        word_pool = ["A", "B", "C", "D", "E"]
        # Particle có nhiều giá trị mapping tới cùng index
        particle = [0.01, 0.02, 0.03, 0.7, 0.9]
        result = _decode_words(particle, word_pool, 3)
        assert len(set(result)) == len(result)

    def test_encoding_formula(self):
        """Index = floor(p × n), clamp to [0, n-1]."""
        word_pool = ["A", "B", "C", "D", "E"]  # n=5
        # p=0.6 → floor(0.6×5)=3 → word_pool[3]="D"
        particle = [0.6, 0.0, 0.0, 0.0, 0.0]
        result = _decode_words(particle, word_pool, 1)
        assert result[0] == "D"

    def test_boundary_value_one(self):
        """p=1.0 phải clamp về index cuối."""
        word_pool = ["A", "B", "C"]
        particle = [1.0]
        result = _decode_words(particle, word_pool, 1)
        assert result[0] == word_pool[-1]  # index = min(3, 2) = 2


class TestAllWordsUpToDepth:
    def test_k_max_1_gives_7_words(self):
        """k_max=1 (specificity=1) → 7 từ cơ bản."""
        params = HAParams(theta=0.5, alpha=0.5)
        words = _all_words_up_to_depth(params, 1)
        assert len(words) == 7

    def test_k_max_2_gives_more_words(self):
        params = HAParams(theta=0.5, alpha=0.5)
        w1 = _all_words_up_to_depth(params, 1)
        w2 = _all_words_up_to_depth(params, 2)
        assert len(w2) > len(w1)

    def test_k_max_3_at_least_31(self):
        params = HAParams(theta=0.5, alpha=0.5)
        words = _all_words_up_to_depth(params, 3)
        assert len(words) >= 31


class TestCOLTSConfig:
    def test_colts3_preset(self):
        cfg = COLTSConfig.colts3()
        assert cfg.k_max == 3
        assert cfg.d_w == 7

    def test_colts4_preset(self):
        cfg = COLTSConfig.colts4()
        assert cfg.k_max == 4
        assert cfg.d_w == 14

    def test_colts5_preset(self):
        cfg = COLTSConfig.colts5()
        assert cfg.k_max == 5
        assert cfg.d_w == 16


class TestCOLTSModel:
    def test_runs_without_error(self, alabama):
        """CO-LTS với config nhỏ phải chạy không lỗi."""
        cfg = COLTSConfig(
            k_max=2, d_w=5,
            outer_n=3, outer_max_iter=3,
            inner_m=5, inner_max_iter=5,
            n_runs=1, order=1
        )
        m = COLTSModel(cfg)
        m.fit(alabama.values, alabama.lb, alabama.ub)
        pred = m.predict()
        assert len(pred) == len(alabama.values) - 1

    def test_best_params_in_bounds(self, alabama):
        """best_params sau fit phải trong search space."""
        cfg = COLTSConfig(
            k_max=2, d_w=5,
            outer_n=3, outer_max_iter=3,
            inner_m=5, inner_max_iter=5,
            n_runs=1, order=1
        )
        m = COLTSModel(cfg)
        m.fit(alabama.values, alabama.lb, alabama.ub)
        assert m.best_params is not None
        assert 0.3 <= m.best_params.theta <= 0.7
        assert 0.3 <= m.best_params.alpha <= 0.7

    def test_best_words_count(self, alabama):
        """Số từ trong best_words phải bằng d_w."""
        cfg = COLTSConfig(
            k_max=2, d_w=5,
            outer_n=3, outer_max_iter=3,
            inner_m=5, inner_max_iter=5,
            n_runs=1, order=1
        )
        m = COLTSModel(cfg)
        m.fit(alabama.values, alabama.lb, alabama.ub)
        assert len(m.best_words) == 5

    def test_get_result_consistent(self, alabama):
        cfg = COLTSConfig(
            k_max=2, d_w=5,
            outer_n=3, outer_max_iter=3,
            inner_m=5, inner_max_iter=5,
            n_runs=1, order=1
        )
        m = COLTSModel(cfg)
        m.fit(alabama.values, alabama.lb, alabama.ub)
        result = m.get_result()
        assert len(result.forecasted) == len(alabama.values) - 1
        assert result.order == 1

    def test_mse_positive(self, alabama):
        cfg = COLTSConfig(
            k_max=2, d_w=5,
            outer_n=3, outer_max_iter=3,
            inner_m=5, inner_max_iter=5,
            n_runs=1, order=1
        )
        m = COLTSModel(cfg)
        m.fit(alabama.values, alabama.lb, alabama.ub)
        metrics = ForecastMetrics.compute(alabama.values[1:], m.predict())
        assert metrics.mse > 0
