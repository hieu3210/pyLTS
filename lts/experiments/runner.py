"""ExperimentRunner — orchestrator pipeline cho một thực nghiệm."""

from __future__ import annotations

from dataclasses import dataclass, field

from lts.config.experiment_config import ExperimentConfig
from lts.data.loader import DataLoader, Dataset
from lts.metrics.measures import ForecastMetrics
from lts.models.base import ForecastResult
from lts.models.chen1996 import Chen1996
from lts.models.lts_model import LTSModel
from lts.models.lts_variations_model import LTSVariationsModel
from lts.models.song_chissom1993 import SongChissom1993


@dataclass
class ExperimentOutput:
    """Toàn bộ kết quả của một lần chạy thực nghiệm.

    Attributes
    ----------
    config : ExperimentConfig
        Cấu hình đã dùng.
    dataset : Dataset
        Dataset đã tải.
    result : ForecastResult
        Kết quả chi tiết từ LTS model.
    metrics : ForecastMetrics
        Các độ đo sai số của LTS model.
    baseline_chen : ForecastMetrics | None
        Độ đo sai số của Chen [1996] (nếu có).
    baseline_song : ForecastMetrics | None
        Độ đo sai số của Song & Chissom [1993] (nếu có).
    """

    config: ExperimentConfig
    dataset: Dataset
    result: ForecastResult
    metrics: ForecastMetrics
    baseline_chen: ForecastMetrics | None = None
    baseline_song: ForecastMetrics | None = None

    def summary(self) -> str:
        """Trả về bảng tóm tắt kết quả thực nghiệm."""
        lines = [
            f"\n{'='*60}",
            f"  {self.config.label}",
            f"{'='*60}",
            f"  Dataset : {self.dataset.name} (n={len(self.dataset)})",
            f"  HAParams: theta={self.config.params.theta}, "
            f"alpha={self.config.params.alpha}",
            f"  Order   : {self.config.order}",
            f"  Words   : {self.result.n_rules} LLRGs",
            f"",
            f"  --- Kết quả LTS Model ---",
            f"  MSE  : {self.metrics.mse:.4f}",
            f"  RMSE : {self.metrics.rmse:.4f}",
            f"  MAE  : {self.metrics.mae:.4f}",
            f"  MAPE : {self.metrics.mape}%",
            f"  SMAPE: {self.metrics.smape}%",
        ]
        if self.baseline_chen:
            lines += [
                f"",
                f"  --- Chen [1996] Baseline ---",
                f"  MSE  : {self.baseline_chen.mse:.4f}",
                f"  MAPE : {self.baseline_chen.mape}%",
            ]
        if self.baseline_song:
            lines += [
                f"",
                f"  --- Song & Chissom [1993] Baseline ---",
                f"  MSE  : {self.baseline_song.mse:.4f}",
                f"  MAPE : {self.baseline_song.mape}%",
            ]
        lines.append(f"{'='*60}\n")
        return "\n".join(lines)

    def forecast_table(self) -> str:
        """Bảng so sánh actual vs forecasted (giống Table 4 trong bài báo)."""
        result = self.result
        header = f"{'Year':>6} {'Actual':>10} {'Forecasted':>12} {'Error%':>8}"
        sep = "-" * len(header)
        rows = [header, sep]
        for i, (a, f) in enumerate(zip(result.actual, result.forecasted)):
            err_pct = abs(f - a) / abs(a) * 100 if a != 0 else 0.0
            rows.append(f"{1971 + result.order + i:>6} {a:>10.0f} {f:>12.2f} {err_pct:>7.2f}%")
        rows.append(sep)
        rows.append(f"{'MSE':>6} {self.metrics.mse:>23.4f}")
        rows.append(f"{'MAPE':>6} {self.metrics.mape:>22.2f}%")
        return "\n".join(rows)


class ExperimentRunner:
    """Orchestrator: nhận ExperimentConfig, chạy pipeline, trả về kết quả.

    Sử dụng:
    --------
    >>> config = ExperimentConfig.paper_table4()
    >>> output = ExperimentRunner(config).run(include_baselines=True)
    >>> print(output.summary())
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def run(self, include_baselines: bool = False) -> ExperimentOutput:
        """Chạy thực nghiệm và trả về ExperimentOutput đầy đủ."""
        dataset = self._load_dataset()
        result, metrics = self._run_lts(dataset)
        output = ExperimentOutput(
            config=self.config,
            dataset=dataset,
            result=result,
            metrics=metrics,
        )
        if include_baselines:
            output.baseline_chen = self._run_chen(dataset)
            output.baseline_song = self._run_song(dataset)
        return output

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_dataset(self) -> Dataset:
        cfg = self.config
        if cfg.dataset_path:
            return DataLoader.from_txt(cfg.dataset_path, name=cfg.dataset_name)
        return DataLoader.bundled(cfg.dataset_name)

    def _run_lts(self, dataset: Dataset) -> tuple[ForecastResult, ForecastMetrics]:
        cfg = self.config

        if cfg.use_variations:
            assert cfg.lb_variation is not None and cfg.ub_variation is not None, (
                "lb_variation và ub_variation phải được đặt khi use_variations=True"
            )
            model = LTSVariationsModel(
                params=cfg.params,
                lb_variation=cfg.lb_variation,
                ub_variation=cfg.ub_variation,
                specificity=cfg.specificity,
                order=cfg.order,
                use_repeat=cfg.use_repeat,
            )
            model.fit(dataset.values)
            forecasted = model.predict()
            actual = model.actual_for_comparison
            result = model.get_variation_result()
        else:
            model = LTSModel(
                params=cfg.params,
                specificity=cfg.specificity,
                order=cfg.order,
                use_repeat=cfg.use_repeat,
                words=cfg.words,
            )
            model.fit(dataset.values, cfg.lb, cfg.ub)
            forecasted = model.predict()
            actual = dataset.values[cfg.order:]
            result = model.get_result()

        metrics = ForecastMetrics.compute(actual, forecasted, cfg.mape_digits)
        return result, metrics

    def _run_chen(self, dataset: Dataset) -> ForecastMetrics:
        cfg = self.config
        n = len(cfg.words) if cfg.words else 7
        model = Chen1996(n_intervals=n, order=cfg.order)
        model.fit(dataset.values, cfg.lb, cfg.ub)
        actual = dataset.values[cfg.order:]
        return ForecastMetrics.compute(actual, model.predict(), cfg.mape_digits)

    def _run_song(self, dataset: Dataset) -> ForecastMetrics:
        cfg = self.config
        model = SongChissom1993(n_intervals=7)
        model.fit(dataset.values, cfg.lb, cfg.ub)
        actual = dataset.values[1:]
        return ForecastMetrics.compute(actual, model.predict(), cfg.mape_digits)
