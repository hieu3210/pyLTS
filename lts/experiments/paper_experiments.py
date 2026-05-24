"""Tái hiện các thực nghiệm trong bài báo Nguyen Duy Hieu (2020).

Mỗi hàm tái hiện một bảng kết quả cụ thể trong bài báo và trả về
ExperimentOutput đầy đủ, có thể dùng để in hoặc phân tích thêm.
"""

from __future__ import annotations

from lts.config.experiment_config import ExperimentConfig
from lts.experiments.runner import ExperimentOutput, ExperimentRunner


def run_table4() -> ExperimentOutput:
    """Tái hiện Table 4: dự báo enrollment bằng LTS.

    Acceptance criteria từ bài báo:
    - MSE  = 262.326
    - MAPE ≈ 1.27%
    """
    config = ExperimentConfig.paper_table4()
    return ExperimentRunner(config).run(include_baselines=True)


def run_table8() -> ExperimentOutput:
    """Tái hiện Table 8: dự báo trên chuỗi variation.

    Acceptance criteria từ bài báo:
    - MSE  = 65.029
    - MAPE = 1.27%
    """
    config = ExperimentConfig.paper_table8()
    return ExperimentRunner(config).run()


def run_table9() -> dict[str, object]:
    """Tái hiện Table 9: so sánh MAPE giữa các phương pháp.

    Returns
    -------
    dict[str, object]
        {method_name: ForecastMetrics}
    """
    from lts.data.loader import DataLoader
    from lts.metrics.measures import ForecastMetrics
    from lts.models.chen1996 import Chen1996
    from lts.models.lts_model import LTSModel
    from lts.models.song_chissom1993 import SongChissom1993
    from lts.core.hedge_algebras import HAParams

    dataset = DataLoader.bundled("alabama")
    data, lb, ub = dataset.values, dataset.lb, dataset.ub

    results: dict[str, ForecastMetrics] = {}

    # LTS (proposed)
    lts = LTSModel(params=HAParams.enrollment(), specificity=1, order=1)
    lts.fit(data, lb, ub)
    results["LTS (Proposed)"] = ForecastMetrics.compute(data[1:], lts.predict())

    # Chen [1996]
    chen = Chen1996(n_intervals=7, order=1)
    chen.fit(data, lb, ub)
    results["Chen [1996]"] = ForecastMetrics.compute(data[1:], chen.predict())

    # Song & Chissom [1993]
    song = SongChissom1993(n_intervals=7)
    song.fit(data, lb, ub)
    results["Song & Chissom [1993]"] = ForecastMetrics.compute(data[1:], song.predict())

    return results


def run_all() -> dict[str, ExperimentOutput | dict]:
    """Chạy tất cả thực nghiệm trong bài báo.

    Returns
    -------
    dict
        {'table4': ExperimentOutput, 'table8': ExperimentOutput,
         'table9': dict[str, ForecastMetrics]}
    """
    return {
        "table4": run_table4(),
        "table8": run_table8(),
        "table9": run_table9(),
    }


def print_all_results() -> None:
    """In toàn bộ kết quả thực nghiệm ra console."""
    print("\n" + "=" * 60)
    print("  THỰC NGHIỆM: Enrollment Forecasting Based on LTS")
    print("  Nguyen Duy Hieu et al. (2020)")
    print("=" * 60)

    t4 = run_table4()
    print(t4.summary())
    print("\n  --- Bảng dự báo chi tiết (Table 4) ---")
    print(t4.forecast_table())

    t8 = run_table8()
    print(t8.summary())

    t9 = run_table9()
    print("\n" + "=" * 60)
    print("  So sánh MAPE (Table 9)")
    print("=" * 60)
    for method, metrics in t9.items():
        print(f"  {method:<30} MAPE={metrics.mape:>6.2f}%  MSE={metrics.mse:>10.3f}")
    print("=" * 60 + "\n")
