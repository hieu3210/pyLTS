"""Tái hiện các thực nghiệm trong các bài báo Nguyen Duy Hieu (2020–2023).

Mỗi hàm tái hiện một bảng kết quả cụ thể trong bài báo và trả về
ExperimentOutput hoặc dict kết quả đầy đủ.
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


def run_holts_table(
    dataset_name: str = "alabama",
    orders: list[int] | None = None,
    specificities: list[int] | None = None,
) -> dict[str, object]:
    """Tái hiện thực nghiệm Table 4 bài báo HO-LTS (hieund_2021).

    So sánh HOLTSModel qua các bậc λ và các mức từ vựng.

    Returns
    -------
    dict[(order, n_words), ForecastMetrics]
    """
    from lts.core.hedge_algebras import HAParams, HedgeAlgebra
    from lts.data.loader import DataLoader
    from lts.metrics.measures import ForecastMetrics
    from lts.models.ho_lts import HOLTSModel

    if orders is None:
        orders = [2, 3, 4, 5, 6, 7, 8, 9]
    if specificities is None:
        specificities = [2, 3, 4]

    dataset = DataLoader.bundled(dataset_name)
    data, lb, ub = dataset.values, dataset.lb, dataset.ub

    results: dict[tuple[int, int], ForecastMetrics] = {}
    for spec in specificities:
        params = HOLTSModel.params_for_specificity(spec)
        ha_tmp = HedgeAlgebra(params)
        n_words = len(ha_tmp.get_words(spec))
        for order in orders:
            if order >= len(data):
                continue
            m = HOLTSModel(params, order=order, specificity=spec)
            m.fit(data, lb, ub)
            actual = data[order:]
            results[(order, n_words)] = ForecastMetrics.compute(actual, m.predict())

    return results


def run_pso_comparison(
    dataset_name: str = "alabama",
    pso_config: object | None = None,
    n_runs: int = 3,
) -> dict[str, object]:
    """Tái hiện thực nghiệm bài báo LTS-PSO (hieund_2022).

    So sánh LTSPSOModel (sau tối ưu PSO) với LTSModel gốc.

    Returns
    -------
    dict[str, ForecastMetrics]  — 'LTS-PSO (optimized)', 'LTS (baseline)'
    """
    from lts.core.hedge_algebras import HAParams
    from lts.data.loader import DataLoader
    from lts.metrics.measures import ForecastMetrics
    from lts.models.lts_model import LTSModel
    from lts.models.lts_pso import LTSPSOModel
    from lts.optimization.pso import PSOConfig

    dataset = DataLoader.bundled(dataset_name)
    data, lb, ub = dataset.values, dataset.lb, dataset.ub

    # LTS-PSO với PSO tối ưu (paper: N=300, G_max=1000)
    cfg = pso_config or PSOConfig(
        n_particles=300, max_iter=1000, omega=0.4, c1=2.0, c2=2.0,
        bounds=[(0.3, 0.7), (0.3, 0.7)]
    )
    m_pso = LTSPSOModel(HAParams(theta=0.5, alpha=0.5), specificity=2, order=1)
    best = m_pso.fit_optimize(data, lb, ub, pso_config=cfg, n_runs=n_runs)

    # LTS baseline
    m_lts = LTSModel(HAParams.enrollment(), specificity=1, order=1, use_repeat=False)
    m_lts.fit(data, lb, ub)

    return {
        f"LTS-PSO (theta={best.theta:.4f}, alpha={best.alpha:.4f})":
            ForecastMetrics.compute(data[1:], m_pso.predict()),
        "LTS (baseline, 2020)":
            ForecastMetrics.compute(data[1:], m_lts.predict()),
    }


def run_colts_table(
    dataset_name: str = "alabama",
    variants: list[str] | None = None,
) -> dict[str, object]:
    """Tái hiện thực nghiệm bài báo CO-LTS (hieund_2023).

    So sánh COLTS3/4/5 trên dataset.

    Returns
    -------
    dict[str, ForecastMetrics]  — 'COLTS3', 'COLTS4', 'COLTS5'
    """
    from lts.data.loader import DataLoader
    from lts.metrics.measures import ForecastMetrics
    from lts.models.co_lts import COLTSConfig, COLTSModel

    if variants is None:
        variants = ["COLTS3", "COLTS4", "COLTS5"]

    dataset = DataLoader.bundled(dataset_name)
    data, lb, ub = dataset.values, dataset.lb, dataset.ub

    preset_map = {
        "COLTS3": COLTSConfig.colts3(),
        "COLTS4": COLTSConfig.colts4(),
        "COLTS5": COLTSConfig.colts5(),
    }

    results: dict[str, ForecastMetrics] = {}
    for name in variants:
        cfg = preset_map[name]
        m = COLTSModel(cfg)
        m.fit(data, lb, ub)
        actual = data[cfg.order:]
        results[name] = ForecastMetrics.compute(actual, m.predict())

    return results


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
