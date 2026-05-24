#!/usr/bin/env python3
"""pyLTS CLI — chạy mô hình dự báo Linguistic Time Series.

Ví dụ nhanh:
  python3 main.py paper                  # tái hiện tất cả kết quả bài báo 2020
  python3 main.py paper --table 4
  python3 main.py paper --table holts    # HO-LTS (2021)
  python3 main.py paper --table pso      # LTS-PSO (2022)
  python3 main.py paper --table colts    # CO-LTS (2023)
  python3 main.py run --dataset alabama
  python3 main.py run --dataset alabama --model ho-lts --order 9 --specificity 4
  python3 main.py run --dataset alabama --model lts-pso --optimize
  python3 main.py run --dataset alabama --model co-lts --kmax 5 --dw 16
"""
from __future__ import annotations

import argparse
import sys


def cmd_paper(args) -> None:
    """Tái hiện kết quả từ các bài báo Nguyen Duy Hieu (2020–2023)."""
    from lts.experiments.paper_experiments import (
        print_all_results,
        run_colts_table,
        run_holts_table,
        run_pso_comparison,
        run_table4,
        run_table8,
        run_table9,
    )

    table = getattr(args, "table", None)

    if table is None:
        print_all_results()
        return

    if table == "4" or table == 4:
        out = run_table4()
        print(out.summary())
        print(out.forecast_table())
    elif table == "8" or table == 8:
        out = run_table8()
        print(out.summary())
    elif table == "9" or table == 9:
        results = run_table9()
        print("\n" + "=" * 60)
        print("  Table 9 — So sánh MAPE")
        print("=" * 60)
        for method, m in results.items():
            print(f"  {method:<30} MAPE={m.mape:>6.2f}%  MSE={m.mse:>10.3f}")
        print("=" * 60)
    elif table == "holts":
        print("\n" + "=" * 60)
        print("  HO-LTS (2021) — High-Order LTS trên Alabama")
        print("=" * 60)
        results = run_holts_table()
        for (order, n_words), m in sorted(results.items()):
            print(f"  λ={order}, {n_words:>2} words  MSE={m.mse:>10.0f}  MAPE={m.mape:>5.2f}%")
        print("=" * 60)
    elif table == "pso":
        print("\n" + "=" * 60)
        print("  LTS-PSO (2022) — PSO Parameter Optimization trên Alabama")
        print("  (Chạy PSO đầy đủ — có thể mất vài phút)")
        print("=" * 60)
        results = run_pso_comparison()
        for method, m in results.items():
            print(f"  {method:<45} MSE={m.mse:>10.0f}  MAPE={m.mape:>5.2f}%")
        print("=" * 60)
    elif table == "colts":
        print("\n" + "=" * 60)
        print("  CO-LTS (2023) — Co-Optimization trên Alabama")
        print("  (Chạy nested PSO đầy đủ — có thể mất vài phút)")
        print("=" * 60)
        results = run_colts_table()
        for name, m in results.items():
            print(f"  {name:<10} MSE={m.mse:>10.0f}  MAPE={m.mape:>5.2f}%")
        print("=" * 60)
    else:
        print(f"Table '{table}' không được hỗ trợ.")
        print("Dùng: --table 4, 8, 9, holts, pso, hoặc colts")
        sys.exit(1)


def cmd_run(args) -> None:
    """Chạy mô hình LTS trên dataset tuỳ chọn."""
    from lts import (
        COLTSConfig,
        COLTSModel,
        DataLoader,
        ForecastMetrics,
        HAParams,
        HOLTSModel,
        LTSModel,
        LTSPSOModel,
        LTSVariationsModel,
    )

    # Load dataset
    if args.dataset:
        dataset = DataLoader.bundled(args.dataset)
    elif args.csv:
        dataset = DataLoader.from_csv(
            args.csv,
            column=args.column,
            lb=args.lb,
            ub=args.ub,
        )
    elif args.txt:
        dataset = DataLoader.from_txt(args.txt)
    else:
        print("Cần chỉ định --dataset, --csv, hoặc --txt.")
        sys.exit(1)

    params = HAParams(theta=args.theta, alpha=args.alpha)
    data, lb, ub = dataset.values, dataset.lb, dataset.ub
    model_name = getattr(args, "model", "lts")

    print(f"\nDataset: {dataset.name}  (n={len(data)}, lb={lb}, ub={ub})")

    # Build and fit model
    if model_name == "ho-lts":
        model = HOLTSModel(
            params=params,
            order=args.order,
            specificity=args.specificity,
        )
        model.fit(data, lb, ub)
        forecasted = model.predict()
        actual = data[args.order:]
        print(f"Model: HO-LTS  order={args.order}  specificity={args.specificity}  words={len(model.words)}")

    elif model_name == "lts-pso":
        from lts.optimization.pso import PSOConfig
        m_pso = LTSPSOModel(params=params, specificity=args.specificity, order=args.order)
        if getattr(args, "optimize", False):
            print("Model: LTS-PSO (chạy PSO tối ưu — có thể mất vài phút...)")
            cfg = PSOConfig(
                n_particles=300, max_iter=1000, omega=0.4, c1=2.0, c2=2.0,
                bounds=[(0.3, 0.7), (0.3, 0.7)]
            )
            best = m_pso.fit_optimize(data, lb, ub, pso_config=cfg, n_runs=3)
            print(f"  Best params: theta={best.theta:.4f}, alpha={best.alpha:.4f}")
        else:
            print(f"Model: LTS-PSO  theta={params.theta}  alpha={params.alpha}  (không PSO)")
            m_pso.fit(data, lb, ub)
        model = m_pso
        forecasted = model.predict()
        actual = data[args.order:]

    elif model_name == "co-lts":
        kmax = getattr(args, "kmax", 3)
        dw = getattr(args, "dw", 7)
        cfg = COLTSConfig(k_max=kmax, d_w=dw)
        print(f"Model: CO-LTS  k_max={kmax}  d_w={dw}  (chạy nested PSO — có thể mất vài phút...)")
        model = COLTSModel(cfg)
        model.fit(data, lb, ub)
        forecasted = model.predict()
        actual = data[cfg.order:]
        print(f"  Best params: {model.best_params}")
        print(f"  Best words: {model.best_words}")

    elif args.variations:
        model = LTSVariationsModel(
            params=params,
            lb_variation=args.lb_var,
            ub_variation=args.ub_var,
            specificity=args.specificity,
            order=args.order,
            use_repeat=args.repeat,
        )
        model.fit(data)
        forecasted = model.predict()
        actual = model.actual_for_comparison
        print(f"Model: LTS Variations  order={args.order}")

    else:
        model = LTSModel(
            params=params,
            specificity=args.specificity,
            order=args.order,
            use_repeat=args.repeat,
        )
        model.fit(data, lb, ub)
        forecasted = model.predict()
        actual = data[args.order:]
        result = model.get_result()
        print(f"Model: LTS  order={args.order}  specificity={args.specificity}")
        print(f"HAParams: theta={params.theta}, alpha={params.alpha}")
        print(f"Words ({len(model.words)}): {model.words}")
        print("\nSemantic points:")
        for w, v in sorted(result.semantic_points.items(), key=lambda x: x[1]):
            print(f"  {w:>6}: {v:.2f}")
        print(f"\n{result.n_rules} LLRGs:")
        for line in result.llrg_summary():
            print(f"  {line}")

    # Metrics
    mets = ForecastMetrics.compute(actual, forecasted, mape_digits=2)
    print(f"\nResults: {len(forecasted)} forecasts")
    if args.show:
        print("Actual  :", [round(v, 2) for v in actual])
        print("Forecast:", [round(v, 2) for v in forecasted])
    print(f"\nMSE  = {mets.mse:.3f}")
    print(f"RMSE = {mets.rmse:.3f}")
    print(f"MAE  = {mets.mae:.3f}")
    print(f"MAPE = {mets.mape}%")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python3 main.py",
        description="pyLTS — Linguistic Time Series Forecasting",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # --- paper subcommand ---
    pp = sub.add_parser("paper", help="Tái hiện kết quả các bài báo 2020–2023")
    pp.add_argument(
        "--table",
        metavar="TABLE",
        help="Bảng cụ thể: 4, 8, 9, holts, pso, colts",
    )

    # --- run subcommand ---
    rp = sub.add_parser("run", help="Chạy mô hình LTS trên dataset tuỳ chọn")

    src = rp.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset", metavar="NAME", help="Dataset bundled (vd: alabama, taifex_1998)")
    src.add_argument("--csv", metavar="FILE", help="File CSV")
    src.add_argument("--txt", metavar="FILE", help="File .txt định dạng 3 dòng")

    rp.add_argument("--column", metavar="COL", help="Tên cột trong CSV")
    rp.add_argument("--lb", type=float, help="Cận dưới universe of discourse")
    rp.add_argument("--ub", type=float, help="Cận trên universe of discourse")

    rp.add_argument(
        "--model",
        default="lts",
        choices=["lts", "ho-lts", "lts-pso", "co-lts"],
        help="Mô hình dự báo: lts (mặc định), ho-lts, lts-pso, co-lts",
    )
    rp.add_argument("--theta", type=float, default=0.57, metavar="θ", help="HAParams theta (mặc định: 0.57)")
    rp.add_argument("--alpha", type=float, default=0.49, metavar="α", help="HAParams alpha (mặc định: 0.49)")
    rp.add_argument("--specificity", type=int, default=1, help="Mức từ vựng: 1=7, 2=17, 3=33, 4=65 từ (mặc định: 1)")
    rp.add_argument("--order", type=int, default=1, help="Bậc mô hình LTS (mặc định: 1)")
    rp.add_argument("--repeat", action="store_true", help="Cho phép RHS trùng lặp trong LLRG")

    rp.add_argument("--variations", action="store_true", help="Dùng mô hình variation series (chỉ với --model lts)")
    rp.add_argument("--lb-var", type=float, default=-1000.0, help="Cận dưới universe of variation (mặc định: -1000)")
    rp.add_argument("--ub-var", type=float, default=1400.0, help="Cận trên universe of variation (mặc định: 1400)")

    # LTS-PSO specific
    rp.add_argument("--optimize", action="store_true", help="[lts-pso] Chạy PSO tối ưu tham số")

    # CO-LTS specific
    rp.add_argument("--kmax", type=int, default=3, help="[co-lts] Độ sâu hedge tối đa (mặc định: 3)")
    rp.add_argument("--dw", type=int, default=7, help="[co-lts] Số từ được chọn (mặc định: 7)")

    rp.add_argument("--show", action="store_true", help="Hiển thị danh sách giá trị actual và forecast")

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "paper":
        cmd_paper(args)
    elif args.command == "run":
        cmd_run(args)


if __name__ == "__main__":
    main()
