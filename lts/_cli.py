#!/usr/bin/env python3
"""pyLTS CLI — chạy mô hình dự báo Linguistic Time Series.

Ví dụ nhanh:
  python main.py paper          # tái hiện tất cả kết quả trong bài báo 2020
  python main.py paper --table 4
  python main.py run --dataset alabama
  python main.py run --csv datasets/TAIEX.csv --column Close --lb 6000 --ub 12000
"""
from __future__ import annotations

import argparse
import sys


def cmd_paper(args) -> None:
    """Tái hiện kết quả từ bài báo Nguyen Duy Hieu et al. (2020)."""
    from lts.experiments.paper_experiments import (
        print_all_results,
        run_table4,
        run_table8,
        run_table9,
    )

    table = getattr(args, "table", None)

    if table is None:
        print_all_results()
        return

    if table == 4:
        out = run_table4()
        print(out.summary())
        print(out.forecast_table())
    elif table == 8:
        out = run_table8()
        print(out.summary())
    elif table == 9:
        results = run_table9()
        print("\n" + "=" * 60)
        print("  Table 9 — So sánh MAPE")
        print("=" * 60)
        for method, m in results.items():
            print(f"  {method:<30} MAPE={m.mape:>6.2f}%  MSE={m.mse:>10.3f}")
        print("=" * 60)
    else:
        print(f"Table {table} không được hỗ trợ. Dùng --table 4, 8, hoặc 9.")
        sys.exit(1)


def cmd_run(args) -> None:
    """Chạy mô hình LTS trên dataset tuỳ chọn."""
    from lts import (
        DataLoader,
        ExperimentConfig,
        ExperimentRunner,
        ForecastMetrics,
        HAParams,
        LTSModel,
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

    # Build and fit model
    if args.variations:
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

        # Print semantic points and rules
        print(f"\nDataset: {dataset.name}  (n={len(data)}, lb={lb}, ub={ub})")
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
        prog="python main.py",
        description="pyLTS — Linguistic Time Series Forecasting",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # --- paper subcommand ---
    pp = sub.add_parser("paper", help="Tái hiện kết quả bài báo 2020")
    pp.add_argument("--table", type=int, choices=[4, 8, 9], help="Chỉ in bảng cụ thể (4, 8, 9)")

    # --- run subcommand ---
    rp = sub.add_parser("run", help="Chạy LTSModel trên dataset tuỳ chọn")

    src = rp.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset", metavar="NAME", help="Dataset bundled (vd: alabama, taifex_1998)")
    src.add_argument("--csv", metavar="FILE", help="File CSV")
    src.add_argument("--txt", metavar="FILE", help="File .txt định dạng 3 dòng")

    rp.add_argument("--column", metavar="COL", help="Tên cột trong CSV (mặc định: cột số đầu tiên)")
    rp.add_argument("--lb", type=float, help="Cận dưới universe of discourse")
    rp.add_argument("--ub", type=float, help="Cận trên universe of discourse")

    rp.add_argument("--theta", type=float, default=0.57, metavar="θ", help="HAParams theta (mặc định: 0.57)")
    rp.add_argument("--alpha", type=float, default=0.49, metavar="α", help="HAParams alpha (mặc định: 0.49)")
    rp.add_argument("--specificity", type=int, default=1, help="Mức từ vựng: 1=7 từ, 2=15 từ, 3=33 từ (mặc định: 1)")
    rp.add_argument("--order", type=int, default=1, help="Bậc mô hình LTS (mặc định: 1)")
    rp.add_argument("--repeat", action="store_true", help="Cho phép RHS trùng lặp trong LLRG")

    rp.add_argument("--variations", action="store_true", help="Dùng mô hình variation series")
    rp.add_argument("--lb-var", type=float, default=-1000.0, help="Cận dưới universe of variation (mặc định: -1000)")
    rp.add_argument("--ub-var", type=float, default=1400.0, help="Cận trên universe of variation (mặc định: 1400)")

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
