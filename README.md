# pyLTS — Linguistic Time Series Forecasting

Python implementation của mô hình dự báo chuỗi thời gian ngôn ngữ (LTS) dựa trên Hedge Algebras, chuẩn hóa theo bài báo:

> **Nguyen Duy Hieu, Nguyen Cat Ho, Vu Nhu Lan (2020).**  
> "Enrollment Forecasting Based on Linguistic Time Series."  
> *Journal of Computer Science and Cybernetics*, V.36, N.2.  
> DOI: [10.15625/1813-9663/36/2/14396](https://doi.org/10.15625/1813-9663/36/2/14396)

---

## Ý tưởng cốt lõi

Thay vì dùng tập mờ (fuzzy sets) với hàm membership tùy ý, pyLTS mã hóa dữ liệu bằng **ngôn ngữ tự nhiên** theo cấu trúc **Hedge Algebras (HA)** — một đại số ngôn ngữ có nền tảng toán học chặt chẽ.

Từ vựng mặc định gồm 7 nhãn: `Very Small`, `Small`, `Rather Small`, `Medium`, `Rather Large`, `Large`, `Very Large` — ánh xạ tới các điểm ngữ nghĩa xác định bởi tham số `(θ, α)`.

**Pipeline 6 bước:**
1. Định nghĩa Hedge Algebra với tham số `(θ, α)`
2. Tính SQM — ánh xạ từ → `[0, 1]`
3. Tính semantic points — ánh xạ từ → `[lb, ub]`
4. Fuzzification — gán nhãn ngôn ngữ cho từng điểm dữ liệu
5. Xây dựng LLRGs (Linguistic Logical Relationship Groups)
6. Dự báo — trung bình semantic points của RHS

---

## Cài đặt

```bash
# Clone repository
git clone https://github.com/hieu3210/pyLTS.git
cd pyLTS

# Tạo virtual environment và cài đặt
python3 -m venv .venv
source .venv/bin/activate          # macOS/Linux
# .venv\Scripts\activate           # Windows

pip install -e ".[dev]"            # cài đặt cả pytest cho phát triển
```

**Yêu cầu:** Python ≥ 3.11. Không cần thư viện ngoài cho phần lõi.

---

## Chạy nhanh

### Tái hiện kết quả bài báo

```bash
# Tất cả bảng kết quả (Table 4, 8, 9)
python3 main.py paper

# Chỉ Table 4 — Enrollment forecasting
python3 main.py paper --table 4

# Chỉ Table 8 — Variation-series forecasting
python3 main.py paper --table 8

# Chỉ Table 9 — So sánh với Chen và Song & Chissom
python3 main.py paper --table 9
```

### Chạy trên dataset tuỳ chọn

```bash
# Dataset bundled
python3 main.py run --dataset alabama
python3 main.py run --dataset taifex_1998

# Từ file CSV (cột Close, universe [6000, 12000])
python3 main.py run --csv datasets/TAIEX.csv --column Close --lb 6000 --ub 12000

# Tham số tùy chỉnh
python3 main.py run --dataset alabama --theta 0.55 --alpha 0.52 --order 2

# Mô hình variation series
python3 main.py run --dataset alabama --variations --lb-var -1000 --ub-var 1400

# Hiển thị danh sách dự báo
python3 main.py run --dataset alabama --show
```

### Trong Python

```python
from lts import HAParams, LTSModel, ForecastMetrics, DataLoader

# Tải dataset
dataset = DataLoader.bundled("alabama")

# Khởi tạo và fit mô hình
params = HAParams.enrollment()          # θ=0.57, α=0.49
model = LTSModel(params, specificity=1, order=1, use_repeat=False)
model.fit(dataset.values, dataset.lb, dataset.ub)

# Dự báo và đánh giá
forecasted = model.predict()
actual = dataset.values[1:]
metrics = ForecastMetrics.compute(actual, forecasted)
print(f"MSE={metrics.mse:.0f}, MAPE={metrics.mape}%")

# Xem chi tiết
result = model.get_result()
print("Semantic points:", result.semantic_points)
for line in result.llrg_summary():
    print(" ", line)
```

### Chạy experiments có sẵn

```python
from lts import ExperimentConfig, ExperimentRunner

config = ExperimentConfig.paper_table4()
output = ExperimentRunner(config).run(include_baselines=True)
print(output.summary())
print(output.forecast_table())
```

---

## Chạy tests

```bash
python3 -m pytest tests/ -v
```

Kết quả kỳ vọng: **75/75 passed**.

---

## Cấu trúc project

```
pyLTS/
├── lts/                           # Package chính (pip install -e .)
│   ├── core/
│   │   ├── hedge_algebras.py      # HAParams + HedgeAlgebra (SQM, fm, sign, omega)
│   │   └── sqm_formulas.py        # Công thức closed-form (eqs 4.1–4.7 bài báo)
│   ├── models/
│   │   ├── base.py                # BaseForecaster ABC + ForecastResult
│   │   ├── lts_model.py           # LTSModel — thuật toán 6 bước
│   │   ├── lts_variations_model.py# Dự báo trên chuỗi sai phân
│   │   ├── chen1996.py            # Baseline: Chen [1996]
│   │   └── song_chissom1993.py    # Baseline: Song & Chissom [1993]
│   ├── metrics/
│   │   └── measures.py            # ForecastMetrics: MSE, MAE, RMSE, MAPE, SMAPE
│   ├── data/
│   │   ├── loader.py              # DataLoader: bundled / txt / csv / list
│   │   ├── transforms.py          # DataTransformer: chuỗi sai phân
│   │   └── datasets/              # Datasets bundled
│   │       ├── alabama.txt        # Alabama enrollment (1971–1992)
│   │       ├── taifex_1998.txt    # TAIFEX 1998
│   │       ├── temperature.txt    # Dữ liệu nhiệt độ
│   │       └── ...
│   ├── config/
│   │   └── experiment_config.py   # ExperimentConfig + presets paper
│   └── experiments/
│       ├── runner.py              # ExperimentRunner: orchestrator
│       └── paper_experiments.py   # run_table4/8/9(), print_all_results()
├── tests/                         # 75 tests (pytest)
├── datasets/                      # CSV datasets cho nghiên cứu tương lai
│   ├── TAIEX.csv
│   ├── Enrollments.csv
│   └── ...
├── publications/                  # Bài báo gốc (PDF + TXT)
│   ├── hieund_2020.pdf            # Bài báo 2020 (LTS)
│   ├── hieund_2021.pdf            # Bài báo 2021
│   └── ...
├── main.py                        # CLI entry point
└── pyproject.toml
```

---

## Kết quả thực nghiệm

### LTS (2020) — Alabama Enrollment

| Mô hình | MSE | MAPE |
|---|---|---|
| **LTS (đề xuất)** | **262,211** | **2.57%** |
| Chen [1996] | 407,521 | 3.11% |
| Song & Chissom [1993] | 806,087 | 3.76% |

> Semantic points: `{V−: 14,038 · −: 15,035 · L−: 16,032 · W: 16,990 · L+: 17,713 · +: 18,465 · V+: 19,217}`.

---

## Mô hình mở rộng

### HO-LTS (2021) — High-Order LTS

Mô hình bậc cao (λ=1..9) với fallback rule cải tiến: khi không tìm thấy LLR group cho một LHS, thay vì lấy từ cuối, **HO-LTS lấy trung bình semantic points của TẤT CẢ từ trong LHS**.

**Kết quả tốt nhất (Alabama, λ=9, 65 từ):** MSE ≈ 283, AFE ≈ 0.07%.

```python
from lts import HOLTSModel, DataLoader

ds = DataLoader.bundled("alabama")

# Bậc 2, 17 từ
m = HOLTSModel(HOLTSModel.params_for_word_count(17), order=2, specificity=2)
m.fit(ds.values, ds.lb, ds.ub)
print(f"HO-LTS order=2: {m.predict()[:3]}")

# Bậc 9, 65 từ (tốt nhất theo bài báo)
m9 = HOLTSModel(HOLTSModel.params_for_word_count(65), order=9, specificity=4)
m9.fit(ds.values, ds.lb, ds.ub)
```

```bash
python3 main.py run --dataset alabama --model ho-lts --order 9 --specificity 4

# Tái hiện bảng HO-LTS (tất cả orders × word counts)
python3 main.py paper --table holts
```

**Tham số từ bài báo:**

| Số từ (code) | specificity | theta | alpha |
|---|---|---|---|
| 7 | 1 | 0.437 | 0.511 |
| 15 | 2 | 0.527 | 0.412 |
| 31 | 3 | 0.65 | 0.35 |
| 63 | 4 | 0.65 | 0.35 |

> Lưu ý: Bài báo dùng 9/17/33/65 từ với cấu trúc HA hơi khác; code này dùng 7/15/31/63 từ nhưng cùng params.

---

### LTS-PSO (2022) — PSO Parameter Optimization

PSO tối ưu `(theta, alpha)` để minimize MSE trên training data. Công thức dự báo được cải tiến:

```
Có rule:    forecast = 0.5 × (s_lhs_last + mean(s_RHS))
Không rule: forecast = s_lhs_last
```

**Kết quả (Alabama):** MSE ≈ 22,403 (theta=0.4789, alpha=0.608).

```python
from lts import LTSPSOModel, HAParams, DataLoader
from lts.optimization.pso import PSOConfig

ds = DataLoader.bundled("alabama")

# Dự báo với params cố định (không PSO)
m = LTSPSOModel(HAParams(theta=0.4789, alpha=0.608), specificity=2)
m.fit(ds.values, ds.lb, ds.ub)

# Tối ưu PSO (paper: N=300, G_max=1000, chạy 3 lần)
m_opt = LTSPSOModel(HAParams(theta=0.5, alpha=0.5), specificity=2)
cfg = PSOConfig(n_particles=300, max_iter=1000, omega=0.4, c1=2.0, c2=2.0,
                bounds=[(0.3, 0.7), (0.3, 0.7)])
best_params = m_opt.fit_optimize(ds.values, ds.lb, ds.ub, pso_config=cfg, n_runs=3)
print(f"Best: theta={best_params.theta:.4f}, alpha={best_params.alpha:.4f}")
```

```bash
python3 main.py run --dataset alabama --model lts-pso --optimize

# Tái hiện bảng LTS-PSO (chạy PSO đầy đủ — vài phút)
python3 main.py paper --table pso
```

---

### CO-LTS (2023) — Co-Optimization

PSO lồng nhau đồng tối ưu cả tham số HA lẫn tập từ vựng:
- **Outer PSO**: tối ưu `(theta, alpha)` — N=20 particles, 30 vòng.
- **Inner PSO (UWO)**: tối ưu chọn `d_w` từ từ W_all — M=30 particles, 100 vòng.

**Kết quả Alabama (5 lần chạy, lấy best):**

| Variant | k_max | d_w | MSE (paper) |
|---|---|---|---|
| COLTS3 | 3 | 7 | ≈ 47,628 |
| COLTS4 | 4 | 14 | ≈ 16,344 |
| COLTS5 | 5 | 16 | ≈ 6,332 |

```python
from lts import COLTSModel, COLTSConfig, DataLoader

ds = DataLoader.bundled("alabama")

# COLTS5 — kết quả tốt nhất
cfg = COLTSConfig.colts5()   # k_max=5, d_w=16, n_runs=5
m = COLTSModel(cfg)
m.fit(ds.values, ds.lb, ds.ub)
print(f"Best params: {m.best_params}")
print(f"Best words:  {m.best_words}")
print(f"Best MSE:    {m.best_mse:.0f}")
```

```bash
python3 main.py run --dataset alabama --model co-lts --kmax 5 --dw 16

# Tái hiện bảng CO-LTS (chạy nested PSO — vài phút)
python3 main.py paper --table colts
```

---

## Phát triển mô hình mới

### Thêm model mới

Implement `BaseForecaster` và trả về `ForecastResult`:

```python
from lts.models.base import BaseForecaster, ForecastResult

class MyNewModel(BaseForecaster):
    def fit(self, data: list[float], lb: float, ub: float) -> None:
        ...

    def predict(self) -> list[float]:
        ...

    def get_result(self) -> ForecastResult:
        ...
```

### PSO optimizer dùng chung

`lts/optimization/pso.py` là generic PSO có thể dùng cho bất kỳ objective function nào:

```python
from lts.optimization.pso import PSO, PSOConfig

def my_objective(position: list[float]) -> float:
    theta, alpha = position
    # ... fit model, compute MSE ...
    return mse

cfg = PSOConfig(n_particles=50, max_iter=200, omega=0.4, c1=2.0, c2=2.0,
                bounds=[(0.3, 0.7), (0.3, 0.7)])
best_pos, best_val = PSO(my_objective, cfg).run()
```

### Mở rộng từ vựng

```python
# 7 từ (specificity=1)
params = HAParams(theta=0.57, alpha=0.49)
model = LTSModel(params, specificity=1)    # V-, -, L-, W, L+, +, V+

# 17 từ (specificity=2)
model = LTSModel(params, specificity=2)

# Từ vựng tường minh
model = LTSModel(params, words=["V-", "-", "W", "+", "V+"])
```

### Tải dataset mới

```python
from lts import DataLoader

# Từ CSV
dataset = DataLoader.from_csv("path/to/data.csv", column="Close", lb=6000, ub=12000)

# Từ list
dataset = DataLoader.from_list([100, 105, 98, ...], lb=90, ub=120)
```

### Dataset bundled sẵn có

```python
from lts import DataLoader
datasets = ["alabama", "taifex_1998", "temperature", "rice_vietnam",
            "rice_production", "spot_gold", "car_accident", "gas_vietnam"]
for name in datasets:
    ds = DataLoader.bundled(name)
    print(ds)
```

---

## Nền tảng lý thuyết

**Hedge Algebra** `AX = (X, G, C, H, ≤)`:
- Generators: `c− = "-"` (Small), `c+ = "+"` (Large)
- Hedges: `L` (Rather — negative), `V` (Very — positive)
- Neutral: `W` (Middle)

**Semantic Quantifying Measure (SQM):**
```
v(W)  = θ
v(−)  = θ(1 − α)
v(+)  = θ + α(1 − θ)
v(Lx) = v(x) + sign(Lx) · fm(Lx) · (1 − α)
v(Vx) = v(x) + sign(Vx) · fm(Vx) · (1 − β)
```

Công thức closed-form cho 7 từ: [`lts/core/sqm_formulas.py`](lts/core/sqm_formulas.py).

---

## Trích dẫn

```bibtex
@article{hieu2020enrollment,
  title   = {Enrollment Forecasting Based on Linguistic Time Series},
  author  = {Nguyen Duy Hieu and Nguyen Cat Ho and Vu Nhu Lan},
  journal = {Journal of Computer Science and Cybernetics},
  volume  = {36}, number  = {2}, year = {2020},
  doi     = {10.15625/1813-9663/36/2/14396}
}

@article{hieu2021holts,
  title  = {High-Order Linguistic Time Series Forecasting Based on Hedge Algebras},
  author = {Nguyen Duy Hieu},
  year   = {2021}
}

@article{hieu2022ltspso,
  title  = {Linguistic Time Series Forecasting Based on Hedge Algebras and PSO},
  author = {Nguyen Duy Hieu},
  year   = {2022}
}

@article{hieu2023colts,
  title  = {Co-Optimization of Hedge Algebra Parameters and Word-Set Selection for LTS},
  author = {Nguyen Duy Hieu},
  year   = {2023}
}
```

---

## Liên hệ

- Email: hieu3210@gmail.com
- Issues: [github.com/hieu3210/pyLTS/issues](https://github.com/hieu3210/pyLTS/issues)

**License:** GNU GPL v3.0
