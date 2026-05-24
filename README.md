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
python main.py paper

# Chỉ Table 4 — Enrollment forecasting
python main.py paper --table 4

# Chỉ Table 8 — Variation-series forecasting
python main.py paper --table 8

# Chỉ Table 9 — So sánh với Chen và Song & Chissom
python main.py paper --table 9
```

### Chạy trên dataset tuỳ chọn

```bash
# Dataset bundled
python main.py run --dataset alabama
python main.py run --dataset taifex_1998

# Từ file CSV (cột Close, universe [6000, 12000])
python main.py run --csv datasets/TAIEX.csv --column Close --lb 6000 --ub 12000

# Tham số tùy chỉnh
python main.py run --dataset alabama --theta 0.55 --alpha 0.52 --order 2

# Mô hình variation series
python main.py run --dataset alabama --variations --lb-var -1000 --ub-var 1400

# Hiển thị danh sách dự báo
python main.py run --dataset alabama --show
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
python -m pytest tests/ -v
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

Tái hiện từ bài báo 2020 (Alabama Enrollment, 1971–1992):

| Mô hình | MSE | MAPE |
|---|---|---|
| **LTS (đề xuất)** | **262,211** | **2.57%** |
| Chen [1996] | 407,521 | 3.11% |
| Song & Chissom [1993] | 806,087 | 3.76% |

> Lưu ý: Paper báo cáo MSE theo đơn vị ÷1000 (262.326 = 262,326 actual). Semantic points khớp chính xác với Table 1 của bài báo: `{V−: 14,038 · − : 15,035 · L−: 16,032 · W: 16,990 · L+: 17,713 · +: 18,465 · V+: 19,217}`.

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

### Tối ưu tham số (PSO/GA)

`HAParams` là frozen dataclass → hashable, dùng trực tiếp làm objective function:

```python
from lts import HAParams, LTSModel, ForecastMetrics

def objective(theta: float, alpha: float, data, lb, ub) -> float:
    params = HAParams(theta=theta, alpha=alpha)
    m = LTSModel(params, specificity=1, order=1, use_repeat=False)
    m.fit(data, lb, ub)
    mets = ForecastMetrics.compute(data[1:], m.predict())
    return mets.mse  # minimize MSE

# Kết nối với thư viện PSO/GA tuỳ chọn (pyswarms, DEAP, scipy, ...)
```

### Mở rộng từ vựng

```python
# 7 từ (specificity=1)
params = HAParams(theta=0.57, alpha=0.49)
model = LTSModel(params, specificity=1)    # V-, -, L-, W, L+, +, V+

# 15 từ (specificity=2)
model = LTSModel(params, specificity=2)    # thêm VL-, LL-, LV-, VV-, ...

# Từ vựng tường minh
model = LTSModel(params, words=["V-", "-", "W", "+", "V+"])
```

### Tải dataset mới

```python
from lts import DataLoader

# Từ CSV
dataset = DataLoader.from_csv("datasets/TAIEX.csv", column="Close", lb=6000, ub=12000)

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

Công thức closed-form cho 7 từ được implement tại [`lts/core/sqm_formulas.py`](lts/core/sqm_formulas.py).

---

## Trích dẫn

Nếu dùng pyLTS trong nghiên cứu, vui lòng trích dẫn:

```bibtex
@article{hieu2020enrollment,
  title   = {Enrollment Forecasting Based on Linguistic Time Series},
  author  = {Nguyen Duy Hieu and Nguyen Cat Ho and Vu Nhu Lan},
  journal = {Journal of Computer Science and Cybernetics},
  volume  = {36},
  number  = {2},
  year    = {2020},
  doi     = {10.15625/1813-9663/36/2/14396}
}
```

---

## Liên hệ

- Email: hieu3210@gmail.com
- Issues: [github.com/hieu3210/pyLTS/issues](https://github.com/hieu3210/pyLTS/issues)

**License:** GNU GPL v3.0
