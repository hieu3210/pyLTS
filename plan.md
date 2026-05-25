# Kế hoạch Nghiên cứu: Mở rộng mô hình LTS

**Người thực hiện:** Nguyen Duy Hieu  
**Cơ sở:** pyLTS — Linguistic Time Series Forecasting based on Hedge Algebras  
**Mục tiêu:** Công bố khoa học (Journal + Conference) về các mở rộng của mô hình LTS gốc, giải quyết các hạn chế hiện tại và nâng cao hiệu suất dự báo.

---

## Nền tảng đã có

| Mô hình | Bài báo | Trạng thái |
|---|---|---|
| LTS | hieund_2020 | ✅ Implemented, tested |
| LTS Variations | hieund_2020 | ✅ Implemented, tested |
| HO-LTS | hieund_2021 | ✅ Implemented, tested |
| LTS-PSO | hieund_2022 | ✅ Implemented, tested |
| CO-LTS | hieund_2023 | ✅ Implemented, tested |

**Tests:** 116/116 passed.

---

## Phân tích khoảng trống (Research Gaps)

### Điểm yếu chung của tất cả mô hình hiện tại

1. **Exact-match lookup**: Khi LHS không có trong LLRG, phải dùng fallback thô — bỏ qua hoàn toàn thông tin từ các rule *tương tự*.
2. **Dự báo điểm duy nhất**: Không có ước lượng độ không chắc chắn (uncertainty quantification).
3. **Tĩnh sau khi fit**: Mô hình không cập nhật khi dữ liệu mới đến (non-adaptive).
4. **Một biến (univariate)**: Không khai thác tương quan với chuỗi thời gian khác.
5. **Một bước (one-step-ahead)**: Chưa có dự báo nhiều bước (multi-step forecasting).

---

## Hướng nghiên cứu đề xuất

---

### Hướng 1: SW-LTS — Similarity-Weighted LTS ⭐ *Ưu tiên cao*

**Mức độ ưu tiên:** Cao — phù hợp hội nghị Scopus ngắn hạn (3–6 tháng)

#### Vấn đề giải quyết
Exact-match trong LTS gốc bỏ qua thông tin từ các LLRG *gần giống* với trạng thái hiện tại. Khi tập dữ liệu nhỏ (thường gặp trong LTS), sparse rules là vấn đề nghiêm trọng.

#### Ý tưởng cốt lõi
Thay exact-match bằng **kernel-weighted aggregation** trên toàn bộ LLRG, sử dụng độ tương tự trong không gian semantic của Hedge Algebras.

#### Công thức dự báo

```
forecast(lhs_t) = Σ_k [ sim(lhs_t, lhs_k) × mean_s(RHS_k) ]
                  ─────────────────────────────────────────────
                         Σ_k sim(lhs_t, lhs_k)

sim(u, v) = exp( -||sp(u) - sp(v)||² / (2σ²) )
```

Trong đó:
- `sp(lhs)` = vector semantic points của các từ trong LHS
- `mean_s(RHS_k)` = mean semantic point của RHS trong LLRG thứ k
- `σ` = bandwidth (tham số học được)

#### Tính chất lý thuyết
- **σ → 0**: degenerates về exact-match (LTS gốc)
- **σ → ∞**: degenerates về mean of all RHS (HO-LTS fallback toàn cục)
- SW-LTS là **generalization** của cả LTS và HO-LTS

#### Mở rộng: SW-PSO-LTS
Tối ưu đồng thời `(θ, α, σ)` bằng PSO:
```
bounds = [(0.3, 0.7), (0.3, 0.7), (0.1, 5.0)]
         [  theta   ] [  alpha  ] [  sigma  ]
```

#### Điểm mới (Novelty)
- Lần đầu áp dụng kernel similarity trên LLRG của Hedge Algebras
- `σ` có ý nghĩa vật lý rõ trong không gian semantic HA
- Unifies LTS, HO-LTS như các trường hợp đặc biệt

#### Kế hoạch implement
- [ ] `lts/models/sw_lts.py` — SWLTSModel với kernel lookup
- [ ] `lts/models/sw_pso_lts.py` — SW-PSO-LTS (tối ưu sigma + params)
- [ ] `tests/test_sw_lts.py`
- [ ] Thực nghiệm: Alabama, car_accident, spot_gold, TAIFEX, temperature

#### Kết quả kỳ vọng (so với LTS gốc)
- Giảm MSE 20–40% nhờ giải quyết sparse-rule problem
- Tốt hơn CO-LTS trên dataset nhỏ (ít overfitting)
- Nhanh hơn CO-LTS (không cần nested PSO)

---

### Hướng 2: IV-LTS — Interval-Valued LTS ⭐⭐ *Ưu tiên cho journal*

**Mức độ ưu tiên:** Cao — phù hợp journal SCI/SCIE (6–12 tháng)

#### Vấn đề giải quyết
Tất cả mô hình LTS hiện tại chỉ cho **dự báo điểm** — không có thông tin về độ không chắc chắn. Trong thực tế, người dùng cần biết khoảng tin cậy của dự báo.

#### Ý tưởng cốt lõi
HA tự nhiên gán cho mỗi từ một **khoảng ngữ nghĩa**, không chỉ một điểm:
```
Khoảng của từ x: [v(Lx), v(x)]  nếu x là từ dương (c+)
                 [v(x), v(Vx)]  nếu x là từ âm (c-)
```
Thay vì dự báo `y_hat`, dự báo khoảng `[y_lo, y_hi]`.

#### Công thức dự báo khoảng
```
[lo_t, hi_t] = mean_interval(RHS_k)
             = [mean(lo(RHS_k)), mean(hi(RHS_k))]
```

#### Điểm mới
- Khai thác đầy đủ cấu trúc interval của HA (hiện tại chỉ dùng midpoint)
- Uncertainty quantification có nền tảng lý thuyết từ HA
- Có thể đo bằng Coverage Rate và Interval Width ngoài MSE/MAPE

#### Kế hoạch implement
- [ ] Mở rộng `HedgeAlgebra.sqm_interval(word)` → `(lo, hi)`
- [ ] `lts/models/iv_lts.py` — IVLTSModel
- [ ] Metrics: coverage rate, interval width, Winkler score
- [ ] `tests/test_iv_lts.py`

---

### Hướng 3: TA-LTS — Trend-Adaptive LTS

**Mức độ ưu tiên:** Trung bình — hội nghị hoặc journal (6–9 tháng)

#### Vấn đề giải quyết
LTS không tách bạch trend và fluctuation. Trên các chuỗi có xu hướng rõ (giá cổ phiếu, dân số), mô hình phân vùng sai ngôn ngữ.

#### Ý tưởng
1. Tách: `data = trend + residual` (linear detrend hoặc STL)
2. Áp dụng LTS trên `residual`
3. Dự báo cuối: `y_hat = trend_hat + lts_residual_hat`

#### Điểm mới
Kết hợp classical decomposition với HA-based linguistic modeling — giải quyết non-stationarity trong LTS.

#### Kế hoạch implement
- [ ] `lts/data/transforms.py` — thêm `detrend()`, `STL_decompose()`
- [ ] `lts/models/ta_lts.py` — TALTSModel
- [ ] Thực nghiệm trên chuỗi có trend rõ: spot_gold, TAIEX

---

### Hướng 4: MS-LTS — Multi-Step LTS

**Mức độ ưu tiên:** Trung bình (6–9 tháng)

#### Vấn đề giải quyết
Tất cả mô hình LTS hiện tại chỉ dự báo **1 bước** (one-step-ahead). Ứng dụng thực tế cần dự báo h bước.

#### Hai chiến lược
1. **Recursive**: dùng dự báo t+1 làm input cho t+2, ... (error accumulation)
2. **Direct**: xây h mô hình riêng biệt cho mỗi horizon

#### Điểm mới
Phân tích error propagation trong không gian ngôn ngữ của HA qua nhiều bước dự báo.

---

## Ma trận ưu tiên

| Hướng | Novelty | Khó cài đặt | Thời gian | Venue phù hợp |
|---|---|---|---|---|
| SW-LTS | Cao | Trung bình | 3–6 tháng | Conference Scopus |
| SW-PSO-LTS | Rất cao | Trung bình | 4–6 tháng | Conference Scopus |
| IV-LTS | Rất cao | Cao | 6–12 tháng | Journal SCI |
| TA-LTS | Trung bình | Thấp | 3–5 tháng | Conference Scopus |
| MS-LTS | Trung bình | Trung bình | 5–8 tháng | Conference/Journal |

---

## Roadmap

```
Tháng 1–2:  SW-LTS implement + thực nghiệm sơ bộ
Tháng 2–3:  SW-PSO-LTS + viết paper draft
Tháng 3–4:  Submit hội nghị Scopus
            Song song: bắt đầu IV-LTS (lý thuyết)
Tháng 5–8:  IV-LTS implement + thực nghiệm + viết journal
Tháng 9+:   Submit journal SCI/SCIE
```

---

## Datasets cho thực nghiệm

| Dataset | Bundled | Đặc điểm | Dùng cho |
|---|---|---|---|
| Alabama enrollment | ✅ | Nhỏ (n=22), chuẩn LTS | Tất cả |
| Car accidents | ✅ | Biến động cao | SW-LTS, CO-LTS |
| Spot gold | ✅ | Trend rõ | TA-LTS, SW-LTS |
| TAIFEX 1998 | ✅ | Tài chính | SW-LTS, IV-LTS |
| Temperature | ✅ | Mùa vụ | TA-LTS |
| Rice Vietnam | ✅ | Nông nghiệp | SW-LTS |
| UCI datasets | ❌ | Cần download | Journal |

---

## Baseline so sánh (đủ cho paper)

- **LTS** (Nguyen Duy Hieu et al., 2020)
- **HO-LTS** (Nguyen Duy Hieu, 2021)
- **LTS-PSO** (Nguyen Duy Hieu, 2022)
- **CO-LTS** (Nguyen Duy Hieu, 2023)
- **Chen** (Chen, 1996)
- **Song & Chissom** (Song & Chissom, 1993)
- **FLR-based methods** (nếu cần, từ tài liệu)

---

## Metrics đánh giá

| Metric | Công thức | Dùng cho |
|---|---|---|
| MSE | `mean((y - ŷ)²)` | Tất cả |
| RMSE | `√MSE` | Tất cả |
| MAPE | `mean(|y-ŷ|/y) × 100%` | Tất cả |
| MAE | `mean(|y - ŷ|)` | Tất cả |
| Coverage Rate | `P(y ∈ [lo, hi])` | IV-LTS |
| Interval Width | `mean(hi - lo)` | IV-LTS |

---

## Ghi chú kỹ thuật

### Cấu trúc code khi thêm mô hình mới

```python
# 1. Kế thừa BaseForecaster
from lts.models.base import BaseForecaster, ForecastResult

class SWLTSModel(BaseForecaster):
    def fit(self, data, lb, ub): ...
    def predict(self): ...
    def get_result(self): ...

# 2. Thêm vào lts/__init__.py
from lts.models.sw_lts import SWLTSModel

# 3. Thêm experiment function
# lts/experiments/paper_experiments.py → run_swlts_table()

# 4. Thêm CLI option
# lts/_cli.py → --model sw-lts

# 5. Viết tests
# tests/test_sw_lts.py
```

### PSO configuration chuẩn (từ các bài báo)

```python
PSOConfig(
    n_particles=300,  # N=300 (LTS-PSO 2022)
    max_iter=1000,    # G_max=1000
    omega=0.4,
    c1=2.0,
    c2=2.0,
    bounds=[...],
    seed=run_id,
)
# Chạy 3 lần, lấy MSE nhỏ nhất
```

---

## Tài liệu tham khảo chính

- `publications/hieund_2020.pdf` — LTS gốc
- `publications/hieund_2021.pdf` — HO-LTS
- `publications/hieund_2022.pdf` — LTS-PSO
- `publications/hieund_2023.pdf` — CO-LTS
