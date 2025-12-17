try:
    from pyswarm import pso
except Exception as e:
    pso = None

from Errors import Measure
from LTS import LTS
from HAs import HedgeAlgebras
import os


# Read dataset file (path relative to this file)
def read_data(file_path: str):
    path = file_path
    if not os.path.isabs(path):
        base = os.path.dirname(__file__)
        path = os.path.join(base, file_path)
    with open(path, 'r') as f:
        data = list(map(float, f.readline().strip().split(',')))
        lb = float(f.readline().strip())
        ub = float(f.readline().strip())
    return data, lb, ub

# Hàm mục tiêu để tối ưu hóa
def objective_function(params, data, lb, ub, order, repeat):
    # Do not mutate the input `data` list — make a local copy for evaluation
    data_copy = list(data)
    theta, alpha = float(params[0]), float(params[1])
    ha = HedgeAlgebras(theta, alpha)
    words = ha.get_words(3)
    lts = LTS(order, repeat, data_copy, lb, ub, words, theta, alpha)
    forecasted = lts.results

    # Ensure there is enough data for evaluation
    if len(data_copy) <= order:
        return float('inf')

    # Align series for error calculation without mutating original list
    eval_data = list(data_copy)[order:]
    m = Measure(eval_data, forecasted)
    mse = m.mse()
    return mse

# Đọc dữ liệu
data, lb, ub = read_data('Datasets/alabama.txt')

# Tham số của mô hình dự báo
order = 1
repeat = True

# Giới hạn cho các tham số theta và alpha

# Parameter bounds for theta and alpha
param_lb = [0.0, 0.0]
param_ub = [1.0, 1.0]

if pso is None:
    raise RuntimeError("pyswarm is not available in the environment. Please install it (pip install pyswarm) to run PSO optimization.")

# Run PSO optimization
best_params, best_mse = pso(objective_function, param_lb, param_ub, args=(data, lb, ub, order, repeat), swarmsize=50, maxiter=100)

# Kết quả tối ưu
best_theta, best_alpha = best_params
print(f"Best theta: {best_theta}, Best alpha: {best_alpha}")
print(f"Best MSE: {best_mse}")

# Sử dụng giá trị tối ưu để tạo mô hình dự báo
ha = HedgeAlgebras(best_theta, best_alpha)
words = ha.get_words(3)
lts = LTS(order, repeat, data, lb, ub, words, best_theta, best_alpha)

print(str(len(words)) + " words and their SQM:")
print(words)
print(lts.get_semantic())
print(lts.get_real_semantics())

print("Data labels (" + str(len(data)) + " points):")
print(lts.get_label_of_data())

if repeat:
    print(str(len(lts.lhs)) + " LLRGs (repeated):")
else:
    print(str(len(lts.lhs)) + " LLRGs (no-repeated):")
for i in range(len(lts.lhs)):
    print(lts.lhs[i], end='')
    print("  \u2192  ", end='')
    print(lts.rhs[i])

forecasted = lts.results
print("Results (" + str(len(forecasted)) + " values):")
print(forecasted)

# Đánh giá kết quả dự báo
if len(data) > 0:
    for i in range(order):
        data.pop(0)

    m = Measure(data, forecasted)
    print("MAE = " + str(m.mae()))
    print("MSE = " + str(m.mse()))
    print("RMSE = " + str(m.rmse()))
    print("MAPE = " + str(m.mape(2)) + "%")
else:
    print("Không có dữ liệu để đánh giá.")