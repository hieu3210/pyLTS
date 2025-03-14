from pyswarm import pso
from Errors import Measure
from LTS import LTS
from HAs import HedgeAlgebras

# Đọc dữ liệu từ file
def read_data(file_path):
    with open(file_path, 'r') as f:
        data = list(map(float, f.readline().split(',')))
        lb = float(f.readline())
        ub = float(f.readline())
    return data, lb, ub

# Hàm mục tiêu để tối ưu hóa
def objective_function(params, data, lb, ub, order, repeat):
    theta, alpha = params
    ha = HedgeAlgebras(theta, alpha)
    words = ha.get_words(3)
    lts = LTS(order, repeat, data, lb, ub, words, theta, alpha)
    forecasted = lts.results

    # Đánh giá kết quả dự báo
    if len(data) <= order:
        return float('inf')  # Trả về giá trị lớn nếu dữ liệu không đủ

    for i in range(order):
        data.pop(0)
    m = Measure(data, forecasted)
    mse = m.mse()
    return mse

# Đọc dữ liệu
data, lb, ub = read_data('Datasets/alabama.txt')

# Tham số của mô hình dự báo
order = 1
repeat = True

# Giới hạn cho các tham số theta và alpha
lb_params = [0, 0]
ub_params = [1, 1]

# Tối ưu hóa bằng PSO
best_params, best_mse = pso(objective_function, lb_params, ub_params, args=(data, lb, ub, order, repeat), swarmsize=50, maxiter=100)

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