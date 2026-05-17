# Main file for Co-Optimized LTS
from Errors import Measure
from LTS_2023 import CoOptimizedLTS

# Get dataset
f = open('Datasets/alabama.txt', 'r')
data = list(map(float, f.readline().split(',')))
lb = float(f.readline())
ub = float(f.readline())

# Time series forecasting model parameters
order = 1
repeat = True
num_particles_outer = 10
max_iter_outer = 20
num_particles_inner = 5
max_iter_inner = 10

# Create forecasting model
lts = CoOptimizedLTS(order, repeat, data, lb, ub, num_particles_outer, max_iter_outer, num_particles_inner, max_iter_inner)

print("Optimized theta: ", lts.best_theta)
print("Optimized alpha: ", lts.best_alpha)
print("Optimized length: ", lts.best_length)

print(str(len(lts.words)) + " words and their SQM:")
print(lts.words)
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

# Assessment
for i in range(order):
    data.pop(0)

m = Measure(data, forecasted)
print("MAE = " + str(m.mae()))
print("MSE = " + str(m.mse()))
print("RMSE = " + str(m.rmse()))
print("MAPE = " + str(m.mape(2)) + "%")