# Main file for High-Order LTS with growth of word-set
from Errors import Measure
from LTS_2021 import HighOrderLTS

# Get dataset
f = open('Datasets/alabama.txt', 'r')
data = list(map(float, f.readline().split(',')))
lb = float(f.readline())
ub = float(f.readline())

# HA Model parameters
theta = 0.57
alpha = 0.49

# Time series forecasting model parameters
order = 4  # High-order, e.g., 2
repeat = True
max_length = 4
grow_wordset = False  # Set to True to grow word-set

# Create forecasting model
lts = HighOrderLTS(order, repeat, data, lb, ub, theta, alpha, max_length, grow_wordset)

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