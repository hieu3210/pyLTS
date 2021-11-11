# Main file of LTS procedure
from Errors import Measure
from LTS import LTS
from HAs import HedgeAlgebras

# Get dataset
f = open('datasets/rice_vietnam.txt', 'r')
data = list(map(float, f.readline().split(',')))
lb = float(f.readline())
ub = float(f.readline())

# HA Model parameters
theta = 0.46
alpha = 0.52
ha = HedgeAlgebras(theta, alpha)
words = ha.get_words(3)
# words = ["V-", "-", "L-", "W", "L+", "+", "V+"]

# Time series forecasting model parameters
order = 1
repeat = False

# Create forecasting model
lts = LTS(order, repeat, data, lb, ub, words, theta, alpha)

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

# Assessment
for i in range(order):
    data.pop(0)

m = Measure(data, forecasted)
print("MAE = " + str(m.mae()))
print("MSE = " + str(m.mse()))
print("RMSE = " + str(m.rmse()))
print("MAPE = " + str(m.mape(2)) + "%")
