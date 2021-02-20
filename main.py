# Main file of LTS procedure
from Errors import Measure
from LTS import LTS
from HAs import HedgeAlgebras

# Get dataset
f = open('datasets/alabama.txt', 'r')
data = list(map(float, f.readline().split(',')))
lb = float(f.readline())
ub = float(f.readline())

# HA Model parameters
theta = 0.46
alpha = 0.52
ha = HedgeAlgebras(theta, alpha)
words = ha.get_words(4)
# words = ["V-", "-", "L-", "W", "L+", "+", "V+"]

# Time series forecasting model parameters
order = 2
repeat = True

# Create forecasting model
lts = LTS(order, repeat, data, lb, ub, words, theta, alpha)

print(str(len(words)) + " words and their SQM:")
print(words)
print(lts.get_semantic())
print(lts.get_real_semantics())

print("Data labels (" + str(len(data)) + " points):")
print(lts.get_label_of_data())

if repeat:
    print("LLRGs (repeated):")
else:
    print("LLRGs (no-repeated):")
for i in range(len(lts.lhs)):
    print(lts.lhs[i], end='')
    print(" \u2192 ", end='')
    print(lts.rhs[i])

forecasted = lts.results
print("Results (" + str(len(forecasted)) + " values):")
print(forecasted)

# Assessment
for i in range(order):
    data.pop(0)

m = Measure(data, forecasted)
print("MSE = " + str(m.mse()))
print("RMSE = " + str(m.rmse()))
