# This file for test only
from HAs import HedgeAlgebras
from ILTS import ILTS
from Errors import Measure

# Get dataset
f = open('datasets/alabama.txt', 'r')
data = list(map(float, f.readline().split(',')))
lb = float(f.readline())
ub = float(f.readline())

# HA Model parameters
theta = 0.57
alpha = 0.49
ha = HedgeAlgebras(theta, alpha)
length = 3
words = ha.get_words(length)

# Time series forecasting model parameters
order = 1
repeat = False

# Time series forecasting model parameters
order = 1
repeat = False

# Create forecasting model
lts = ILTS(order, repeat, data, lb, ub, words, theta, alpha, length)


print(str(len(words)) + " words and their SQM:")
print(words)
print(lts.get_semantic())
print(lts.get_real_semantics())
print(lts.get_interval())
print(lts.get_real_intervals())


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