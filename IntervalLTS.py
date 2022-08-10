# This file for test only
from HAs import HedgeAlgebras
from ILTS import ILTS

# HA Model parameters
theta = 0.5
alpha = 0.5
ha = HedgeAlgebras(theta, alpha)
length = 2
words = ha.get_words(length)

# Get dataset
f = open('datasets/alabama.txt', 'r')
data = list(map(float, f.readline().split(',')))
lb = float(f.readline())
ub = float(f.readline())

# Time series forecasting model parameters
order = 1
repeat = False

# Create forecasting model
lts = ILTS(order, repeat, data, lb, ub, words, theta, alpha, length)
# Time series forecasting model parameters
order = 1
repeat = False


print(str(len(words)) + " words and their SQM:")
print(words)
print(lts.get_semantic())
print(lts.get_real_semantics())
print(lts.get_interval())
print(lts.get_real_intervals())
