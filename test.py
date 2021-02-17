# This file for test only
from Errors import Measure
from HAs import HedgeAlgebras

# Test error measurements
# a = [13563, 13867, 14696, 15460, 15311, 15603, 15861, 16807, 16919, 16388, 15433, 15497, 15145, 15163, 15984, 16859, 18150, 18970, 19328, 19337, 18876]
# b = [14537, 14537, 14537, 15534, 15534, 15534, 16019, 16019, 17162, 17162, 16019, 15534, 15534, 15534, 15514, 16019, 17162, 19217, 19217, 19217, 19217]
# m = Measure(a, b)
# c = m.error_percentage(10)
# print(c)

# Test HA Model
theta = 0.5
alpha = 0.5

p = HedgeAlgebras(theta, alpha)

X = p.get_words(5)
print(len(X))
print(X)
for x in X:
    print(p.sqm(x))

