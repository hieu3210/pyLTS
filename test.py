# This file for test only
from HAs.HA import HedgeAlgebras

theta = 0.55
alpha = 0.52

p = HedgeAlgebras(theta, alpha)

# X = ["V+", "W", "L+", "-", "L-", "+", "V-"]
# print(X)
# p.sort_terms(X)
# print(X)

X = p.get_words(3)
print(len(X))
print(X)
for x in X:
    print(p.sqm(x))
