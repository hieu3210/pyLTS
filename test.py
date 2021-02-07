from HAs.HA import HA

p = HA(0.52, 0.49)

X = ["V-", "-", "L-", "W", "L+", "+", "V+"]

for x in X:
    print(p.sqm(x))
