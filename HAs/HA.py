# HA MODEL WITH 2 GENERATORS, 2 HEDGES DEFINITION
# c_plus = +
# c_minus = -
# h_minus = L (Little)
# h_plus = V (Very)
# alpha = muy(L)
# theta = fm(c_minus)

class HA:
    # Attributes
    alpha = 0
    theta = 0

    # Methods
    def __init__(self, alpha, theta):
        self.alpha = alpha
        self.theta = theta

    # # Sign between two hedges ...h(2)h(1)...
    def signBetween(self, h2, h1):
        if self.h2 == "L":
            return -1
        else:
            return 1

    # Sign of any word h(n)...h(1)c
    def sign(self, x):
        if len(x) == 1:
            if x == "-":
                return -1
            else:
                return 1
        elif len(x) == 2:
            if x[0] == "L":
                return (-1)*self.sign(x[0])
            else:
                return self.sign(x[0])
        else:
            if self.signBetween(x[0], x[1]) == -1:
                return (-1)*self.sign(x[1:])
            else:
                return self.sign(x[1:])
