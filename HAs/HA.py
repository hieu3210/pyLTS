# HA MODEL WITH 2 GENERATORS, 2 HEDGES DEFINITION
# c_plus = +
# c_minus = -
# h_minus = L (Little)
# h_plus = V (Very)
# alpha = muy(L)
# theta = fm(c_minus)

class HA:
    # Attributes
    # alpha = 0
    # theta = 0
    # beta = 0

    # Methods
    def __init__(self, theta, alpha):
        self.alpha = alpha
        self.theta = theta
        self.beta = 1 - self.alpha

    # Fuzzy measure of word hx
    def fm(self, x):
        if len(x) == 1:
            if x == "-":
                return self.alpha
            else:
                return self.beta
        else:
            if x[0] == "L":
                return self.alpha * self.fm(x[1:])
            else:
                return self.beta * self.fm(x[1:])

    # Sign between two hedges ...h(2)h(1)...
    @staticmethod
    def signBetween(h2, h1):
        if h2 == "L":
            return -1
        else:
            return 1

    # Sign of word h(n)...h(1)c
    def sign(self, x):
        if len(x) == 1:
            if x == "-":
                return -1
            else:
                return 1
        elif len(x) == 2:
            if x[0] == "L":
                return (-1) * self.sign(x[1])
            else:
                return self.sign(x[1])
        else:
            if self.signBetween(x[0], x[1]) == -1:
                return (-1) * self.sign(x[1:])
            else:
                return self.sign(x[1:])

    # Omega para
    def omega(self, x):
        return (1 + self.sign(x)*self.sign("V" + x)*(self.beta - self.alpha))/2

    # SQM function
    def sqm(self, x):
        if len(x) == 1:
            if x == "W":
                return self.theta
            if x == "-":
                return self.theta - self.alpha * self.fm(x)
            if x == "+":
                return self.theta + self.alpha * self.fm(x)
        else:
            return self.sqm(x[1:]) + self.sign(x) * (self.fm(x) - self.omega(x) * self.fm(x))   # Only with 2 hedges
