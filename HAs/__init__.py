# HA MODEL WITH 2 GENERATORS, 2 HEDGES DEFINITION
# c_plus = +
# c_minus = -

class HedgeAlgebras:
    # Constructor
    def __init__(self, theta, alpha):
        self.theta = theta    # fm(c_minus)
        self.alpha = alpha    # muy(L)
        self.beta = 1 - self.alpha  # muy(V)

    # Get words at k level
    def get_words(self, k):
        if k == 0:
            return ["0", "W", "1"]
        if k == 1:
            return ["0", "-", "W", "+", "1"]
        # If k >= 2: use recursion to impact (k - 1) length word by hedges
        list_words = self.get_words(k - 1)
        temp = []
        # Get words with length = k - 1
        if k == 2:
            temp = ["-", "+"]
        else:
            for x in list_words:
                if len(x) == k - 1:
                    temp.append(x)
        # Add new words impacted by hedges to list
        for x in temp:
            list_words.append("L" + x)
            list_words.append("V" + x)
        # Sort the list of words
        self.sort_words(list_words)
        # Return list of words
        return list_words

    # Sort list words
    def sort_words(self, list_words):
        for i in range(0, len(list_words) - 1):
            for j in range(i + 1, len(list_words)):
                if self.sqm(list_words[i]) > self.sqm(list_words[j]):
                    temp = list_words[i]
                    list_words[i] = list_words[j]
                    list_words[j] = temp

    # Fuzzy measure of word hx
    def fm(self, x):
        if (x == "W") or (x == "0") or (x == "1"):
            return 0
        if len(x) == 1:
            if x == "-":
                return self.theta
            else:
                return 1 - self.theta
        else:
            if x[0] == "L":
                return self.alpha * self.fm(x[1:])
            else:
                return self.beta * self.fm(x[1:])

    # Sign between two hedges ...h(2)h(1)...
    @staticmethod
    def sign_between(h2, h1):
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
            if self.sign_between(x[0], x[1]) == -1:
                return (-1) * self.sign(x[1:])
            else:
                return self.sign(x[1:])

    # Omega para
    def omega(self, x):
        return (1 + self.sign(x) * self.sign("V" + x) * (self.beta - self.alpha)) / 2

    # SQM function
    def sqm(self, x):
        if len(x) == 1:
            if x == "0":
                return 0
            if x == "W":
                return self.theta
            if x == "1":
                return 1
            if x == "-":
                return self.theta - self.alpha * self.fm(x)
            if x == "+":
                return self.theta + self.alpha * self.fm(x)
        else:
            return self.sqm(x[1:]) + self.sign(x) * (self.fm(x) - self.omega(x) * self.fm(x))  # With 2 hedges only
