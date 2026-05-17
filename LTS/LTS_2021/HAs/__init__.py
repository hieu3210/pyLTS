# HA MODEL WITH 2 GENERATORS, 2 HEDGES DEFINITION
# c_plus = +
# c_minus = -

class HedgeAlgebras:
    # Constructor
    def __init__(self, theta, alpha):
        self.theta = theta    # fm(c_minus)
        self.alpha = alpha    # muy(L)
        self.beta = 1 - alpha  # muy(V)

    # Get words at k level
    def get_words(self, k):
        if k == 0:
            return ["0", "W", "1"]
        if k == 1:
            return ["0", "-", "W", "+", "1"]
        
        # Use recursion to get words of length k - 1
        list_words = self.get_words(k - 1)
        new_words = []

        # Add new words impacted by hedges to list
        for word in list_words:
            if len(word) == k - 1:
                new_words.append("L" + word)
                new_words.append("V" + word)

        # Combine and sort the list of words
        list_words.extend(new_words)
        self.sort_words(list_words)
        
        return list_words

    # Sort list words
    def sort_words(self, list_words):
        list_words.sort(key=self.sqm)

    # Fuzzy measure of word hx
    def fm(self, x):
        if x in {"W", "0", "1"}:
            return 0
        if len(x) == 1:
            return self.theta if x == "-" else 1 - self.theta
        return self.alpha * self.fm(x[1:]) if x[0] == "L" else self.beta * self.fm(x[1:])

    # Sign between two hedges ...h(2)h(1)...
    @staticmethod
    def sign_between(h2, h1):
        return -1 if h2 == "L" else 1

    # Sign of word h(n)...h(1)c
    def sign(self, x):
        if len(x) == 1:
            return -1 if x == "-" else 1
        sign_rest = self.sign(x[1:])
        return sign_rest if x[0] == "V" else -sign_rest

    # Omega para
    def omega(self, x):
        return (1 + self.sign(x) * self.sign("V" + x) * (self.beta - self.alpha)) / 2

    # SQM function
    def sqm(self, x):
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
        return self.sqm(x[1:]) + self.sign(x) * (self.fm(x) - self.omega(x) * self.fm(x))
