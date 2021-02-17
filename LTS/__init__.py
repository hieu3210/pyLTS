# Linguistic Time Series forecasting model
from HAs import HedgeAlgebras


class LTS:
    # Constructor
    def __init__(self, data, lb, ub, words, theta, alpha):
        self.data = data  # Dataset list []
        self.lb = lb  # Lower bound of data
        self.ub = ub  # Upper bound of data
        self.words = words  # Words of HA model using in LTS model
        self.ha = HedgeAlgebras(theta, alpha)  # Instance of HA model to calculate the SQMs of words

    # Get semantics of words in [0,1]
    def get_semantic(self):
        sem = []
        for x in self.words:
            sem.append(self.ha.sqm(x))
        return sem

    # Get real semantics of words in [lb, ub]
    def get_real_semantics(self):
        real_sem = []
        for x in self.get_semantic():
            real_sem.append(self.lb + (self.ub - self.lb) * x)
        return real_sem

    # Get real semantics of historical data by words in HA
    def get_semantic_of_data(self):
        sem = []
        for d in self.data:
            min_distance = float("inf")
            min_distance_pos = 0
            for x in self.get_real_semantics():
                if abs(d - x) > min_distance:
                    min_distance = abs(d - x)
                    min_distance_pos = self.words.index(x)
            sem.append(self.get_real_semantics()[min_distance_pos])
        return sem

    # Get label of historical data
    def get_label_of_data(self):
        label = []
        for d in self.get_semantic_of_data():
            label.append(self.words(self.get_real_semantics().index(d)))
        return label

    # Get linguistic logical relationship groups (repeat = True of False)
    def get_rules(self, repeat, order):
        if repeat:
            pass
        else:
            pass

    # Get forecasted results (repeat = True of False)
    def get_results(self, repeat, order):
        if repeat:
            pass
        else:
            pass
