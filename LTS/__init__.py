# Linguistic Time Series forecasting model
from HAs import HedgeAlgebras


class LTS:
    # Constructor
    def __init__(self, data, lb, ub, words, theta, alpha):
        self.__data = data  # Dataset list []
        self.__lb = lb  # Lower bound of data
        self.__ub = ub  # Upper bound of data
        self.__words = words    # Words of HA model using in LTS model
        self.__ha = HedgeAlgebras(theta, alpha)     # Instance of HA model to calculate the SQMs of words

    # Get semantics of words in [0,1]
    def get_semantic(self):
        sem = []
        for x in self.__words:
            sem.append(self.__ha.sqm(x))
        return sem

    # Get real semantics of words in [lb, ub]
    def get_real_semantics(self):
        real_sem = []
        for x in self.get_semantic():
            real_sem.append(self.__lb + (self.__ub - self.__lb) * x)
        return real_sem

    # Get real semantics of historical data by words in HA
    def get_semantic_of_data(self):
        sem = []
        for d in self.__data:
            min_distance = float("inf")
            min_distance_pos = 0
            for x in self.get_real_semantics():
                if abs(d - x) > min_distance:
                    min_distance = abs(d - x)
                    min_distance_pos = self.__words.index(x)
            sem.append(self.get_real_semantics()[min_distance_pos])
        return sem

    # Get label of historical data
    def get_label_of_data(self):
        label = []
        for d in self.get_semantic_of_data():
            label.append(self.__words(self.get_real_semantics().index(d)))
        return label

    # Get LLRs (not repeated)

    # Get LLRGs (not repeated)

    # Get LLRs (repeated)

    # Get LLRGs (repeated)

    # Get forecasted results (repeat = True of False)

