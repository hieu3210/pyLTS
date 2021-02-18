# Linguistic Time Series forecasting model
from HAs import HedgeAlgebras
import numpy as np


class LTS:
    # Constructor
    def __init__(self, order, repeat, data, lb, ub, words, theta, alpha):
        self.order = order  # The nth-order of LTS
        self.repeat = repeat  # Weighted or no-weighted with repeat LLRs
        self.data = data  # Dataset list []
        self.lb = lb  # Lower bound of data
        self.ub = ub  # Upper bound of data
        self.words = words  # Words of HA model using in LTS model
        self.ha = HedgeAlgebras(theta, alpha)  # Instance of HA model to calculate the SQMs of words
        self.lhs = []  # Left hand side list of rules
        self.rhs = []  # Right hand side list of rules

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
        real_sem = self.get_real_semantics()
        for d in self.data:
            min_distance = float("inf")
            min_distance_pos = 0
            for x in real_sem:
                if abs(d - x) < min_distance:
                    min_distance = abs(d - x)
                    min_distance_pos = real_sem.index(x)
            sem.append(real_sem[min_distance_pos])
        return sem

    # Get label of historical data
    def get_label_of_data(self):
        label = []
        for d in self.get_semantic_of_data():
            label.append(self.words[self.get_real_semantics().index(d)])
        return label

    # Get linguistic logical relationship groups (repeat = True of False)
    def get_rules(self):
        labels = self.get_label_of_data()
        for i in range(self.order, len(self.data)):
            lhs = []
            for j in range(i - self.order, i):
                lhs.append(labels[j])
            if lhs not in self.lhs:
                self.lhs.append(lhs)
                self.rhs.append([labels[i]])
            else:
                pos = self.lhs.index(lhs)
                rhs = []
                if type(self.rhs[pos]) != list:  # Check if current RHS is list or string
                    rhs.append(self.rhs[pos])
                else:
                    for r in self.rhs[pos]:
                        rhs.append(r)
                if self.repeat:
                    rhs.append(labels[i])
                else:
                    if (type(self.rhs[pos]) != str) and (labels[i] not in self.rhs[pos]):
                        rhs.append(labels[i])
                self.rhs[pos] = rhs

    # Get forecasted results (repeat = True of False)
    def get_results(self):
        self.get_rules()
        labels = self.get_label_of_data()
        results = []
        for i in range(self.order, len(self.data)):
            lhs = []
            for j in range(i - self.order, i):
                lhs.append(labels[j])
            for rule in self.lhs:
                if lhs == rule:
                    pos = self.lhs.index(rule)     # Position of rule will be used
                    if type(self.rhs[pos]) != list:
                        result = self.get_real_semantics()[self.words.index(self.rhs[pos])]
                    else:
                        total = 0
                        count = 0
                        for r in self.rhs[pos]:
                            total += self.get_real_semantics()[self.words.index(r)]
                            count += 1
                        result = float(total / count)
            results.append(result)
        return results
