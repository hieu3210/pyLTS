# PSO Optimized Linguistic Time Series forecasting model
from HAs import HedgeAlgebras
import random
import math

class LTS:
    # Constructor
    def __init__(self, order, repeat, data, lb, ub, words, theta, alpha):
        self.order = order  # The nth-order of LTS
        self.repeat = repeat  # True or False, weighted or no-weighted with repeat LLRs
        self.data = data  # Dataset list []
        self.lb = lb  # Lower bound of universe of discourse
        self.ub = ub  # Upper bound of universe of discourse
        self.words = words  # Words of HA model using in LTS model
        self.ha = HedgeAlgebras(theta, alpha)  # Instance of HA model to calculate the SQMs of words
        self.lhs = []  # Left hand side list of rules
        self.rhs = []  # Right hand side list of rules
        self.results = self.get_results()   # Calculate the forecasted results

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

    # Get linguistic logical relationship groups
    def get_rules(self):
        labels = self.get_label_of_data()
        for i in range(self.order, len(self.data)):
            lhs = []
            for j in range(i - self.order, i):
                lhs.append(labels[j])
            if lhs not in self.lhs:     # Check if the rule not existed
                self.lhs.append(lhs)
                self.rhs.append([labels[i]])
            else:                       # or existed
                pos = self.lhs.index(lhs)
                if self.repeat:
                    self.rhs[pos].append(labels[i])
                else:
                    if labels[i] not in self.rhs[pos]:
                        self.rhs[pos].append(labels[i])

    # Get forecasted results
    def get_results(self):
        self.get_rules()
        labels = self.get_label_of_data()
        results = []
        for i in range(self.order, len(self.data)):
            lhs = []
            result = 0
            for j in range(i - self.order, i):
                lhs.append(labels[j])
            for rule in self.lhs:
                if lhs == rule:
                    pos = self.lhs.index(rule)     # Position of rule will be used
                    total = 0
                    count = 0
                    for r in self.rhs[pos]:
                        total += self.get_real_semantics()[self.words.index(r)]
                        count += 1
                    result = float(total / count)
            results.append(result)
        return results

class Particle:
    def __init__(self, bounds):
        self.position = [random.uniform(b[0], b[1]) for b in bounds]  # theta, alpha
        self.velocity = [random.uniform(-1, 1) for _ in bounds]
        self.best_position = self.position[:]
        self.best_fitness = float('inf')

class PSO:
    def __init__(self, num_particles, max_iter, bounds, data, lb, ub, order, repeat, max_length):
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.bounds = bounds
        self.data = data
        self.lb = lb
        self.ub = ub
        self.order = order
        self.repeat = repeat
        self.max_length = max_length
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.particles = [Particle(bounds) for _ in range(num_particles)]

    def fitness(self, position):
        theta, alpha = position
        if alpha < 0 or alpha > 1 or theta < 0 or theta > 1:
            return float('inf')
        ha = HedgeAlgebras(theta, alpha)
        words = ha.get_words(self.max_length)
        lts = LTS(self.order, self.repeat, self.data, self.lb, self.ub, words, theta, alpha)
        forecasted = lts.results
        actual = self.data[self.order:]
        mse = sum((a - f)**2 for a, f in zip(actual, forecasted)) / len(actual)
        return mse

    def optimize(self):
        w = 0.5  # inertia
        c1 = 1.5  # cognitive
        c2 = 1.5  # social

        for _ in range(self.max_iter):
            for particle in self.particles:
                fitness = self.fitness(particle.position)
                if fitness < particle.best_fitness:
                    particle.best_fitness = fitness
                    particle.best_position = particle.position[:]
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = particle.position[:]

            for particle in self.particles:
                for i in range(len(particle.position)):
                    r1 = random.random()
                    r2 = random.random()
                    particle.velocity[i] = w * particle.velocity[i] + c1 * r1 * (particle.best_position[i] - particle.position[i]) + c2 * r2 * (self.global_best_position[i] - particle.position[i])
                    particle.position[i] += particle.velocity[i]
                    # Clamp to bounds
                    particle.position[i] = max(self.bounds[i][0], min(self.bounds[i][1], particle.position[i]))

        return self.global_best_position, self.global_best_fitness

class PSOOptimizedLTS:
    # Constructor
    def __init__(self, order, repeat, data, lb, ub, max_length=3, num_particles=20, max_iter=50):
        self.order = order
        self.repeat = repeat
        self.data = data
        self.lb = lb
        self.ub = ub
        self.max_length = max_length
        bounds = [(0, 1), (0, 1)]  # theta, alpha
        pso = PSO(num_particles, max_iter, bounds, data, lb, ub, order, repeat, max_length)
        self.best_theta, self.best_alpha = pso.optimize()
        self.ha = HedgeAlgebras(self.best_theta, self.best_alpha)
        self.words = self.ha.get_words(max_length)
        self.lts = LTS(order, repeat, data, lb, ub, self.words, self.best_theta, self.best_alpha)
        self.results = self.lts.results
        self.lhs = self.lts.lhs
        self.rhs = self.lts.rhs

    # Delegate methods to lts
    def get_semantic(self):
        return self.lts.get_semantic()

    def get_real_semantics(self):
        return self.lts.get_real_semantics()

    def get_label_of_data(self):
        return self.lts.get_label_of_data()