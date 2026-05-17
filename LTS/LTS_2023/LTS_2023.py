# Co-Optimized Linguistic Time Series forecasting model
from HAs import HedgeAlgebras
import random

class LTS:
    # Copy LTS class here as before
    def __init__(self, order, repeat, data, lb, ub, words, theta, alpha):
        self.order = order
        self.repeat = repeat
        self.data = data
        self.lb = lb
        self.ub = ub
        self.words = words
        self.ha = HedgeAlgebras(theta, alpha)
        self.lhs = []
        self.rhs = []
        self.results = self.get_results()

    def get_semantic(self):
        sem = []
        for x in self.words:
            sem.append(self.ha.sqm(x))
        return sem

    def get_real_semantics(self):
        real_sem = []
        for x in self.get_semantic():
            real_sem.append(self.lb + (self.ub - self.lb) * x)
        return real_sem

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

    def get_label_of_data(self):
        label = []
        for d in self.get_semantic_of_data():
            label.append(self.words[self.get_real_semantics().index(d)])
        return label

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
                if self.repeat:
                    self.rhs[pos].append(labels[i])
                else:
                    if labels[i] not in self.rhs[pos]:
                        self.rhs[pos].append(labels[i])

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
                    pos = self.lhs.index(rule)
                    total = 0
                    count = 0
                    for r in self.rhs[pos]:
                        total += self.get_real_semantics()[self.words.index(r)]
                        count += 1
                    result = float(total / count)
            results.append(result)
        return results

class CoOptimizedLTS:
    def __init__(self, order, repeat, data, lb, ub, num_particles_outer=10, max_iter_outer=20, num_particles_inner=5, max_iter_inner=10):
        self.order = order
        self.repeat = repeat
        self.data = data
        self.lb = lb
        self.ub = ub
        self.best_theta = 0.5
        self.best_alpha = 0.5
        self.best_length = 3
        self.best_fitness = float('inf')

        # Outer PSO for theta, alpha
        bounds_outer = [(0, 1), (0, 1)]
        particles_outer = [self.init_particle(bounds_outer) for _ in range(num_particles_outer)]

        for _ in range(max_iter_outer):
            for p in particles_outer:
                # Inner optimization for length
                best_length = self.optimize_length(p['position'][0], p['position'][1], num_particles_inner, max_iter_inner)
                ha = HedgeAlgebras(p['position'][0], p['position'][1])
                words = ha.get_words(best_length)
                lts = LTS(order, repeat, data, lb, ub, words, p['position'][0], p['position'][1])
                forecasted = lts.results
                actual = data[order:]
                fitness = sum((a - f)**2 for a, f in zip(actual, forecasted)) / len(actual)

                if fitness < p['best_fitness']:
                    p['best_fitness'] = fitness
                    p['best_position'] = p['position'][:]
                    p['best_length'] = best_length

                if fitness < self.best_fitness:
                    self.best_fitness = fitness
                    self.best_theta, self.best_alpha = p['position']
                    self.best_length = best_length

            # Update velocities and positions for outer
            for p in particles_outer:
                for i in range(2):
                    r1 = random.random()
                    r2 = random.random()
                    p['velocity'][i] = 0.5 * p['velocity'][i] + 1.5 * r1 * (p['best_position'][i] - p['position'][i]) + 1.5 * r2 * (self.best_theta if i==0 else self.best_alpha - p['position'][i])
                    p['position'][i] += p['velocity'][i]
                    p['position'][i] = max(bounds_outer[i][0], min(bounds_outer[i][1], p['position'][i]))

        # Final model
        ha = HedgeAlgebras(self.best_theta, self.best_alpha)
        self.words = ha.get_words(self.best_length)
        self.lts = LTS(order, repeat, data, lb, ub, self.words, self.best_theta, self.best_alpha)
        self.results = self.lts.results
        self.lhs = self.lts.lhs
        self.rhs = self.lts.rhs

    def init_particle(self, bounds):
        return {
            'position': [random.uniform(b[0], b[1]) for b in bounds],
            'velocity': [random.uniform(-1, 1) for _ in bounds],
            'best_position': None,
            'best_fitness': float('inf'),
            'best_length': 3
        }

    def optimize_length(self, theta, alpha, num_particles, max_iter):
        bounds_length = [(1, 5)]  # max_length from 1 to 5
        particles = [self.init_particle(bounds_length) for _ in range(num_particles)]
        global_best_length = 3
        global_best_fitness = float('inf')

        for _ in range(max_iter):
            for p in particles:
                length = int(round(p['position'][0]))
                length = max(1, min(5, length))
                ha = HedgeAlgebras(theta, alpha)
                words = ha.get_words(length)
                lts = LTS(self.order, self.repeat, self.data, self.lb, self.ub, words, theta, alpha)
                forecasted = lts.results
                actual = self.data[self.order:]
                fitness = sum((a - f)**2 for a, f in zip(actual, forecasted)) / len(actual)

                if fitness < p['best_fitness']:
                    p['best_fitness'] = fitness
                    p['best_position'] = p['position'][:]

                if fitness < global_best_fitness:
                    global_best_fitness = fitness
                    global_best_length = length

            # Update for inner
            for p in particles:
                r1 = random.random()
                r2 = random.random()
                p['velocity'][0] = 0.5 * p['velocity'][0] + 1.5 * r1 * (p['best_position'][0] - p['position'][0]) + 1.5 * r2 * (global_best_length - p['position'][0])
                p['position'][0] += p['velocity'][0]
                p['position'][0] = max(bounds_length[0][0], min(bounds_length[0][1], p['position'][0]))

        return global_best_length

    def get_semantic(self):
        return self.lts.get_semantic()

    def get_real_semantics(self):
        return self.lts.get_real_semantics()

    def get_label_of_data(self):
        return self.lts.get_label_of_data()