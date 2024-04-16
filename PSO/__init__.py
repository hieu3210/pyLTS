import random

class Particle:
    def __init__(self, dimensions, min_position, max_position):
        self.position = [random.uniform(min_position, max_position) for _ in range(dimensions)]
        self.velocity = [random.uniform(-1, 1) for _ in range(dimensions)]
        self.best_position = self.position.copy()
        self.best_fitness = float('inf')

    def update_velocity(self, global_best_position, inertia_weight, cognitive_weight, social_weight):
        for i in range(len(self.velocity)):
            cognitive_component = cognitive_weight * random.random() * (self.best_position[i] - self.position[i])
            social_component = social_weight * random.random() * (global_best_position[i] - self.position[i])
            self.velocity[i] = inertia_weight * self.velocity[i] + cognitive_component + social_component

    def update_position(self):
        for i in range(len(self.position)):
            self.position[i] += self.velocity[i]

    def evaluate_fitness(self, fitness_function):
        fitness = fitness_function(self.position)
        if fitness < self.best_fitness:
            self.best_fitness = fitness
            self.best_position = self.position.copy()

class PSO:
    def __init__(self, dimensions, min_position, max_position, num_particles, max_iterations, inertia_weight, cognitive_weight, social_weight, fitness_function):
        self.dimensions = dimensions
        self.min_position = min_position
        self.max_position = max_position
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.fitness_function = fitness_function
        self.particles = []

    def initialize_particles(self):
        for _ in range(self.num_particles):
            particle = Particle(self.dimensions, self.min_position, self.max_position)
            self.particles.append(particle)

    def optimize(self):
        global_best_fitness = float('inf')
        global_best_position = None

        for _ in range(self.max_iterations):
            for particle in self.particles:
                if global_best_position != None:
                  particle.update_velocity(global_best_position, self.inertia_weight, self.cognitive_weight, self.social_weight)
                particle.update_position()
                particle.evaluate_fitness(self.fitness_function)

                if particle.best_fitness < global_best_fitness or global_best_position != None:
                    global_best_fitness = particle.best_fitness
                    global_best_position = particle.best_position.copy()

        return global_best_position, global_best_fitness

# Example usage
def fitness_function(position):
    x = position[0]
    y = position[1]
    # Hàm số bậc hai: f(x, y) = x^2 + y^2
    return x**2 + y**2

pso = PSO (
    dimensions=2,
    min_position=-10,
    max_position=10,
    num_particles=10,
    max_iterations=100,
    inertia_weight=0.5,
    cognitive_weight=0.5,
    social_weight=0.5,
    fitness_function=fitness_function
  )
pso.initialize_particles()
best_position, best_fitness = pso.optimize()

if best_position is not None:
    print("Best position:", best_position)
    print("Best fitness:", best_fitness)
else:
    print("No best position found.")