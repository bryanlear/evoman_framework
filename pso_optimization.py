import numpy as np
import os
import sys
from evoman.environment import Environment
from demo_controller import player_controller
import time

# Set up environment (same as demo script)
experiment_name = 'pso_test'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10

# Initialize EvoMan for PSO
env = Environment(experiment_name=experiment_name,
                  enemies=[8],
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  visuals=False)

# PSO parameters
n_particles = 50   # Particles (population size)
n_iterations = 50  # Iterations (generations)
w = 0.5            # Inertia weight
c1 = 1.5           # Cognitive (influence personal best)
c2 = 1.5           # Social (influence global best (swarm))

# NN parameter size
n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5

# Random particle positions and velocities
particles = np.random.uniform(-1, 1, (n_particles, n_vars))
velocities = np.random.uniform(-0.1, 0.1, (n_particles, n_vars))

# Personal and global best positions
personal_best_positions = np.copy(particles)
personal_best_fitness = np.full(n_particles, -np.inf)
global_best_position = None
global_best_fitness = -np.inf

# Fitness function of particle (plays game with given weights)
def evaluate(particle):
    return np.array(list(map(lambda w: env.play(pcont=w)[0], particle)))

# Start first generation
fitness_values = evaluate(particles)

# Update personal / global best 
for i in range(n_particles):
    if fitness_values[i] > personal_best_fitness[i]:
        personal_best_fitness[i] = fitness_values[i]
        personal_best_positions[i] = particles[i]
    if fitness_values[i] > global_best_fitness:
        global_best_fitness = fitness_values[i]
        global_best_position = particles[i]

# PSO main loop
for iteration in range(n_iterations):
    print(f"Iteration {iteration + 1}/{n_iterations} - Global best fitness: {global_best_fitness}")

    # Update particle velocities / positions
    for i in range(n_particles):
        # Calculate new velocity
        inertia = w * velocities[i]
        cognitive = c1 * np.random.uniform(0, 1, n_vars) * (personal_best_positions[i] - particles[i])
        social = c2 * np.random.uniform(0, 1, n_vars) * (global_best_position - particles[i])
        velocities[i] = inertia + cognitive + social

        # Update position
        particles[i] = particles[i] + velocities[i]

        # Clip position within bounds [-1, 1]
        particles[i] = np.clip(particles[i], -1, 1)

    # Evaluate new fitness values
    fitness_values = evaluate(particles)

    # Update personal / global bests
    for i in range(n_particles):
        if fitness_values[i] > personal_best_fitness[i]:
            personal_best_fitness[i] = fitness_values[i]
            personal_best_positions[i] = particles[i]
        if fitness_values[i] > global_best_fitness:
            global_best_fitness = fitness_values[i]
            global_best_position = particles[i]

    # Save best solution so far
    np.savetxt(f"{experiment_name}/best_particle.txt", global_best_position)

print("PSO optimization complete.")
print(f"Best solution found with fitness: {global_best_fitness}")

# Save final bestest solution
np.savetxt(f"{experiment_name}/final_best.txt", global_best_position)






git push origin Evoman2024