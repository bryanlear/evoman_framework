import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from evoman.environment import Environment
from demo_controller import player_controller  # Import your controller

# Initialize environment
experiment_name = 'final_best_testing'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# Parameters
n_hidden_neurons = 10
n_runs = 10
enemies = [3, 6, 8]

# EvoMan environment
env = Environment(experiment_name=experiment_name,
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  contacthurt="player", 
                  visuals=False) 

# Define the function to calculate individual gain
def calculate_individual_gain(player_energy, enemy_energy):
    return player_energy - enemy_energy

# Dictionary to store gain data
gains_data = {}

# Iterate over each enemy and run
for enemy in enemies:
    gains_data[enemy] = []
    
    for run in range(1, n_runs + 1):
        # Load the best solution for this run
        best_solution_path = f'evoman_experiments/enemy_{enemy}/{run}/final_best.txt'
        best_solution = np.loadtxt(best_solution_path)

        individual_gains = []

        # Update environment to match current enemy
        env.update_parameter('enemies', [enemy])
        
        # Test solution 5 times
        for _ in range(5):
            # Run simulation and obtain player's energy and enemy's energy
            fitness, player_energy, enemy_energy, _ = env.play(pcont=best_solution)
            individual_gain = calculate_individual_gain(player_energy, enemy_energy)
            individual_gains.append(individual_gain)

        # Calculate mean gain from the 5 runs and store it
        gains_data[enemy].append(np.mean(individual_gains))

# Save  gain data for future use
np.save('gains_data_pso.npy', gains_data)
print("Gain data saved successfully!")