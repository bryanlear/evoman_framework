import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from evoman.environment import Environment

n_runs = 10
enemies = [2, 5, 8]

# Load the gain data
gains_data = np.load('gains_data_pso.npy', allow_pickle=True).item()

#print(gains_data)

# Access enemy data
enemy_2_gains = gains_data[2]
print(f"Gains for Enemy 2 across 10 runs: {enemy_2_gains}")

enemy_5_gains = gains_data[5]
print(f"Gains for Enemy 5 across 10 runs: {enemy_5_gains}")

enemy_8_gains = gains_data[8]
print(f"Gains for Enemy 8 across 10 runs: {enemy_8_gains}")

