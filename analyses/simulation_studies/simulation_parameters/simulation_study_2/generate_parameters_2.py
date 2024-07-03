#%%
import pandas as pd
import numpy as np
import itertools
import json

# Rate matrices
rate_matrices = [
    [[[2.0, 1.0], 
     [0.3, 8.0]],
    [[5, 1.0],
     [0.3, 8.0]],
    [[3, 1.0],
     [0.3, 8]]]
     ]

# Other simulation and inference parameters
delta_set = [0.1, 1]
change_steps_gap = [10, 5, 4, 3, 2, 1]

all_combinations = list(
    itertools.product(rate_matrices,
                      delta_set, 
                      change_steps_gap)
)

with open("analyses/simulation_studies/"
          "simulation_parameters/simulation_study_2/"
          "sim_params_simulation_study_2.json", 'w') as json_file:
    json.dump(all_combinations, json_file)

print(len(all_combinations))

# %%

