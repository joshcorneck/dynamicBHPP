#%%
import pandas as pd
import numpy as np
import itertools
import json

"""
SIMULATION STUDY 4.
This simulation study is for a node change of 
"""
# Rate and connection matrices

# Other simulation and inference parameters
connection_prob = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1]

all_combinations = list(
    itertools.product(connection_prob)
)

with open("analyses/simulation_studies/"
          "simulation_parameters/simulation_study_4/"
          "sim_params_simulation_study_4.json", 'w') as json_file:
    json.dump(all_combinations, json_file)

print(len(all_combinations))

# %%

