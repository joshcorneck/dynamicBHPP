#%%
import pandas as pd
import numpy as np
import itertools
import json

"""
SIMULATION STUDY 1.
This simulation study is for a varying proportion of group 1 nodes swapping
to group 2 at a given time of 3
"""
delta_set = [0.1, 1]
prop_swap = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]

all_combinations = list(
    itertools.product(delta_set, 
                      prop_swap)
)

with open("analyses/simulation_studies/"
          "simulation_parameters/simulation_study_1/"
          "sim_params_simulation_study_1.json", 'w') as json_file:
    json.dump(all_combinations, json_file)

print(len(all_combinations))

# %%

