#%%
import pandas as pd
import numpy as np
import itertools
import json

"""
SIMULATION STUDY 3.
This simulation study is to compute ARL0 for the model.
"""
# Other simulation and inference parameters
init_groups_set = [[[300, 200], [500, 0]]]
delta_set = [0.1, 1]
prop_new_groups_set = [0.01, 0.1, 0.25, 0.5, 0.75, 0.95]

all_combinations = list(
    itertools.product(init_groups_set, 
                      delta_set, 
                      prop_new_groups_set)
)

with open("analyses/simulation_studies/"
          "simulation_parameters/simulation_study_3/"
          "sim_params_simulation_study_3.json", 'w') as json_file:
    json.dump(all_combinations, json_file)

print(len(all_combinations))

# %%

