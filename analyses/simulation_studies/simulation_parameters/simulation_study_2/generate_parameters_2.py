#%%
import pandas as pd
import numpy as np
import itertools
import json

"""
SIMULATION STUDY 1.
This simulation study is for one changing node in a fully connected graph
using the base model.
"""
# Rate and connection matrices
rate_matrices = [
    [[[2.0, 1.0], 
     [0.3, 8.0]],
    [[5, 1.0],
     [0.3, 8.0]],
    [[3, 1.0],
     [0.3, 8]]]
     ]
rho_mat = [[[1, 1], [1, 1]]]

# Other simulation and inference parameters
num_nodes_set = [500]
num_groups_set = [2]
group_props_set = [[0.6, 0.4]]
n_cavi_set = [3]
int_length_set = [0.1]
delta_set = [0.1, 1]
T_max = [5]
change_steps_gap = [10, 5, 4, 3, 2, 1]

all_combinations = list(
    itertools.product(rate_matrices,
                      rho_mat,
                      num_nodes_set, 
                      num_groups_set,
                      group_props_set, 
                      n_cavi_set, 
                      int_length_set,
                      delta_set, 
                      T_max,
                      change_steps_gap)
)

with open("analyses/simulation_studies/"
          "simulation_parameters/simulation_study_2/"
          "sim_params_simulation_study_2.json", 'w') as json_file:
    json.dump(all_combinations, json_file)

print(len(all_combinations))

# %%

