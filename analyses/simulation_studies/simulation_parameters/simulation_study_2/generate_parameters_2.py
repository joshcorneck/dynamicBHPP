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
lam_mat = [[[2.0, 1.0], [0.3, 8.0]]]
rho_mat = [[[1, 1], [1, 1]]]


# Other simulation and inference parameters
num_nodes_set = [100, 500]
num_groups_set = [2]
group_props_set = [[0.1, 0.9], [0.4, 0.6]]
n_cavi_set = [3]
int_length_set = [0.1]
delta_set = [0.1, 1]
T_max = [10]
num_mem_changes = [10, 25, 50]

all_combinations = list(
    itertools.product(lam_mat,
                      rho_mat,
                      num_nodes_set, 
                      num_groups_set,
                      group_props_set, 
                      n_cavi_set, 
                      int_length_set,
                      delta_set, 
                      T_max,
                      num_mem_changes)
)

with open("analyses/simulation_studies/"
          "simulation_parameters/simulation_study_2/"
          "sim_params_simulation_study_2.json", 'w') as json_file:
    json.dump(all_combinations, json_file)

print(len(all_combinations))

# %%

