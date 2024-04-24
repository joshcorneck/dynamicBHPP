#%%
import pandas as pd
import numpy as np
import itertools
import json

"""
SIMULATION STUDY 3.
This simulation study is to compute ARL0 for the model.
"""
# Rate and connection matrices
lam_mat = [[[2.0, 1.0], [0.3, 8.0]]]
rho_mat = [[[1, 1], [1, 1]]]


# Other simulation and inference parameters
num_nodes_set = [500]
num_groups_set = [2]
init_groups_set = [[[300, 200], [500, 0]]]
n_cavi_set = [3]
int_length_set = [0.1]
delta_set = [0.1, 1]
T_max = [5]
prop_new_groups_set = [0.01, 0.1, 0.25, 0.5, 0.75, 0.95]

all_combinations = list(
    itertools.product(lam_mat,
                      rho_mat,
                      num_nodes_set, 
                      num_groups_set,
                      init_groups_set, 
                      n_cavi_set, 
                      int_length_set,
                      delta_set, 
                      T_max, 
                      prop_new_groups_set)
)

with open("analyses/simulation_studies/"
          "simulation_parameters/simulation_study_3/"
          "sim_params_simulation_study_3.json", 'w') as json_file:
    json.dump(all_combinations, json_file)

print(len(all_combinations))

# %%

