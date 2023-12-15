#%%
import pandas as pd
import numpy as np
import itertools

num_nodes_set = [500]
num_groups_set = [2, 2, 2]
n_cavi_set = [2]
delta_z_set = [1]
delta_pi_set = [1]
delta_lam_set = [1]
# int_length_set = [1]
# T_max_set = [100]

all_combinations = list(
    itertools.product(num_nodes_set, num_groups_set, n_cavi_set,
                      delta_z_set, delta_pi_set, delta_lam_set)
)

with open('simulation_parameters.txt', 'w') as file:
    for combination in all_combinations:
        line = ' '.join(map(str, combination)) + '\n'
        file.write(line)

print(len(all_combinations))
# %%
