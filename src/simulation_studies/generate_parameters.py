#%%
import pandas as pd
import numpy as np
import itertools

num_nodes_set = [100, 500, 1000]
num_groups_set = [2]
n_cavi_set = [1]
delta_set = [0.01, 0.1, 0.5, 1, 2, 5]
int_length_set = [0.005, 0.01, 0.05]
T_max_set = [1]

all_combinations = list(
    itertools.product(num_nodes_set, num_groups_set, n_cavi_set,
                      delta_set, int_length_set, T_max_set)
)

with open('simulation_parameters.txt', 'w') as file:
    for combination in all_combinations:
        line = ' '.join(map(str, combination)) + '\n'
        file.write(line)

print(len(all_combinations))