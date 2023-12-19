#%%
import pandas as pd
import numpy as np
import itertools

# Simulation arrays
lam_mat = np.array([[5, 1], [2, 8]])
rho_mat = np.array([[0.6, 0.1], [0.2, 0.7]])
group_sizes = np.array([300, 200])
change_point_times = np.array([30])

np.savez('online-networks-change-points/analyses/simulation_studies/sim_mats.npz', 
         lam_mat=lam_mat, rho_mat=rho_mat, group_sizes=group_sizes,
         change_point_times=change_point_times)

num_nodes_set = [500]
num_groups_set = [2]
n_cavi_set = [2]
delta_z_set = [1]
delta_pi_set = [1]
delta_lam_set = [1]

all_combinations = list(
    itertools.product(num_nodes_set, num_groups_set, n_cavi_set,
                      delta_z_set, delta_pi_set, delta_lam_set)
)

with open('online-networks-change-points/analyses/simulation_studies/sim_params.txt', 'w') as file:
    for combination in all_combinations:
        line = ' '.join(map(str, combination)) + '\n'
        file.write(line)

print(len(all_combinations))
