#%%
import pandas as pd
import numpy as np
import itertools

# Simulation arrays
lam_mat = np.array([[5., 3.], [2., 8.]])
rho_mat = np.array([[1, 1], [1, 1]])
group_sizes = np.array([300, 200])
change_point_times = np.array([30])
mem_change_nodes = np.array([10])

# lam_matrices = []
# lam_matrices.append(lam_mat)
# lam_mat_temp = lam_mat.copy()

# base_val_00 = lam_mat_temp[0,0]
# base_val_11 = lam_mat_temp[1,1]

# for time in np.arange(30,100,0.25):
    # lam_mat_temp[0,0] = base_val_00 * (1 + np.sin(2 * np.pi * (time - 30) / 25) / 2)
    # lam_mat_temp[1,1] = base_val_11 * (1 + np.cos(2 * np.pi * (time - 30) / 25) / 2 - 1/2)
    # lam_matrices.append(lam_mat_temp.copy())

# change_point_times = np.arange(30,100,0.25)

np.savez('analyses/simulation_studies/sim_mats.npz', 
         lam_mat=lam_mat, rho_mat=rho_mat, group_sizes=group_sizes,
         change_point_times=change_point_times,
         changing_nodes=mem_change_nodes)
# np.save('analyses/simulation_studies/lam_matrices.npy', lam_matrices)

num_nodes_set = [500]
num_groups_sim_set = [2]
num_groups_alg_set = [2]
n_cavi_set = [2]
int_length_set = [1, 0.1, 0.01]
delta_z_set = [1, 0.1]
delta_pi_set = [1, 0.1]
delta_rho_set = [1]
delta_lam_set = [1, 0.1]

all_combinations = list(
    itertools.product(num_nodes_set, num_groups_sim_set, 
                      num_groups_alg_set, n_cavi_set, int_length_set,
                      delta_z_set, delta_pi_set, delta_rho_set, 
                      delta_lam_set)
)

with open('analyses/simulation_studies/sim_params.txt', 'w') as file:
    for combination in all_combinations:
        line = ' '.join(map(str, combination)) + '\n'
        file.write(line)

print(len(all_combinations))

df_combs = pd.DataFrame(all_combinations, 
                        columns=['num_nodes', 'num_groups_sim', 
                                'num_groups_alg', 'n_cavi',
                                'int_length', 'delta_z',
                                'delta_pi', 'delta_rho',
                                'delta_lam'])
df_combs.to_pickle('analyses/simulation_studies/df_sim_params.pkl')
# %%

