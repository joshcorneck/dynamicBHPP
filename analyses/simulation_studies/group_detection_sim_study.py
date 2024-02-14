import numpy as np
import pickle
import os

from src.variational_bayes import VariationalBayes
from src.network_simulator import PoissonNetwork
# from src.plotting_functions.plot_maker_functions import (
#                             posterior_rate_mean,
#                             node_membership_probability,
#                             posterior_adjacency_mean,
#                             global_group_probability)

###
# STEP 1 - READ IN PARAMETERS
###

# Use index for reading relevant arguments
pbs_index = int(os.environ['PBS_ARRAYID'])

# Read in the relevant data from the array
with open('sim_params.txt', 'r') as file:
    lines = file.readlines()
    data = lines[pbs_index].split()

num_nodes = int(data[0])
num_groups_sim = int(data[1])
num_groups_alg = int(data[2])
n_cavi = int(data[3])
int_length = float(data[4])
delta_z = float(data[5])
delta_pi = float(data[6])
delta_rho = float(data[7])
delta_lam = float(data[8])

T_max = 50

# Read in the simulation arrays
loaded_data = np.load('sim_mats.npz')
lam_matrix = loaded_data['lam_mat']
rho_matrix = loaded_data['rho_mat']
group_sizes = loaded_data['group_sizes']
# mem_cp_times = loaded_data['change_point_times']
# mem_change_nodes = loaded_data['changing_nodes']
# num_mem_cps = len(mem_cp_times)
rate_cp_times = loaded_data['change_point_times']
num_rate_cps = len(rate_cp_times)


# If we want to specify rate changes
lam_matrices = np.load('lam_matrices.npy', allow_pickle=True)

###
# STEP 2 - SIMULATE A NETWORK
###
PN = PoissonNetwork(num_nodes, num_groups_sim, T_max,
                rho_matrix=rho_matrix, lam_matrix=lam_matrix)

# RATE CHANGE POINTS
sampled_network, groups_in_regions = (
    PN.sample_network(group_sizes=group_sizes,
                    rate_change=True,
                    num_rate_cps=num_rate_cps,
                    rate_change_times=rate_cp_times))
adj_mat = PN.adjacency_matrix

# # MEMBERSHIP CHANGE POINTS
# sampled_network, groups_in_regions = (
#     PN.sample_network(group_sizes=group_sizes,
#                     mem_change=True,
#                     num_mem_cps=num_mem_cps,
#                     mem_change_times=mem_cp_times,
#                     mem_change_nodes=mem_change_nodes))
# adj_mat = PN.adjacency_matrix

# sampled_network, groups_in_regions = (
#     PN.sample_network(group_sizes=group_sizes,
#                     rate_change=True,
#                     num_rate_cps=len(np.arange(30, 100, 0.25))))

print("Network simulated.")

###
# STEP 3 - RUN THE INFERENCE
###

# Parameter initialisations
sigma_init = (np.random.uniform(0,1,num_nodes ** 2).
              reshape((num_nodes,num_nodes)))
np.fill_diagonal(sigma_init,0)
param_prior = (np.array([1] * num_groups_alg ** 2).
                reshape((num_groups_alg, num_groups_alg)))

# Run the VB algorithm
# INFER GRAPH STRUCTURE
# VB = VariationalBayes(sampled_network=sampled_network, 
#                       num_nodes=num_nodes, 
#                       num_groups=num_groups_alg, 
#                       sigma_0=sigma_init,
#                       eta_0=param_prior, 
#                       zeta_0=param_prior, 
#                       alpha_0=param_prior, 
#                       beta_0=param_prior,
#                       n_0 = np.array([1/2] * int(num_groups_alg)))
# VB.run_full_var_bayes(delta_pi=delta_pi,
#                       delta_rho=delta_rho,
#                       delta_lam=delta_lam,
#                       n_cavi=n_cavi)

# KNOWN GRAPH STRUCTURE
VB = VariationalBayes(sampled_network=sampled_network, 
                      num_nodes=num_nodes, 
                      num_groups=num_groups_alg, 
                      alpha_0=param_prior, 
                      beta_0=param_prior,
                      gamma_0 = np.random.uniform(low=0.1, high=0.3, 
                                              size=int(num_groups_alg)),
                      adj_mat=adj_mat,
                      int_length=int_length,
                      T_max=T_max)
VB.run_full_var_bayes(delta_z=delta_z,
                      delta_pi=delta_pi,
                      delta_rho=delta_rho,
                      delta_lam=delta_lam,
                      n_cavi=n_cavi)

###
# STEP 4 - SAVE NETWORK AND INFERRED PARAMETERS
###

# Save the output
# lam_matrices = PN.lam_matrices
# with open(f'sim_output/lam_matrices_{pbs_index}.pkl','wb') as f:
#     pickle.dump(lam_matrices, f)
# f.close()

# with open(f'sim_output/groups_in_regions_{pbs_index}.pkl','wb') as f:
#     pickle.dump(groups_in_regions, f)
# f.close()

tau_store = VB.tau_store
with open(f'sim_output/tau_store_{pbs_index}.pkl','wb') as f:
    pickle.dump(tau_store, f)
f.close()

# sigma_store = VB.sigma_store
# with open(f'sim_output/sigma_store_{pbs_index}.pkl','wb') as f:
#     pickle.dump(sigma_store, f)
# f.close()

gamma_store = VB.gamma_store
with open(f'sim_output/gamma_store_{pbs_index}.pkl','wb') as f:
    pickle.dump(gamma_store, f)
f.close()

alpha_store = VB.alpha_store
with open(f'sim_output/alpha_store_{pbs_index}.pkl','wb') as f:
    pickle.dump(alpha_store, f)
f.close()

beta_store = VB.beta_store
with open(f'sim_output/beta_store_{pbs_index}.pkl','wb') as f:
    pickle.dump(beta_store, f)
f.close()

# eta_store = VB.eta_store
# with open(f'sim_output/eta_store_{pbs_index}.pkl','wb') as f:
#     pickle.dump(eta_store, f)
# f.close()

# zeta_store = VB.zeta_store
# with open(f'sim_output/zeta_store_{pbs_index}.pkl','wb') as f:
#     pickle.dump(zeta_store, f)
# f.close()

# KL_div = VB.KL_div
# with open(f'sim_output/KL_div_{pbs_index}.pkl','wb') as f:
#     pickle.dump(KL_div, f)
# f.close()

# ###
# # STEP 5 - PRODUCE AND SAVE PLOTS
# ###

# posterior_rate_mean(num_groups=num_groups,
#                     alpha=alpha_store,
#                     beta=beta_store,
#                     true_rates=lam_matrix,
#                     file_path=f'sim_output/plots/posterior_rates_{pbs_index}')

# # if change_node:
# #     node_membership_probability()

# posterior_adjacency_mean(num_groups=num_groups,
#                          eta=eta_store,
#                          zeta=zeta_store,
#                          true_con_prob=rho_matrix,
#                          file_path=f'sim_output/plots/posterior_adjacency_{pbs_index}')

# global_group_probability(n=n_store,
#                          true_pi=group_sizes/group_sizes.sum(),
#                          file_path=f'sim_output/plots/posterior_group_prob_{pbs_index}')