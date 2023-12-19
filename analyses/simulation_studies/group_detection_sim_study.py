import numpy as np
import pickle
import os

from variational_bayes import VariationalBayes
from network_simulator import PoissonNetwork
from plot_maker_functions import (posterior_rate_mean,
                                  node_membership_probability,
                                  posterior_adjacency_mean,
                                  global_group_probability)

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
num_groups = int(data[1])
n_cavi = int(data[2])
delta_z = int(data[3])
delta_pi = int(data[4])
delta_lam = int(data[5])

T_max = 100

# Read in the simulation arrays
loaded_data = np.load('sim_mats.npz')
lam_matrix = loaded_data['lam_mat']
rho_matrix = loaded_data['rho_mat']
group_sizes = loaded_data['group_sizes']
rate_cp_times = loaded_data['change_point_times']
num_rate_cps = len(change_point_times)

###
# STEP 2 - SIMULATE A NETWORK
###
PN = PoissonNetwork(num_nodes, num_groups, T_max,
                rho_matrix=rho_matrix, lam_matrix=lam_matrix)
sampled_network, groups_in_regions = (
    PN.sample_network(group_sizes=group_sizes,
                    rate_change=True,
                    num_rate_cps=num_rate_cps,
                    rate_change_times=rate_cp_times))

# Parameter initialisations
sigma_init = np.random.uniform(0,1,num_nodes ** 2).reshape((num_nodes,num_nodes))
np.fill_diagonal(sigma_init,0)
param_prior = np.array([1] * num_groups ** 2).reshape((num_groups, num_groups))

###
# STEP 3 - RUN THE INFERENCE
###

# Run the VB algorithm
VB = VariationalBayes(sampled_network=sampled_network, num_nodes=num_nodes, 
                      num_groups=num_groups, sigma_0=sigma_init,
                      eta_0=param_prior, zeta_0=param_prior, 
                      alpha_0=param_prior, beta_0=param_prior,
                      n_0 = np.array([1/2 - 0.01, 1/2 + 0.01] * int(num_groups / 2)))
VB.run_full_var_bayes(n_cavi=n_cavi)

###
# STEP 4 - SAVE NETWORK AND INFERRED PARAMETERS
###

# Save the output
lam_matrices = PN.lam_matrices
with open(f'output/lam_matrices_{pbs_index}.pkl','wb') as f:
    pickle.dump(lam_matrices, f)
f.close()

with open(f'output/groups_in_regions_{pbs_index}.pkl','wb') as f:
    pickle.dump(groups_in_regions, f)
f.close()

tau_store = VB.tau_store
with open(f'output/tau_store_{pbs_index}.pkl','wb') as f:
    pickle.dump(tau_store, f)
f.close()

sigma_store = VB.sigma_store
with open(f'output/sigma_store_{pbs_index}.pkl','wb') as f:
    pickle.dump(sigma_store, f)
f.close()

n_store = VB.n_store
with open(f'output/n_store_{pbs_index}.pkl','wb') as f:
    pickle.dump(n_store, f)
f.close()

alpha_store = VB.alpha_store
with open(f'output/alpha_store_{pbs_index}.pkl','wb') as f:
    pickle.dump(alpha_store, f)
f.close()

beta_store = VB.beta_store
with open(f'output/beta_store_{pbs_index}.pkl','wb') as f:
    pickle.dump(beta_store, f)
f.close()

eta_store = VB.eta_store
with open(f'output/eta_store_{pbs_index}.pkl','wb') as f:
    pickle.dump(eta_store, f)
f.close()

zeta_store = VB.zeta_store
with open(f'output/zeta_store_{pbs_index}.pkl','wb') as f:
    pickle.dump(zeta_store, f)
f.close()

###
# STEP 5 - PRODUCE AND SAVE PLOTS
###
# posterior_rate_mean(num_groups, alpha_store[])