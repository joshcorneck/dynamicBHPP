import numpy as np
import pickle
import os

from src.variational_bayes import VariationalBayes

###
# STEP 1 - READ IN PARAMETERS
###

# Use index for reading relevant arguments
pbs_index = int(os.environ['PBS_ARRAYID'])

# Load the high_school_network
with open('high_school_network_2012_0.pkl', 'rb') as file:
    high_school_network = pickle.load(file)

# Read in the relevant data from the array
with open('high_school_2012_params.txt', 'r') as file:
    lines = file.readlines()
    data = lines[pbs_index].split()

num_nodes = int(data[0])
num_groups = int(data[1])
n_cavi = int(data[2])
delta_z = float(data[3])
delta_pi = float(data[4])
delta_lam = float(data[5])

adj_mat = np.ones((num_nodes, num_nodes))
np.fill_diagonal(adj_mat, 0)

T_max = 100
int_length = 1

# Parameter initialisations
param_prior = (np.array([1] * num_groups ** 2).
                reshape((num_groups, num_groups)))

###
# STEP 3 - RUN THE INFERENCE
###

# Run the VB algorithm
VB = VariationalBayes(sampled_network=high_school_network, 
                      num_nodes=num_nodes, 
                      num_groups=num_groups, 
                      alpha_0=param_prior, 
                      beta_0=param_prior,
                      gamma_0 = np.random.uniform(low=0.1, high=0.3, 
                                              size=int(num_groups)),
                      adj_mat=adj_mat,
                      int_length=int_length,
                      T_max=T_max)
VB.run_full_var_bayes(delta_z=delta_z,
                      delta_pi=delta_pi,
                      delta_rho=1,
                      delta_lam=delta_lam,
                      n_cavi=n_cavi)

###
# STEP 4 - SAVE NETWORK AND INFERRED PARAMETERS
###

# Save the output
tau_store = VB.tau_store
with open(f'high_school_output/tau_store_{pbs_index}.pkl','wb') as f:
    pickle.dump(tau_store, f)
f.close()

gamma_store = VB.gamma_store
with open(f'high_school_output/n_store_{pbs_index}.pkl','wb') as f:
    pickle.dump(gamma_store, f)
f.close()

alpha_store = VB.alpha_store
with open(f'high_school_output/alpha_store_{pbs_index}.pkl','wb') as f:
    pickle.dump(alpha_store, f)
f.close()

beta_store = VB.beta_store
with open(f'high_school_output/beta_store_{pbs_index}.pkl','wb') as f:
    pickle.dump(beta_store, f)
f.close()

# ###
# # STEP 5 - PRODUCE AND SAVE PLOTS
# ###

# posterior_rate_mean(num_groups=num_groups,
#                     alpha=alpha_store,
#                     beta=beta_store,
#                     true_rates=lam_matrix,
#                     file_path=f'output/plots/posterior_rates_{pbs_index}')

# # if change_node:
# #     node_membership_probability()

# posterior_adjacency_mean(num_groups=num_groups,
#                          eta=eta_store,
#                          zeta=zeta_store,
#                          true_con_prob=rho_matrix,
#                          file_path=f'output/plots/posterior_adjacency_{pbs_index}')

# global_group_probability(n=n_store,
#                          true_pi=group_sizes/group_sizes.sum(),
#                          file_path=f'output/plots/posterior_group_prob_{pbs_index}')