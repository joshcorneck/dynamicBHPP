import numpy as np
import pickle
import os

from src.variational_bayes import VariationalBayes

###
# STEP 1 - READ IN PARAMETERS
###

# Use index for reading relevant arguments
pbs_index = int(os.environ['PBS_ARRAYID'])

# Load the enron_network
with open('enron_network.pkl', 'rb') as file:
    enron_network = pickle.load(file)

# Read in the relevant data from the array
with open('enron_params.txt', 'r') as file:
    lines = file.readlines()
    data = lines[pbs_index].split()

num_nodes = int(data[0])
num_groups = int(data[1])
n_cavi = int(data[2])
delta_pi = float(data[3])
delta_rho = float(data[4])
delta_lam = float(data[5])
T_max = 189

# Parameter initialisations
sigma_init = (np.random.uniform(0,1,num_nodes ** 2).
              reshape((num_nodes,num_nodes)))
np.fill_diagonal(sigma_init,0)
param_prior = (np.array([1] * num_groups ** 2).
                reshape((num_groups, num_groups)))

###
# STEP 3 - RUN THE INFERENCE
###

# Run the VB algorithm
VB = VariationalBayes(sampled_network=enron_network, 
                      num_nodes=num_nodes, 
                      num_groups=num_groups, 
                      sigma_0=sigma_init,
                      eta_0=param_prior, 
                      zeta_0=param_prior, 
                      alpha_0=param_prior, 
                      beta_0=param_prior,
                      n_0 = np.random.uniform(low=0.4, high=0.6, size=int(num_groups)),
                      T_max=T_max)
VB.run_full_var_bayes(delta_pi=delta_pi,
                      delta_rho=delta_rho,
                      delta_lam=delta_lam,
                      n_cavi=n_cavi)

###
# STEP 4 - SAVE NETWORK AND INFERRED PARAMETERS
###

# Save the output
tau_store = VB.tau_store
with open(f'enron_output/tau_store_{pbs_index}.pkl','wb') as f:
    pickle.dump(tau_store, f)
f.close()

sigma_store = VB.sigma_store
with open(f'enron_output/sigma_store_{pbs_index}.pkl','wb') as f:
    pickle.dump(sigma_store, f)
f.close()

n_store = VB.n_store
with open(f'enron_output/n_store_{pbs_index}.pkl','wb') as f:
    pickle.dump(n_store, f)
f.close()

alpha_store = VB.alpha_store
with open(f'enron_output/alpha_store_{pbs_index}.pkl','wb') as f:
    pickle.dump(alpha_store, f)
f.close()

beta_store = VB.beta_store
with open(f'enron_output/beta_store_{pbs_index}.pkl','wb') as f:
    pickle.dump(beta_store, f)
f.close()

eta_store = VB.eta_store
with open(f'enron_output/eta_store_{pbs_index}.pkl','wb') as f:
    pickle.dump(eta_store, f)
f.close()

zeta_store = VB.zeta_store
with open(f'enron_output/zeta_store_{pbs_index}.pkl','wb') as f:
    pickle.dump(zeta_store, f)
f.close()

KL_div = VB.KL_div
with open(f'enron_output/KL_div_{pbs_index}.pkl','wb') as f:
    pickle.dump(KL_div, f)
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