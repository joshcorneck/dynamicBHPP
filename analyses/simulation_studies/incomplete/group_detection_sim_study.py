import numpy as np
import pickle
import os

from variational_bayes import VariationalBayes

# Use index for reading relevant arguments
pbs_index = int(os.environ['PBS_ARRAYID'])

print(pbs_index)

# Read in the relevant data from the array
with open('simulation_parameters.txt', 'r') as file:
    lines = file.readlines()
    data = lines[pbs_index].split()

num_nodes = int(data[0])
num_groups = int(data[1])
n_cavi = int(data[2])
delta_z = int(data[3])
delta_pi = int(data[4])
delta_lam = int(data[5])
int_length = float(data[6])
T_max = int(data[7])

# Load the simulated data
with open('sampled_network.pkl', 'rb') as f:
    sampled_network_full = pickle.load(f) 
f.close()

subset = False
# Extract relevant data subset
if subset:
    if num_nodes != 1000:
        sampled_network = {}

        # Iterate over the first keys
        for key in list(sampled_network_full.keys())[:num_nodes]:
            # Extract the nested dictionary
            inner_dict = sampled_network_full[key]

            # Take only the first num_nodes
            inner_dict_subset = {k: inner_dict[k] for k in list(inner_dict.keys())[:num_nodes]}

            # Store the result in the new dictionary
            sampled_network[key] = inner_dict_subset
else:
    sampled_network = sampled_network_full

# Parameter initialisations
sigma_init = np.random.uniform(0,1,num_nodes ** 2).reshape((num_nodes,num_nodes))
np.fill_diagonal(sigma_init, 0)
param_prior = np.array([1] * num_groups ** 2).reshape((num_groups, num_groups))

# Run the VB algorithm
VB = VariationalBayes(sampled_network, num_nodes, num_groups, T_max, int_length, 
                      np.array([0.01] * num_groups ** 2).reshape((num_groups, num_groups)), 
                      np.array([0.01] * num_groups ** 2).reshape((num_groups,num_groups)),
                      np.array([1/2, 1/2 + 0.01] * int(num_groups / 2)),
                      simple=False)
VB = VariationalBayes(sampled_network, num_nodes, num_groups, T_max, 
                      int_length, sigma_0=sigma_init,
                      eta_0=param_prior, zeta_0=param_prior, 
                      alpha_0=param_prior, beta_0=param_prior,
                      n_0 = np.array([1/2 - 0.01, 1/2 + 0.01] * int(num_groups / 2)),
                      simple=False)
VB.run_full_var_bayes(n_cavi=n_cavi)

# Save the output
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
