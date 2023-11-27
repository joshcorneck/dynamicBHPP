import numpy as np
import pickle
import os

from variational_bayes import VariationalBayes

# Use index for reading relevant arguments
pbs_index = int(os.environ['PBS_ARRAY_INDEX'])

# Read in the relevant data from the array
with open('simulation_parameters.txt', 'r') as file:
    lines = file.readlines()
    data = lines[pbs_index].split()

num_nodes = int(data[0])
num_groups = int(data[1])
n_cavi = int(data[2])
delta = float(data[3])
int_length = float(data[4])
T_max = float(data[5])

# Load the simulated data
with open('sampled_network.pkl', 'rb') as f:
    sampled_network_full = pickle.load(f) 
f.close()

# Extract relevant data subset
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

# Run the VB algorithm
VB = VariationalBayes(sampled_network, num_nodes, num_groups, T_max, int_length, 
                      np.array([0.01] * num_groups ** 2).reshape((num_groups, num_groups)), 
                      np.array([0.01] * num_groups ** 2).reshape((num_groups,num_groups)),
                      np.array([1/2, 1/2 + 0.01] * int(num_groups / 2)),
                      simple=False)
VB.run_full_var_bayes(delta_z=delta, n_cavi=n_cavi)

# Save the output
tau_store = VB.tau_store
with open(f'output/tau_store_{pbs_index}.pkl','wb') as f:
    pickle.dump(tau_store, f)
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
