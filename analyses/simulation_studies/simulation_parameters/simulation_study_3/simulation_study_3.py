import numpy as np
import pickle
import os
import json
import time

from src.variational_bayes import VariationalBayes
from src.network_simulator import PoissonNetwork

"""
SIMULATION STUDY 3.
This simulation study is for one changing node in a fully connected graph
using the base model.
This simulation study does 50 iterations for a given set of parameter 
combinations.
"""

# Number of times we simulate.
N_runs = 50

###
# READ IN PARAMETERS
###

print("...Loading parameters...")

# Use index for reading relevant arguments
pbs_index = int(os.environ['PBS_ARRAYID'])

# Read in the relevant data from the array
with open('sim_params_simulation_study_3.json', 'r') as file:
    full_data = json.load(file)
    data = full_data[pbs_index]

lam_matrix = np.array(data[0])
rho_matrix = np.array(data[1])
num_nodes = int(data[2])
num_groups = int(data[3])
group_props = np.array(data[4])
n_cavi = int(data[5])
int_length = float(data[6])
delta = float(data[7])
T_max = float(data[8])
num_rate_changes = int(data[9])

# Extract the group sizes
group_sizes = (group_props * num_nodes).astype('int')

# Ensure that group_sizes sums to num_nodes and there are no empty groups
missing_nodes = num_nodes - group_sizes.sum()
if missing_nodes != 0:
    if np.sum(np.where(group_sizes == 0)) != 0:
        group_sizes[np.where(group_sizes == 0)] = missing_nodes
    else:
        group_sizes[-1] += missing_nodes

## Function for sampling rate change times
def sample_numbers_with_min_distance(B, num_rate_changes, T_max, cp_start_time,
                                     timeout=1):

    sampled_numbers = []
    sampled_numbers.append(np.random.uniform(cp_start_time, T_max - B))
    
    # Sample subsequent numbers ensuring minimum distance
    for _ in range(1, num_rate_changes):
        
        new_number = np.random.uniform(cp_start_time, T_max - B)
        
        # Ensure minimum distance from previous sampled numbers
        # Start timer to ensure not getting stuck
        start_time = time.time()
        while any(abs(new_number - x) < B for x in sampled_numbers):
            new_number = np.random.uniform(cp_start_time, T_max)
            if time.time() - start_time > timeout:
                print("Timeout reached. Restarting...")
                return sample_numbers_with_min_distance(
                    B, num_rate_changes, T_max, cp_start_time)
        
        sampled_numbers.append(new_number)
    
    return sampled_numbers
    
# Create rate change times
rate_change_times = np.sort(
    sample_numbers_with_min_distance(0.4, num_rate_changes, T_max, 2)
    )
with open(f'param_output/true_changes/rate_change_times_{pbs_index}.pkl','wb') as f:
        pickle.dump(rate_change_times, f)
        f.close()

for glob_iteration in range(N_runs):

    print(f"...GLOBAL ITERATION: {glob_iteration + 1}...")

    ###
    # SIMULATE A NETWORK
    ###

    print("...Beginning network simulation...")

    PN = PoissonNetwork(num_nodes=num_nodes, 
                        num_groups=num_groups, 
                        T_max=T_max, 
                        rho_matrix=rho_matrix, 
                        lam_matrix=lam_matrix)

    sampled_network, groups_in_regions = (
        PN.sample_network(group_sizes=group_sizes,
                          rate_change=True,
                          rate_change_times=rate_change_times
                          )
        )

    adj_mat = PN.adjacency_matrix

    with open(f'param_output/true_changes/rate_matrices_{pbs_index}_{glob_iteration}.pkl','wb') as f:
        pickle.dump(PN.lam_matrices, f)
        f.close()

    print("...Network simulated...")

    ###
    # RUN THE INFERENCE
    ###

    print("...Beginning inference procedure...")

    # KNOWN GRAPH STRUCTURE
    VB = VariationalBayes(sampled_network=sampled_network, 
                          num_nodes=num_nodes, 
                          num_groups=num_groups, 
                          alpha_0=1, 
                          beta_0=1,
                          gamma_0 = np.array([0.99, 1.01]),
                          adj_mat=adj_mat,
                          int_length=int_length,
                          T_max=T_max
                          )
    VB.run_full_var_bayes(delta_pi=delta,
                          delta_rho=delta,
                          delta_lam=delta,
                          n_cavi=n_cavi,
                          cp_burn_steps=10,
                          cp_kl_lag_steps=2,
                          cp_kl_thresh=10,
                          cp_rate_wait=0.4
                          )
    
    print("...Inference procedure completed...")

    ###
    # STEP 4 - SAVE NETWORK AND INFERRED PARAMETERS
    ###

    tau_store = VB.tau_store
    with open(f'param_output/tau/tau_store_{pbs_index}_{glob_iteration}.pkl','wb') as f:
        pickle.dump(tau_store, f)
    f.close()

    gamma_store = VB.gamma_store
    with open(f'param_output/gamma/gamma_store_{pbs_index}_{glob_iteration}.pkl','wb') as f:
        pickle.dump(gamma_store, f)
    f.close()

    alpha_store = VB.alpha_store
    with open(f'param_output/alpha/alpha_store_{pbs_index}_{glob_iteration}.pkl','wb') as f:
        pickle.dump(alpha_store, f)
    f.close()

    beta_store = VB.beta_store
    with open(f'param_output/beta/beta_store_{pbs_index}_{glob_iteration}.pkl','wb') as f:
        pickle.dump(beta_store, f)
    f.close()

    flagged_changes = np.array(VB.rate_changes_list)
    with open(f'param_output/inf_changes/changes_store_{pbs_index}_{glob_iteration}.pkl','wb') as f:
        pickle.dump(flagged_changes, f)
    f.close()
