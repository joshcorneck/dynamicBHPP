import numpy as np
import pickle
import os
import json

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
group_sizes = [np.array(x) for x in data[4]]
n_cavi = int(data[5])
int_length = float(data[6])
delta = float(data[7])
T_max = float(data[8])
prop_new_group_1 = float(data[9])

group_num_change_times = [2, 3]

group1 = int(prop_new_group_1 * num_nodes)
group2 = num_nodes - group1
group_sizes.append(np.array([group1, group2]))
print(f"Group sizes: {group_sizes}")

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
                          group_num_change=True,
                          group_num_change_times=group_num_change_times  
                          )
        )

    adj_mat = PN.adjacency_matrix

    with open(f'param_output/true_changes/groups_in_regions_{pbs_index}_{glob_iteration}.pkl','wb') as f:
        pickle.dump(groups_in_regions, f)
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
                          nu_0=1,
                          infer_num_groups_bool=True,
                          num_var_groups=2,
                          adj_mat=adj_mat,
                          int_length=int_length,
                          T_max=T_max
                          )
    VB.run_full_var_bayes(delta_pi=delta,
                          delta_rho=delta,
                          delta_lam=delta,
                          n_cavi=n_cavi,
                          cp_burn_steps=5,
                          cp_stationary_steps=10,
                          cp_kl_lag_steps=2,
                          cp_kl_thresh=100,
                          cp_rate_wait=0
                          )
    
    print("...Inference procedure completed...")

    ###
    # STEP 4 - SAVE NETWORK AND INFERRED PARAMETERS
    ###

    tau_store = VB.tau_store
    with open(f'param_output/tau/tau_store_{pbs_index}_{glob_iteration}.pkl','wb') as f:
        pickle.dump(tau_store, f)
    f.close()

    alpha_store = VB.alpha_store
    with open(f'param_output/alpha/alpha_store_{pbs_index}_{glob_iteration}.pkl','wb') as f:
        pickle.dump(alpha_store, f)
    f.close()

    beta_store = VB.beta_store
    with open(f'param_output/beta/beta_store_{pbs_index}_{glob_iteration}.pkl','wb') as f:
        pickle.dump(beta_store, f)
    f.close()

