import numpy as np
import pickle
import os
import json

from src.variational_bayes import VariationalBayes
from src.network_simulator import PoissonNetwork

"""
SIMULATION STUDY 4.
This simulation study is for one changing node in a graph with decreasing
connection density. We run once to infer the graph, and once with an assumed
fully-connected graph. This simulation study does 50 iterations for a given 
set of parameter combinations.
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
with open('sim_params_simulation_study_4.json', 'r') as file:
    full_data = json.load(file)
    data = full_data[pbs_index]

lam_matrix = np.array(data[0])
num_nodes = int(data[1])
num_groups = int(data[2])
group_props = np.array(data[3])
n_cavi = int(data[4])
int_length = float(data[5])
delta = float(data[6])
T_max = float(data[7])
connection_prob = float(data[8])
N_runs_lims = np.array(data[9])

prop_swap = 0.25

# Extract the group sizes
group_sizes = (group_props * num_nodes).astype('int')

# Ensure that group_sizes sums to num_nodes and there are no empty groups
missing_nodes = num_nodes - group_sizes.sum()
if missing_nodes != 0:
    if np.sum(np.where(group_sizes == 0)) != 0:
        group_sizes[np.where(group_sizes == 0)] = missing_nodes
    else:
        group_sizes[-1] += missing_nodes

mem_change_nodes = np.arange(int(group_sizes[0] * prop_swap))
mem_change_times = np.array([10])

# Create rho matrix
rho_matrix = np.tile(connection_prob, (num_groups, num_groups))

# Save index
if connection_prob == 0.01:
    save_idx = 0
elif connection_prob == 0.05:
    save_idx = 1
elif connection_prob == 0.1:
    save_idx = 2
elif connection_prob == 0.25:
    save_idx = 3
elif connection_prob == 0.5:
    save_idx = 4
elif connection_prob == 0.75:
    save_idx = 5

print(f"save_idx: {save_idx}")

for glob_iteration in range(N_runs_lims[0], N_runs_lims[1]):

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
                          mem_change=True,
                          mem_change_times=mem_change_times,
                          mem_change_nodes=mem_change_nodes)
        )

    # Fully-connected adjacency matrix
    adj_mat = np.ones((num_nodes, num_nodes))
    np.fill_diagonal(adj_mat, 0)

    print("...Network simulated...")

    ###
    # RUN THE INFERENCE (INFER GRAPH)
    ###

    print("...Beginning inference procedure (infer graph)...")

    # KNOWN GRAPH STRUCTURE
    VB = VariationalBayes(sampled_network=sampled_network, 
                          num_nodes=num_nodes, 
                          num_groups=num_groups, 
                          alpha_0=1, 
                          beta_0=1,
                          sigma_0=0.5,
                          eta_0=1,
                          zeta_0=1,
                          gamma_0 = np.array([0.99, 1.01]),
                          infer_graph_bool=True,
                          int_length=int_length,
                          T_max=T_max)
    VB.run_full_var_bayes(delta_pi=1,
                          delta_rho=1,
                          delta_lam=delta,
                          n_cavi=n_cavi)
    
    print("...Inference procedure completed (infer graph)...")

    tau_store = VB.tau_store
    with open(f'param_output/tau_store_inf_{pbs_index}_{glob_iteration}.pkl','wb') as f:
        pickle.dump(tau_store, f)
    f.close()

    gamma_store = VB.gamma_store
    with open(f'param_output/gamma_store_inf_{pbs_index}_{glob_iteration}.pkl','wb') as f:
        pickle.dump(gamma_store, f)
    f.close()

    alpha_store = VB.alpha_store
    with open(f'param_output/alpha_store_inf_{pbs_index}_{glob_iteration}.pkl','wb') as f:
        pickle.dump(alpha_store, f)
    f.close()

    beta_store = VB.beta_store
    with open(f'param_output/beta_store_inf_{pbs_index}_{glob_iteration}.pkl','wb') as f:
        pickle.dump(beta_store, f)
    f.close()

    ###
    # RUN THE INFERENCE (KNOWN GRAPH)
    ###

    print("...Beginning inference procedure (known graph)...")

    # KNOWN GRAPH STRUCTURE
    VB = VariationalBayes(sampled_network=sampled_network, 
                          num_nodes=num_nodes, 
                          num_groups=num_groups, 
                          alpha_0=1, 
                          beta_0=1,
                          gamma_0 = np.array([0.99, 1.01]),
                          adj_mat=adj_mat,
                          int_length=int_length,
                          T_max=T_max)
    VB.run_full_var_bayes(delta_pi=delta,
                          delta_rho=delta,
                          delta_lam=delta,
                          n_cavi=n_cavi)
    
    print("...Inference procedure completed (known graph)...")

    tau_store = VB.tau_store
    with open(f'param_output/tau/tau_store_{save_idx}_{glob_iteration}.pkl','wb') as f:
        pickle.dump(tau_store, f)
    f.close()

    gamma_store = VB.gamma_store
    with open(f'param_output/gamma/gamma_store_{save_idx}_{glob_iteration}.pkl','wb') as f:
        pickle.dump(gamma_store, f)
    f.close()

    alpha_store = VB.alpha_store
    with open(f'param_output/alpha/alpha_store_{save_idx}_{glob_iteration}.pkl','wb') as f:
        pickle.dump(alpha_store, f)
    f.close()

    beta_store = VB.beta_store
    with open(f'param_output/beta/beta_store_{save_idx}_{glob_iteration}.pkl','wb') as f:
        pickle.dump(beta_store, f)
    f.close()