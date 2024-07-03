import numpy as np
import pickle
import os
import json
import random
import argparse

from src.variational_bayes import VariationalBayes
from src.network_simulator import PoissonNetwork

"""
SIMULATION STUDY 4.
This simulation study is for one changing node in a graph with decreasing
connection density. We run once to infer the graph, and once with an assumed
fully-connected graph. This simulation study does 50 iterations for a given 
set of parameter combinations.
"""
# Set seed
random.seed(12)

# Number of times we simulate.
N_runs = 1

###
# READ IN PARAMETERS
###

print("...Loading parameters...")

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Run simulation study 4.')
parser.add_argument('--index', type=int, required=True, help='Index for reading relevant arguments')
args = parser.parse_args()

# Use index for reading relevant arguments
pbs_index = args.index

# Read in the relevant data from the array
with open('analyses/simulation_studies/simulation_parameters/simulation_study_4/sim_params_simulation_study_4.json', 'r') as file:
    full_data = json.load(file)
    data = full_data[pbs_index]

lam_matrix = np.array([[2.0, 1.0], [0.3, 8.0]])
num_nodes = int(500)
num_groups = int(2)
group_props = np.array([0.6, 0.4])
n_cavi = int(3)
num_fp_its = int(3)
int_length = float(0.1)
delta = float(0.1)
T_max = float(25)
connection_prob = float(data[0])

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
mem_change_times = np.tile([10], len(mem_change_nodes))

# Create rho matrix
rho_matrix = np.tile(connection_prob, (num_groups, num_groups))

for glob_iteration in range(N_runs):

    print(f"...GLOBAL ITERATION: {glob_iteration + 1}...")

    ###
    # SIMULATE A NETWORK
    ###

    print("...Beginning network simulation...")

    PN = PoissonNetwork(num_nodes=num_nodes, 
                        num_groups=num_groups,
                        num_groups_prime=1, 
                        T_max=T_max,
                        rho_matrix=rho_matrix,
                        lam_matrix=lam_matrix)

    sampled_network, groups_in_regions = (
        PN.sample_network(group_sizes=group_sizes,
                          group_sizes_prime=np.array([num_nodes]),
                          mem_change=True,
                          mem_change_times=mem_change_times,
                          mem_change_nodes=mem_change_nodes)
        )

    # Fully-connected adjacency matrix
    adj_mat = np.ones((num_nodes, num_nodes))

    print("...Network simulated...")

    ###
    # RUN THE INFERENCE (INFER GRAPH)
    ###

    print("...Beginning inference procedure (infer graph)...")

    # UNKNOWN GRAPH STRUCTURE
    VB = VariationalBayes(sampled_network=sampled_network, 
                          num_nodes=num_nodes, 
                          num_groups=num_groups,
                          num_groups_prime=1, 
                          alpha_0=1., beta_0=1., sigma_0=0.5,
                          eta_0=1., zeta_0=1.,
                          gamma_0=np.random.uniform(0.95, 1.05, size=(2,)),
                          xi_0=np.random.uniform(0.95, 1.05, size=(2,)),
                          infer_graph_bool=True,
                          int_length=int_length,
                          T_max=T_max)
    VB.run_full_var_bayes(delta_pi=1,
                          delta_rho=1,
                          delta_u=1,
                          delta_lam=delta,
                          n_cavi=n_cavi,
                          num_fp_its=num_fp_its,
                          L_js=2
                          )
    
    print("...Inference procedure completed (infer graph)...")

    def save_to_pickle(obj, base_path, file_name):
        with open(f'{base_path}/{file_name}', 'wb') as f:
            pickle.dump(obj, f)

    base_dir = 'analyses/simulation_studies/simulation_output/simulation_study_4'

    tau_store = VB.tau_store
    save_to_pickle(tau_store, f'{base_dir}/tau', f'tau_store_inf_{pbs_index}_{glob_iteration}.pkl')

    group_memberships = VB.group_memberships
    save_to_pickle(group_memberships, f'{base_dir}/group_memberships', f'group_memberships_inf_{pbs_index}_{glob_iteration}.pkl')

    group_changes_list = VB.group_changes_list
    save_to_pickle(group_changes_list, f'{base_dir}/group_changes', f'group_changes_list_inf_{pbs_index}_{glob_iteration}.pkl')

    alpha_store = VB.alpha_store
    save_to_pickle(alpha_store, f'{base_dir}/alpha', f'alpha_store_inf_{pbs_index}_{glob_iteration}.pkl')

    beta_store = VB.beta_store
    save_to_pickle(beta_store, f'{base_dir}/beta', f'beta_store_inf_{pbs_index}_{glob_iteration}.pkl')

    ###
    # RUN THE INFERENCE (KNOWN GRAPH)
    ###

    print("...Beginning inference procedure (known graph)...")

    # KNOWN GRAPH STRUCTURE
    VB = VariationalBayes(sampled_network=sampled_network, 
                          num_nodes=num_nodes, 
                          num_groups=num_groups, 
                          num_groups_prime=1,
                          alpha_0=1., beta_0=1.,
                          eta_0=1., zeta_0=1.,
                          gamma_0=np.random.uniform(0.95, 1.05, size=(2,)),
                          xi_0=np.random.uniform(0.95, 1.05, size=(2,)),
                          adj_mat=adj_mat,
                          int_length=int_length,
                          T_max=T_max)
    VB.run_full_var_bayes(delta_pi=1,
                          delta_z=1,
                          delta_u=1,
                          delta_lam=delta,
                          n_cavi=n_cavi,
                          num_fp_its=num_fp_its,
                          L_js=2)
    
    print("...Inference procedure completed (known graph)...")

    tau_store = VB.tau_store
    save_to_pickle(tau_store, f'{base_dir}/tau', f'tau_store_{pbs_index}_{glob_iteration}.pkl')

    group_memberships = VB.group_memberships
    save_to_pickle(group_memberships, f'{base_dir}/group_memberships', f'group_memberships_{pbs_index}_{glob_iteration}.pkl')

    group_changes_list = VB.group_changes_list
    save_to_pickle(group_changes_list, f'{base_dir}/group_changes', f'group_changes_list_{pbs_index}_{glob_iteration}.pkl')

    alpha_store = VB.alpha_store
    save_to_pickle(alpha_store, f'{base_dir}/alpha', f'alpha_store_{pbs_index}_{glob_iteration}.pkl')

    beta_store = VB.beta_store
    save_to_pickle(beta_store, f'{base_dir}/beta', f'beta_store_{pbs_index}_{glob_iteration}.pkl')
