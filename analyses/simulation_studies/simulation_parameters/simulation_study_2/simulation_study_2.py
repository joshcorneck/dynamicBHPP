import numpy as np
import pickle
import os
import json
import random
import argparse

from src.variational_bayes import VariationalBayes
from src.network_simulator import PoissonNetwork

"""
SIMULATION STUDY 2.
This simulation study is for sequential changes to the rate matrix.
"""
# Set seed
random.seed(12)

# Number of times we simulate.
N_runs = 50

###
# READ IN PARAMETERS
###

print("...Loading parameters...")

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Run simulation study 2.')
parser.add_argument('--index', type=int, required=True, help='Index for reading relevant arguments')
args = parser.parse_args()

# Use index for reading relevant arguments
pbs_index = args.index

# Read in the relevant data from the array
with open('analyses/simulation_studies/simulation_parameters/simulation_study_2/sim_params_simulation_study_2.json', 'r') as file:
    full_data = json.load(file)
    data = full_data[pbs_index]

rate_matrices = list(data[0])
rate_matrices = [np.array(x) for x in rate_matrices]
rho_matrix = np.array([[1, 1], [1, 1]])
num_nodes = int(500)
num_groups = int(2)
group_props = np.array([0.6, 0.4])
n_cavi = int(3)
num_fp_its = int(3)
int_length = float(0.1)
T_max = float(5)
delta = float(data[1])
change_steps_gap = int(data[2])

# Rate change times
first_change = 3
second_change = first_change + change_steps_gap * int_length
rate_change_times = np.array([first_change, second_change])

# Extract the group sizes
group_sizes = (group_props * num_nodes).astype('int')

# Ensure that group_sizes sums to num_nodes and there are no empty groups
missing_nodes = num_nodes - group_sizes.sum()
if missing_nodes != 0:
    if np.sum(np.where(group_sizes == 0)) != 0:
        group_sizes[np.where(group_sizes == 0)] = missing_nodes
    else:
        group_sizes[-1] += missing_nodes

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
                        lam_matrix=rate_matrices[0])

    sampled_network, groups_in_regions = (
        PN.sample_network(group_sizes=group_sizes,
                          group_sizes_prime=np.array([num_nodes]),
                          rate_change=True,
                          rate_change_times=rate_change_times,
                          rate_matrices=rate_matrices)
        )


    adj_mat = PN.adjacency_matrix

    print("...Network simulated...")

    ###
    # RUN THE INFERENCE
    ###

    print("...Beginning inference procedure...")

    # KNOWN GRAPH STRUCTURE
    VB = VariationalBayes(sampled_network=sampled_network, 
                          num_nodes=num_nodes, 
                          num_groups=num_groups,
                          num_groups_prime=1, 
                          alpha_0=1., beta_0=1.,
                          gamma_0=np.random.uniform(0.95, 1.05, size=(num_groups,)),
                          xi_0=np.random.uniform(0.95, 1.05, size=(num_groups,)),
                          adj_mat=adj_mat,
                          int_length=int_length,
                          T_max=T_max)
    VB.run_full_var_bayes(delta_pi=1,
                        delta_rho=1,
                        delta_u=1,
                        delta_lam=delta,
                        n_cavi=n_cavi,
                        num_fp_its=num_fp_its,
                        L_js=2,
                        reset_stream=False)
    
    print("...Inference procedure completed...")

    ###
    # STEP 4 - SAVE NETWORK AND INFERRED PARAMETERS
    ###

    def save_to_pickle(obj, base_path, file_name):
        with open(f'{base_path}/{file_name}', 'wb') as f:
            pickle.dump(obj, f)

    base_dir = 'analyses/simulation_studies/simulation_output/simulation_study_2'

    tau_store = VB.tau_store
    save_to_pickle(tau_store, f'{base_dir}/tau', f'tau_store_{pbs_index}_{glob_iteration}.pkl')

    group_memberships = VB.group_memberships
    save_to_pickle(group_memberships, f'{base_dir}/group_memberships', f'group_memberships_{pbs_index}_{glob_iteration}.pkl')

    group_changes_list = VB.group_changes_list
    save_to_pickle(group_changes_list, f'{base_dir}/group_changes', f'group_changes_list_{pbs_index}_{glob_iteration}.pkl')

    rate_changes_list = VB.rate_changes_list
    save_to_pickle(group_changes_list, f'{base_dir}/rate_changes', f'rate_changes_list_{pbs_index}_{glob_iteration}.pkl')

    alpha_store = VB.alpha_store
    save_to_pickle(alpha_store, f'{base_dir}/alpha', f'alpha_store_{pbs_index}_{glob_iteration}.pkl')

    beta_store = VB.beta_store
    save_to_pickle(beta_store, f'{base_dir}/beta', f'beta_store_{pbs_index}_{glob_iteration}.pkl')

