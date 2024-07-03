import numpy as np
import pickle
import os
import json
import random
import argparse

from src.variational_bayes import VariationalBayes
from src.network_simulator import PoissonNetwork

"""
SIMULATION STUDY 3.
This simulation study is for merging group 2 into group 1, and then
creating a new group 2.
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
parser = argparse.ArgumentParser(description='Run simulation study 3.')
parser.add_argument('--index', type=int, required=True, help='Index for reading relevant arguments')
args = parser.parse_args()

# Use index for reading relevant arguments
pbs_index = args.index

# Read in the relevant data from the array
with open('analyses/simulation_studies/simulation_parameters/simulation_study_3/sim_params_simulation_study_3.json', 'r') as file:
    full_data = json.load(file)
    data = full_data[pbs_index]

lam_matrix = np.array([[2.0, 1.0], [0.3, 8.0]])
rho_matrix = np.array([[1, 1], [1, 1]])
num_nodes = int(500)
num_groups = int(2)
group_sizes = [np.array(x) for x in data[0]]
n_cavi = int(3)
num_fp_its = int(3)
int_length = float(0.1)
T_max = float(5)
delta = float(data[1])
prop_new_group_1 = float(data[2])

group_num_change_times = [2.5, 3.5]

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
                        num_groups_prime=1, 
                        T_max=T_max, 
                        rho_matrix=rho_matrix, 
                        lam_matrix=lam_matrix)

    sampled_network, groups_in_regions = (
        PN.sample_network(group_sizes=group_sizes,
                          group_sizes_prime=np.array([num_nodes]),
                          group_num_change=True,
                          group_num_change_times=group_num_change_times  
                          )
        )

    adj_mat = PN.adjacency_matrix

    def save_to_pickle(obj, base_path, file_name):
        with open(f'{base_path}/{file_name}', 'wb') as f:
            pickle.dump(obj, f)

    base_dir = 'analyses/simulation_studies/simulation_output/simulation_study_3'

    save_to_pickle(groups_in_regions, f'{base_dir}/true_changes', f'groups_in_regions_{pbs_index}_{glob_iteration}.pkl')

    print("...Network simulated...")

    ###
    # RUN THE INFERENCE
    ###

    print("...Beginning inference procedure...")

    # KNOWN GRAPH STRUCTURE
    VB = VariationalBayes(sampled_network=sampled_network, 
                          num_nodes=num_nodes, 
                          num_groups_prime=1,
                          alpha_0=1., beta_0=1., nu_0=1.,
                          zeta_0=1., eta_0=1.,
                          gamma_0=np.random.uniform(0.95, 1.05, size=(2,)),
                          xi_0=np.random.uniform(0.95, 1.05, size=(2,)),
                          infer_num_groups_bool=True,
                          num_var_groups=2,
                          adj_mat=adj_mat,
                          int_length=int_length,
                          T_max=T_max
                          )
    VB.run_full_var_bayes(delta_pi=1,
                          delta_rho=1,
                          delta_lam=delta,
                          delta_u=delta,
                          n_cavi=n_cavi,
                          num_fp_its=num_fp_its,
                          L_js=2
                          )
    
    print("...Inference procedure completed...")

    ###
    # STEP 4 - SAVE NETWORK AND INFERRED PARAMETERS
    ###

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
