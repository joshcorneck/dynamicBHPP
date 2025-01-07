import numpy as np
import pickle
import os
import json
import random
import argparse

from src.variational_bayes import VariationalBayes
from src.network_simulator import PoissonNetwork

"""
SIMULATION STUDY 5.
This simulation study is for continually varying rates with a single group membership
channge of 50 nodes.
"""

# Number of times we simulate.
N_runs = 50

###
# READ IN PARAMETERS
###

print("...Loading parameters...")

num_nodes = int(500)
num_groups = int(3)
n_cavi = int(3)
num_fp_its = int(3)
int_length = float(0.1)
T_max = float(5)

group_sizes = np.array([200, 200, 100])
group_sizes_prime = np.array([num_nodes])
rho_matrix = np.array([[1]]) # Fully-connected

rate_matrices = []
Delta = 0.1
for n in range(50):
    matrix = np.zeros((3, 3))
    matrix[0, 0] = 2 * np.sin(2 * np.pi / 5 * n * Delta) + 5
    matrix[0, 1] = 0.1 * np.sin(2 * np.pi / 5 * n * Delta) + 0.2
    matrix[0, 2] = 0.05 * np.sin(2 * np.pi / 5 * n * Delta) + 0.1
    matrix[1, 0] = 0.2 * np.cos(2 * np.pi / 5 * n * Delta) + 1
    matrix[1, 1] = np.cos(2 * np.pi / 5 * n * Delta) + 2
    matrix[1, 2] = 0.01 * np.sin(2 * np.pi / 5 * n * Delta) + 0.8
    matrix[2, 0] = 0.1 * np.cos(2 * np.pi / 5 * n * Delta) + 0.9
    matrix[2, 1] = 0.1 * np.sin(2 * np.pi / 5 * n * Delta) + 0.5
    matrix[2, 2] = 0.01 * np.cos(2 * np.pi / 5 * n * Delta) + 0.03
    rate_matrices.append(matrix)
    
for glob_iteration in range(N_runs):

    print(f"...GLOBAL ITERATION: {glob_iteration + 1}...")

    ###
    # SIMULATE A NETWORK
    ###
    
    # Set seed
    random.seed(int(glob_iteration))

    print("...Beginning network simulation...")

    PN = PoissonNetwork(num_nodes=num_nodes, 
                    num_groups=num_groups, 
                    num_groups_prime=1,
                    T_max=T_max,
                    lam_matrix=rate_matrices[0],
                    rho_matrix=rho_matrix)
    sampled_network, groups_in_regions = (
        PN.sample_network(group_sizes=group_sizes,
                        group_sizes_prime=group_sizes_prime,
                        rate_change=True,
                        mem_change=True,
                        rate_matrices=rate_matrices, # List of the rate matrices
                        rate_change_times=np.arange(0.1, 5, 0.1),
                        mem_change_times=np.tile([3.05], 50),
                        mem_change_nodes=np.arange(50)) # Times of the rate changes
    )

    # Fully-connected adjacency matrix
    adj_mat = np.ones((num_nodes, num_nodes))

    print("...Network simulated...")

    ###
    # RUN THE INFERENCE (INFER GRAPH)
    ###

    print("...Beginning inference procedure...")

    # KNOWN GRAPH STRUCTURE
    VB = VariationalBayes(sampled_network=sampled_network,
                        num_groups=3, num_groups_prime=1,
                        T_max=T_max, int_length=int_length,
                        adj_mat=adj_mat, 
                        num_nodes=num_nodes, alpha_0=1., beta_0=1.,
                        sigma_0=0.5, eta_0=1., zeta_0=1., nu_0=1.,
                        gamma_0=np.array([0.99, 1.02, 0.98]),
                        xi_0=np.array([0.99, 1.02, 0.98]), # Initial parameter values
                        infer_graph_bool=False,
                        infer_num_groups_bool=False) 
    VB.run_full_var_bayes(delta_pi=1,
                        delta_rho=1,
                        delta_lam=0.1,
                        delta_u=1,
                        n_cavi=n_cavi,
                        num_fp_its=num_fp_its)
    
    print("...Inference procedure completed...")

    def save_to_pickle(obj, base_path, file_name):
        with open(f'{base_path}/{file_name}', 'wb') as f:
            pickle.dump(obj, f)

    base_dir = 'analyses/simulation_studies/simulation_output/simulation_study_5'

    save_to_pickle(groups_in_regions, f'{base_dir}/true_groups', f'true_groups_{glob_iteration}.pkl')

    tau_store = VB.tau_store
    save_to_pickle(tau_store, f'{base_dir}/tau', f'tau_store_{glob_iteration}.pkl')

    group_memberships = VB.group_memberships
    save_to_pickle(group_memberships, f'{base_dir}/group_memberships', f'group_memberships_{glob_iteration}.pkl')

    group_changes_list = VB.group_changes_list
    save_to_pickle(group_changes_list, f'{base_dir}/group_changes', f'group_changes_list_{glob_iteration}.pkl')

    alpha_store = VB.alpha_store
    save_to_pickle(alpha_store, f'{base_dir}/alpha', f'alpha_store_{glob_iteration}.pkl')

    beta_store = VB.beta_store
    save_to_pickle(beta_store, f'{base_dir}/beta', f'beta_store_{glob_iteration}.pkl')
