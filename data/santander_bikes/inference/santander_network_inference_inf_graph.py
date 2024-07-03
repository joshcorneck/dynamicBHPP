import numpy as np
import pickle
import os
import random

from src.variational_bayes import VariationalBayes

random.seed(12)

###
# STEP 1 - SET PARAMETERS AND LOAD DATA
###

# Set parameters
num_nodes = 791
num_groups = 6
num_groups_prime = 5
n_cavi = 3
num_fp_its = 3
delta_lam = 0.1
L_js = 1.55
adj_mat = np.ones((num_nodes, num_nodes))
T_max = 100
int_length = 1

# Load the santander_network
with open('data/santander_bikes/get_and_process_data/santander_network.pkl', 'rb') as file:
    santander_network = pickle.load(file)

###
# STEP 2 - RUN THE INFERENCE (FIXED NUMBER OF GROUPS)
###

# Run the VB algorithm
VB = VariationalBayes(sampled_network=santander_network,
                      num_groups=num_groups,
                      num_groups_prime=num_groups_prime, 
                      num_nodes=num_nodes, alpha_0=1., beta_0=1.,
                         sigma_0=0.5, eta_0=1., zeta_0=1., 
                      gamma_0=np.random.uniform(0.95,1.05,size=(num_groups, )),
                      xi_0=np.random.uniform(0.95,0.105,size=(num_groups_prime, )),
                      infer_graph_bool=True,
                      infer_num_groups_bool=False,
                      nu_0=1,
                      T_max=T_max, int_length=int_length)
VB.run_full_var_bayes(delta_pi=1,
                      delta_rho=1,
                      delta_lam=delta_lam,
                      n_cavi=n_cavi,
                      num_fp_its=num_fp_its,
                      B1=15, B2=10,
                      L_js=L_js)

###
# STEP 4 - SAVE INFERRED PARAMETERS
###

def save_to_pickle(obj, base_path, file_name):
        with open(f'{base_path}/{file_name}', 'wb') as f:
            pickle.dump(obj, f)

base_dir = 'data/santander_bikes/output'

tau_store = VB.tau_store
save_to_pickle(tau_store, f'{base_dir}/tau', f'tau_store_inf_graph.pkl')

group_memberships = VB.group_memberships
save_to_pickle(group_memberships, f'{base_dir}/group_memberships', f'group_memberships_inf_graph.pkl')

alpha_store = VB.alpha_store
save_to_pickle(alpha_store, f'{base_dir}/alpha', f'alpha_store_inf_graph.pkl')

beta_store = VB.beta_store
save_to_pickle(beta_store, f'{base_dir}/beta', f'beta_store_inf_graph.pkl')
