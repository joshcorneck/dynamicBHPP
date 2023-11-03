#%%
import numpy as np
from scipy.stats import norm
from scipy.optimize import root
from scipy.special import digamma, logsumexp

from fully_connected_poisson import FullyConnectedPoissonNetwork


class VariationalBayes:

    def __init__(self, sampled_network, num_nodes, num_groups, T_max,
                 time_step, alpha_0, beta_0, n_0, simple=True) -> None:
        # Network parameters
        self.N = num_nodes; self.K = num_groups
        self.T_max = T_max; self.sampled_network = sampled_network

        # Sampling 
        self.time_step = time_step
        self.intervals = np.arange(time_step, T_max, time_step)
        self.int_length = self.intervals[1]
        self.simple = simple

        # Algorithm parameters
        self.alpha = np.zeros((self.K, self.K))
        self.alpha_0 = alpha_0

        self.beta = np.zeros((self.K, self.K))
        self.beta_0 = beta_0
        
        self.n = np.zeros((self.K, ))
        self.n_0 = n_0

        self.eff_count = np.zeros((self.N, self.N))
        self.eff_obs_time = np.zeros((self.N, self.N))
        
        self.tau = np.zeros((self.N, self.K))


    def _compute_eff_count(self, update_time):
        """
        """
        if self.simple:
            for i in range(self.N):
                for j in range(self.N):
                    if i == j:
                        pass
                    else:
                        np_edge = np.array(
                            self.sampled_network[i][j]
                        )
                        self.eff_count[i,j] = (
                            len(
                                np_edge[
                                    np_edge < update_time
                                ]
                            )
                        )
        else:
            for i in range(self.N):
                for j in range(self.N):
                    if i == j:
                        pass
                    else:
                        np_edge = np.array(
                            self.sampled_network[i][j]
                        )
                        self.eff_count[i,j] = (
                            len(
                                np_edge[
                                    (update_time-self.int_length <= np_edge)
                                    &
                                    (np_edge < update_time)
                                ]
                            )
                        )

    def _compute_eff_obs_time(self, update_time):
        """
        """
        if self.simple:
            self.eff_obs_time.fill(update_time)
        else:
            self.eff_obs_time.fill(self.int_length)


    def _update_q_z(self):
        """
        """
        # Function to compute sum over groups within exponential
        def sum_term(i_prime, j, k_prime):
            term = self.tau[j,:] * (
                self.eff_count[i_prime,j] * (
                    digamma(self.alpha[k_prime,:]) - np.log(self.beta[k_prime,:])
                ) +
                self.eff_count[j,i_prime] * (
                    digamma(self.alpha[:,k_prime]) - np.log(self.beta[:,k_prime])
                ) -
                self.eff_obs_time[i_prime,j] * self.alpha[k_prime,:] / self.beta[k_prime,:] -
                self.eff_obs_time[j,i_prime] * self.alpha[:,k_prime] / self.beta[:,k_prime]
            )

            return term.sum()
        
        tau_temp = np.zeros((self.N, self.K))
        for i_prime in range(self.N):
            for k_prime in range(self.K):
                temp_sum = digamma(self.n[k_prime]) - digamma(self.N)
                for j in range(self.N):
                    if j != i_prime:
                        temp_sum += sum_term(i_prime, j, k_prime)
                tau_temp[i_prime,k_prime] = temp_sum
                
        self.tau_temp = tau_temp
        # Convert to exponential and normalise using logsumexp
        self.tau = np.exp(tau_temp - logsumexp(tau_temp, axis=1)[:,None])

    def _update_q_pi(self):
        """
        """
        self.n = self.tau.sum(axis=0) + self.decay_pi * self.n

    def _update_q_lam(self):
        """
        """
        self.alpha = self.tau.T @ self.eff_count @ self.tau + self.decay_lam * self.alpha
        self.beta = self.tau.T @ self.eff_obs_time @ self.tau + self.decay_lam * self.beta

    def run_full_var_bayes(self, decay_lam, decay_pi = 1):
        """
        """
        self.decay_pi = decay_pi
        self.decay_lam = decay_lam

        # Empty arrays for storage
        self.tau_store = np.zeros((len(self.intervals), self.N, self.K))
        self.n_store = np.zeros((len(self.intervals), self.K))
        self.alpha_store = np.zeros((len(self.intervals), self.K, self.K))
        self.beta_store = np.zeros((len(self.intervals), self.K, self.K))
    
        # Initialise tau, n, alpha and beta
        self.tau = np.array([1 / self.K] * (self.N * self.K)).reshape((self.N, self.K))
        self.n = self.n_0 
        self.alpha = self.alpha_0
        self.beta = self.beta_0

        for it_num, update_time in enumerate(self.intervals):
            print(f"...Iteration: {it_num + 1} of {len(self.intervals)}...")

            self._compute_eff_count(update_time)
            self._compute_eff_obs_time(update_time)

            self._update_q_z()
            self._update_q_pi()
            self._update_q_lam()

            self.tau_store[it_num,:,:] = self.tau
            self.n_store[it_num,:] = self.n
            self.alpha_store[it_num,:,:] = self.alpha
            self.beta_store[it_num,:,:] = self.beta

# #%%
# num_nodes = 60; num_groups = 2; T_max = 10; int_length=0.1
# FCP = FullyConnectedPoissonNetwork(num_nodes, num_groups, T_max, np.array([[3, 1], [2, 5]]))
# # sampled_network, groups_in_regions = FCP.sample_network(change_point=True, num_cps=1)
# sampled_network, groups_in_regions = (
#     FCP.sample_network(change_point=True, random_groups=False,
#                        group_sizes=np.array([10,50]), num_cps=1)
# )
# change_node = FCP.changing_node
# change_time = FCP.change_point_time

# VB = VariationalBayes(sampled_network, num_nodes, num_groups, T_max, 
#                       int_length, 
#                       np.random.uniform(1,2,(num_groups,num_groups)), 
#                       np.ones((num_groups,num_groups)),
#                       np.array([(1/num_groups)*num_nodes] * num_groups),
#                       simple=False)
# VB.run_full_var_bayes()
# #%%
# import matplotlib.pyplot as plt
# #%%
# plt.plot(np.arange(int(T_max/int_length) - 1), VB.tau_store[:,change_node,1])
#  # %%
# VB.tau_store[50,:,:].mean(axis=0)
# # %%

# for j in range(3):
#     plt.plot(np.arange(int(T_max/int_length) - 1),
#           [VB.alpha_store[i,j,:] / VB.beta_store[i,j,:] for i in np.arange(int(T_max/int_length) - 1)])
# # %%
