from typing import Dict
import numpy as np
from scipy.special import gammaln, digamma, logsumexp

class VariationalBayes:

    def __init__(self, sampled_network: Dict[int, Dict[int, list]], 
                 num_nodes: int, num_groups: int, alpha_0: np.array, 
                 beta_0: np.array, gamma_0: np.array, int_length: float=1, 
                 T_max: float=100) -> None:
        
        # Network parameters
        self.num_nodes = num_nodes; self.num_groups = num_groups
        self.T_max = T_max; self.sampled_network = sampled_network

        # Sampling
        self.intervals = np.arange(int_length, T_max, int_length)
        self.int_length = int_length

        # Algorithm parameters
        self.alpha_prior = alpha_0
        self.beta_prior = beta_0
        self.gamma_prior = gamma_0
        self.eff_count = np.zeros((self.num_nodes, self.num_nodes))
        self.eff_obs_time = np.zeros((self.num_nodes, self.num_nodes))
        self.eff_obs_time.fill(self.int_length)
        self.tau = np.zeros((self.num_nodes, self.num_groups))

    def _compute_eff_count(self, update_time: float):
        """
        """
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if self.sampled_network[i][j] is not None:
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

    def _update_q_z(self):
        """
        """
        # Function to compute sum over groups within exponential
        def sum_term(i_prime, j, k_prime):
            term = self.tau_prior[j,:] * (
                self.eff_count[i_prime,j] * (
                    digamma(self.alpha[k_prime,:]) - np.log(self.beta[k_prime,:])
                ) +
                self.eff_count[j,i_prime] * (
                    digamma(self.alpha[:,k_prime]) - np.log(self.beta[:,k_prime])
                ) -

                self.int_length * (
                self.alpha[k_prime,:] / self.beta[k_prime,:] +
                self.alpha[:,k_prime] / self.beta[:,k_prime]
                )
            )

            return term.sum()
        
        tau_temp = np.zeros((self.num_nodes, self.num_groups))
        for i_prime in range(self.num_nodes):
            for k_prime in range(self.num_groups):
                temp_sum = self.delta_z * (
                    digamma(self.gamma[k_prime]) - digamma(self.gamma.sum())
                    )
                for j in range(self.num_nodes):
                    if j != i_prime:
                        temp_sum += sum_term(i_prime, j, k_prime)
                tau_temp[i_prime,k_prime] = temp_sum
            
        # self.tau_temp = tau_temp

        # Convert to exponential and normalise using logsumexp
        self.tau = np.exp(tau_temp - logsumexp(tau_temp, axis=1)[:,None])

    def _update_q_pi(self):
        """
        """
        self.gamma = (self.delta_pi * (self.gamma_prior - 1) + 
                      self.delta_z * self.tau.sum(axis=0) + 1
        )

    def _update_q_lam(self):
        """
        """
        self.alpha = (self.delta_lam * (self.alpha_prior - 1) + 
                      self.tau.T @ self.eff_count @ self.tau + 1)
        self.beta = (self.delta_lam * self.beta_prior + self.tau.T @ self.tau)


    def _KL_div_gammas(self, a1, a2, b1, b2):
        """
        Parameters:
            - a1, b1: the rate and scale of the approx posterior from t-1.
            - a2, b2: the rate and scale of the approx posterior from t.
        """
        return (
            a2 * np.log(b1 / b2) - gammaln(a1) + gammaln(a2) +
            (a1 - a2) * digamma(a1) - (b1 - b2) * a1 / b1
        )

    def run_full_var_bayes(self, delta_z=1, delta_pi=1, delta_rho=1, delta_lam=1, n_cavi=2):
        """
        """
        # Decay rates for the prior
        self.delta_z = delta_z
        self.delta_pi = delta_pi
        self.delta_rho = delta_rho
        self.delta_lam = delta_lam

        # Empty arrays for storage
        self.tau_store = np.zeros((len(self.intervals) + 1, self.num_nodes, self.num_groups))
        self.gamma_store = np.zeros((len(self.intervals) + 1, self.num_groups))
        self.alpha_store = np.zeros((len(self.intervals) + 1, self.num_groups, self.num_groups))
        self.beta_store = np.zeros((len(self.intervals) + 1, self.num_groups, self.num_groups))
    
        # Initialise tau, gamma, alpha and beta
        self.tau_prior = (
            np.array([1 / self.num_groups] * (self.num_nodes * self.num_groups))
            .reshape((self.num_nodes, self.num_groups))
        )
        self.gamma = self.gamma_prior
        self.alpha = self.alpha_prior
        self.beta = self.beta_prior

        self.tau_store[0,:,:] = self.tau_prior
        self.gamma_store[0,:] = self.gamma
        self.alpha_store[0,:,:] = self.alpha
        self.beta_store[0,:,:] = self.beta

        # Change-point indicators
        self.KL_div = np.zeros((len(self.intervals), self.num_groups, self.num_groups))

        for it_num, update_time in enumerate(self.intervals):
            print(f"...Iteration: {it_num + 1} of {len(self.intervals)}...", end='\r')

            self._compute_eff_count(update_time)

            # Update estimates (run CAVI n_cavi times)
            cavi_count = 0
            while cavi_count < n_cavi:
                self._update_q_a()
                self._update_q_z()
                cavi_count += 1

            self._update_q_pi()
            self._update_q_rho()
            self._update_q_lam()
            
            # Compute the KL-divergence for each rate parameter
            for i in range(self.num_groups):
                for j in range(self.num_groups):
                    self.KL_div[it_num,i,j] = self._KL_div_gammas(
                                                    self.alpha_store[it_num,i,j],
                                                    self.alpha[i,j],
                                                    self.beta_store[it_num,i,j],
                                                    self.beta[i,j]
                                                    )
            
            # Flag if CP has occurred
            

            # Store estimates
            self.tau_store[it_num + 1,:,:] = self.tau
            self.gamma_store[it_num + 1,:] = self.gamma
            self.alpha_store[it_num + 1,:,:] = self.alpha
            self.beta_store[it_num + 1,:,:] = self.beta

            # Update priors
            self.tau_prior = self.tau.copy()
            self.gamma_prior = self.gamma.copy()
            self.alpha_prior = self.alpha.copy()
            self.beta_prior = self.beta.copy()