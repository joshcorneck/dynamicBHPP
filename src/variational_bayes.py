from typing import Dict
import numpy as np
from scipy.special import gamma, digamma, logsumexp

class VariationalBayes:

    def __init__(self, sampled_network: Dict[int, Dict[int, list]], 
                 num_nodes: int, 
                 num_groups: int, sigma_0: np.array, eta_0: np.array, 
                 zeta_0: np.array, alpha_0: np.array, beta_0: np.array, 
                 n_0: np.array, int_length: float=1, T_max: float=100) -> None:
        
        # Network parameters
        self.num_nodes = num_nodes; self.num_groups = num_groups
        self.T_max = T_max; self.sampled_network = sampled_network

        # Sampling
        self.intervals = np.arange(int_length, T_max, int_length)
        self.int_length = int_length

        # Algorithm parameters
        self.sigma_0 = sigma_0

        self.eta_prior = eta_0

        self.zeta_prior = zeta_0

        self.alpha_prior = alpha_0

        self.beta_prior = beta_0

        self.n_prior = n_0

        self.eff_count = np.zeros((self.num_nodes, self.num_nodes))
        self.eff_obs_time = np.zeros((self.num_nodes, self.num_nodes))
        self.eff_obs_time.fill(self.int_length)
        
        self.sigma = np.zeros((self.num_nodes, self.num_nodes))

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

    def _update_q_a(self):
        """
        """
        def r_sum_term(i,j):
            r_sum = 0
            for k in range(self.num_groups):
                for m in range(self.num_groups):
                    r_sum += (
                        self.tau[i,k] * self.tau[j,m] * (
                            self.eff_count[i,j] * (
                                digamma(self.alpha[k,m])
                                -
                                np.log(self.beta[k,m])
                            )
                            -
                            self.alpha[k,m] / self.beta[k,m] 
                            +
                            digamma(self.eta[k,m]) 
                            -
                            digamma(self.eta[k,m] + self.zeta[k,m])
                        )
                    )

            return r_sum

        def s_sum_term(i,j):
            s_sum = 0
            for k in range(self.num_groups):
                for m in range(self.num_groups):
                    s_sum += (
                        self.tau[i,k] * self.tau[j,m] * (
                            digamma(self.zeta[k,m])
                            -
                            digamma(self.zeta[k,m] + self.eta[k,m])
                        )
                    )
            
            return s_sum

        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i == j:
                    pass
                else:
                    if self.eff_count[i,j] == 0:
                        log_r = r_sum_term(i,j)
                        log_s = s_sum_term(i,j)

                        self.sigma[i,j] = (
                            np.exp(log_r - logsumexp([log_r, log_s]))
                        )
                    else:
                        self.sigma[i,j] = 1

    def _update_q_z(self):
        """
        """
        # Function to compute sum over groups within exponential
        def sum_term(i_prime, j, k_prime):
            term = self.tau_prior[j,:] * (
                self.eff_count[i_prime,j] * self.sigma[i_prime,j] * (
                    digamma(self.alpha[k_prime,:]) - np.log(self.beta[k_prime,:])
                ) +
                self.eff_count[j,i_prime] * self.sigma[j,i_prime] * (
                    digamma(self.alpha[:,k_prime]) - np.log(self.beta[:,k_prime])
                ) -
                
                self.sigma[i_prime,j] * 
                self.alpha[k_prime,:] / self.beta[k_prime,:] -
                self.sigma[j,i_prime] * 
                self.alpha[:,k_prime] / self.beta[:,k_prime] + 
                
                self.sigma[i_prime,j] * (
                    digamma(self.eta[k_prime,:]) - 
                    digamma(self.eta[k_prime,:] + self.zeta[k_prime,:])
                ) +
                (1 - self.sigma[i_prime,j]) * (
                    digamma(self.zeta[k_prime,:]) - 
                    digamma(self.eta[k_prime,:] + self.zeta[k_prime,:])
                ) + 

                self.sigma[j,i_prime] * (
                    digamma(self.eta[:,k_prime]) - 
                    digamma(self.eta[:,k_prime] + self.zeta[:,k_prime])
                ) +
                (1 - self.sigma[j,i_prime]) * (
                    digamma(self.zeta[:,k_prime]) - 
                    digamma(self.eta[:,k_prime] + self.zeta[:,k_prime])
                )
            )

            return term.sum()
        
        tau_temp = np.zeros((self.num_nodes, self.num_groups))
        for i_prime in range(self.num_nodes):
            # Fix node 0 as belonging to group 1.
            # if i_prime == 0:
            #     tau_temp[i_prime,0] = -np.inf
            for k_prime in range(self.num_groups):
                temp_sum = (digamma(self.n[k_prime]) - digamma(self.n.sum()))
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
        self.n = self.delta_pi * (self.n_prior - 1) + self.tau.sum(axis=0) + 1
        
    def _update_q_rho(self):
        """
        """
        self.eta = (self.delta_rho * (self.eta_prior - 1) + 
                    self.tau.T @ self.sigma @ self.tau + 1) 
        self.zeta = (self.delta_rho * (self.zeta_prior -1 ) + 
                     self.tau.T @ (1 - self.sigma) @ self.tau + 1) 

    def _update_q_lam(self):
        """
        """
        had_prod_x_sig = self.eff_count * self.sigma
        self.alpha = (self.delta_lam * (self.alpha_prior - 1) + 
                      self.tau.T @ had_prod_x_sig @ self.tau + 1)
        self.beta = (self.delta_lam * self.beta_prior + self.tau.T @ self.sigma @ self.tau)


    def _KL_div_gammas(self, a1, a2, b1, b2):
        """
        Parameters:
            - a1, b1: the rate and scale of the approx posterior from t-1.
            - a2, b2: the rate and scale of the approx posterior from t.
        """
        return (
            a2 * np.log(b1 / b2) - np.log(gamma(a1) / gamma(a2)) +
            (a1 - a2) * digamma(a1) - (b1 - b2) * a1 / b1
        )

    def run_full_var_bayes(self, delta_pi=1, delta_rho=1, delta_lam=1, n_cavi=2):
        """
        """
        # Decay rates for the prior
        self.delta_pi = delta_pi
        self.delta_rho = delta_rho
        self.delta_lam = delta_lam

        # Empty arrays for storage
        self.tau_store = np.zeros((len(self.intervals) + 1, self.num_nodes, self.num_groups))
        self.sigma_store = np.zeros((len(self.intervals) + 1, self.num_nodes, self.num_nodes))
        self.n_store = np.zeros((len(self.intervals) + 1, self.num_groups))
        self.eta_store = np.zeros((len(self.intervals) + 1, self.num_groups, self.num_groups))
        self.zeta_store = np.zeros((len(self.intervals) + 1, self.num_groups, self.num_groups))
        self.alpha_store = np.zeros((len(self.intervals) + 1, self.num_groups, self.num_groups))
        self.beta_store = np.zeros((len(self.intervals) + 1, self.num_groups, self.num_groups))
    
        # Initialise tau, sigma, n, eta and zeta, alpha and beta
        self.tau_prior = (
            np.array([1 / self.num_groups] * (self.num_nodes * self.num_groups))
            .reshape((self.num_nodes, self.num_groups))
        )
        self.sigma_prior = self.sigma_0
        self.n = self.n_prior
        self.eta = self.eta_prior
        self.zeta = self.zeta_prior
        self.alpha = self.alpha_prior
        self.beta = self.beta_prior

        self.tau_store[0,:,:] = self.tau_prior
        self.sigma_store[0,:,:] = self.sigma_prior
        self.n_store[0,:] = self.n
        self.eta_store[0,:,:] = self.eta
        self.zeta_store[0,:,:] = self.zeta
        self.alpha_store[0,:,:] = self.alpha
        self.beta_store[0,:,:] = self.beta

        # Change-point indicators
        self.KL_div = np.zeros((len(self.intervals), self.num_groups, self.num_groups))

        for it_num, update_time in enumerate(self.intervals):
            print(f"...Iteration: {it_num + 1} of {len(self.intervals)}...")

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

            # Store estimates
            self.tau_store[it_num + 1,:,:] = self.tau
            self.sigma_store[it_num + 1,:,:] = self.sigma
            self.n_store[it_num + 1,:] = self.n
            self.eta_store[it_num + 1,:,:] = self.eta
            self.zeta_store[it_num + 1,:,:] = self.zeta
            self.alpha_store[it_num + 1,:,:] = self.alpha
            self.beta_store[it_num + 1,:,:] = self.beta

            # Update priors
            self.tau_prior = self.tau.copy()
            self.n_prior = self.n.copy()
            self.eta_prior = self.eta.copy()
            self.zeta_prior = self.zeta.copy()
            self.alpha_prior = self.alpha.copy()
            self.beta_prior = self.beta.copy()