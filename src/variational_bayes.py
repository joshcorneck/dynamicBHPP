import numpy as np
from scipy.stats import norm
from scipy.optimize import root
from scipy.special import digamma, logsumexp

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
        # self.alpha = np.zeros((self.K, self.K))
        self.alpha_prior = alpha_0

        # self.beta = np.zeros((self.K, self.K))
        self.beta_prior = beta_0
        
        # self.n = np.zeros((self.K, ))
        self.n_prior = n_0

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
            term = self.tau_prior[j,:] * (
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
                temp_sum = (self.delta_z[i_prime] * 
                            (digamma(self.n[k_prime]) - digamma(self.n.sum())))
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
        self.n = (self.delta_pi * (self.n_prior - 1) + 
                  (np.tile(self.delta_z[:, np.newaxis], self.K) * self.tau).sum(axis=0) +
                  1)

    def _update_q_lam(self):
        """
        """
        self.alpha = self.delta_lam * self.alpha_prior + self.tau.T @ self.eff_count @ self.tau 
        self.beta = self.delta_lam * self.beta_prior + self.tau.T @ self.eff_obs_time @ self.tau 

    def run_full_var_bayes(self, delta_z=1, n_cavi=1):
        """
        """
        # Decay rates for the prior
        # self.delta_z = np.zeros((self.N, ))
        self.delta_z = np.array([delta_z] * self.N).reshape((self.N, ))
        self.delta_pi = delta_z
        self.delta_lam = delta_z

        # Empty arrays for storage
        self.tau_store = np.zeros((len(self.intervals) + 1, self.N, self.K))
        self.n_store = np.zeros((len(self.intervals) + 1, self.K))
        self.alpha_store = np.zeros((len(self.intervals) + 1, self.K, self.K))
        self.beta_store = np.zeros((len(self.intervals) + 1, self.K, self.K))
    
        # Initialise tau, n, alpha and beta
        self.tau_prior = np.array([1 / self.K] * (self.N * self.K)).reshape((self.N, self.K))
        self.n = self.n_prior 
        self.alpha = self.alpha_prior
        self.beta = self.beta_prior

        self.tau_store[0,:,:] = self.tau
        self.n_store[0,:] = self.n
        self.alpha_store[0,:,:] = self.alpha
        self.beta_store[0,:,:] = self.beta

        for it_num, update_time in enumerate(self.intervals):
            print(f"...Iteration: {it_num + 1} of {len(self.intervals)}...")

            self._compute_eff_count(update_time)
            self._compute_eff_obs_time(update_time)

            # Update estimates (run CAVI n_cavi times)
            cavi_count = 0
            while cavi_count < n_cavi:
                self._update_q_z()
                self._update_q_pi()
                self._update_q_lam()
                cavi_count += 1

            # Store estimates
            self.tau_store[it_num + 1,:,:] = self.tau
            self.n_store[it_num + 1,:] = self.n
            self.alpha_store[it_num + 1,:,:] = self.alpha
            self.beta_store[it_num + 1,:,:] = self.beta

            # Update priors
            self.tau_prior = self.tau.copy()
            self.n_prior = self.n.copy()
            self.alpha_prior = self.alpha.copy()
            self.beta_prior = self.beta.copy()