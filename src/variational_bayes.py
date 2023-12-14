import numpy as np
from scipy.stats import norm
from scipy.optimize import root
from scipy.special import digamma, logsumexp

class VariationalBayes:

    def __init__(self, sampled_network, num_nodes, num_groups, T_max,
                 time_step, sigma_0, eta_0, zeta_0, alpha_0, beta_0, n_0, 
                 simple=True) -> None:
        # Network parameters
        self.N = num_nodes; self.K = num_groups
        self.T_max = T_max; self.sampled_network = sampled_network

        # Sampling 
        self.time_step = time_step
        self.intervals = np.arange(time_step, T_max, time_step)
        self.int_length = self.intervals[1]
        self.simple = simple

        # Algorithm parameters
        self.sigma_0 = sigma_0

        self.eta_prior = eta_0

        self.zeta_prior = zeta_0

        self.alpha_prior = alpha_0

        self.beta_prior = beta_0

        self.n_prior = n_0

        self.eff_count = np.zeros((self.N, self.N))
        self.eff_obs_time = np.zeros((self.N, self.N))
        
        self.sigma = np.zeros((self.N, self.N))

        self.tau = np.zeros((self.N, self.K))

    def _compute_eff_count(self, update_time):
        """
        """
        if self.simple:
            for i in range(self.N):
                for j in range(self.N):
                    if self.sampled_network[i][j] is not None:
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

    def _compute_eff_obs_time(self, update_time):
        """
        """
        if self.simple:
            self.eff_obs_time.fill(update_time)
        else:
            self.eff_obs_time.fill(self.int_length)

    def _update_q_a(self):
        """
        """
        def r_sum_term(i,j):
            r_sum = 0
            for k in range(self.K):
                for m in range(self.K):
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
            for k in range(self.K):
                for m in range(self.K):
                    s_sum += (
                        self.tau[i,k] * self.tau[j,m] * (
                            digamma(self.zeta[k,m])
                            -
                            digamma(self.zeta[k,m] + self.eta[k,m])
                        )
                    )
            
            return s_sum

        for i in range(self.N):
            for j in range(self.N):
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
        
        tau_temp = np.zeros((self.N, self.K))
        for i_prime in range(self.N):
            # Fix node 0 as belonging to group 1.
            # if i_prime == 0:
            #     tau_temp[i_prime,0] = -np.inf
            for k_prime in range(self.K):
                temp_sum = (self.delta_z[i_prime] * 
                        (digamma(self.n[k_prime]) - digamma(self.n.sum())))
                for j in range(self.N):
                    if j != i_prime:
                        temp_sum += sum_term(i_prime, j, k_prime)
                tau_temp[i_prime,k_prime] = temp_sum
            
        # self.tau_temp = tau_temp

        # Convert to exponential and normalise using logsumexp
        self.tau = np.exp(tau_temp - logsumexp(tau_temp, axis=1)[:,None])

    def _update_q_pi(self):
        """
        """
        self.n = (self.delta_pi * (self.n_prior - 1) + 
                  (np.tile(self.delta_z[:, np.newaxis], self.K) * self.tau).sum(axis=0) +
                  1)
        
    def _update_q_rho(self):
        """
        """
        self.eta = self.delta_lam * self.eta_prior + self.tau.T @ self.sigma @ self.tau 
        self.zeta = self.delta_lam * self.zeta_prior + self.tau.T @ (1 - self.sigma) @ self.tau 

    def _update_q_lam(self):
        """
        """
        had_prod_x_sig = self.eff_count * self.sigma
        self.alpha = self.delta_lam * self.alpha_prior + self.tau.T @ had_prod_x_sig @ self.tau 
        self.beta = self.delta_lam * self.beta_prior + self.tau.T @ self.sigma @ self.tau 

    def run_full_var_bayes(self, delta_pi=1, delta_lam=1, delta_z=1, n_cavi=1):
        """
        """
        # Decay rates for the prior
        # self.delta_z = np.zeros((self.N, ))
        self.delta_z = np.array([delta_z] * self.N).reshape((self.N, ))
        self.delta_pi = delta_pi
        self.delta_lam = delta_lam

        # Empty arrays for storage
        self.tau_store = np.zeros((len(self.intervals) + 1, self.N, self.K))
        self.sigma_store = np.zeros((len(self.intervals) + 1, self.N, self.N))
        self.n_store = np.zeros((len(self.intervals) + 1, self.K))
        self.eta_store = np.zeros((len(self.intervals) + 1, self.K, self.K))
        self.zeta_store = np.zeros((len(self.intervals) + 1, self.K, self.K))
        self.alpha_store = np.zeros((len(self.intervals) + 1, self.K, self.K))
        self.beta_store = np.zeros((len(self.intervals) + 1, self.K, self.K))
    
        # Initialise tau, sigma, n, eta and zeta, alpha and beta
        self.tau_prior = np.array([1 / self.K] * (self.N * self.K)).reshape((self.N, self.K))
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

        for it_num, update_time in enumerate(self.intervals):
            print(f"...Iteration: {it_num + 1} of {len(self.intervals)}...")

            self._compute_eff_count(update_time)
            self._compute_eff_obs_time(update_time)

            # Update estimates (run CAVI n_cavi times)
            cavi_count = 0
            while cavi_count < n_cavi:
                self._update_q_a()
                self._update_q_z()
                cavi_count += 1

            self._update_q_pi()
            self._update_q_rho()
            self._update_q_lam()
            

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

def adjacency_initialisation(sampled_network, num_nodes, delta):
    """
    A method to take a sample of the data and infer an initial adjacency
    matrix. Parameters:
        - init_network: samples from the network from [0,delta].
        - num_nodes: number of nodes in the network.
    """
    eff_count = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            if sampled_network[i][j] is not None:
                np_edge = np.array(
                    sampled_network[i][j]
                )
                eff_count[i,j] = (
                    len(
                        np_edge[
                            (0 <= np_edge)
                            &
                            (np_edge < delta)
                        ]
                    )
                )

    sigma_init = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                pass
            elif eff_count[i,j] == 0:
                sigma_init[i,j] = 0.5
            else:
                sigma_init[i,j] = 1
    
    return sigma_init

