from typing import Dict
import numpy as np
from scipy.special import gammaln, digamma, logsumexp
from scipy.stats import median_abs_deviation, dirichlet, gamma, bernoulli
from scipy.stats import beta as beta_

class VariationalBayes:

    def __init__(self, sampled_network: Dict[int, Dict[int, list]]=None, 
                 num_nodes: int=None, num_groups: int=None, alpha_0: float=None, 
                 beta_0: float=None, gamma_0: np.array=None, adj_mat: np.array=None,
                 infer_graph_bool: bool=False, sigma_0: np.array=None, eta_0: float=None, 
                 zeta_0: float=None, infer_num_groups_bool: bool=False, 
                 num_var_groups: int=None, nu_0: float=None, int_length: float=1, 
                 T_max: float=50) -> None:
        
        """
        A class to run a full variational Bayesian inference procedure for a network
        point process with latent group structure. Parameters:
            - sampled_network: the network of point patterns. This should be a dictionary
                               of dictionaries, where the key of the outer dictionary is
                               the source node, with value being another dictionary with
                               key as destination node and value a list of arrival times.
            - num_nodes: number of nodes in the network.
            - num_groups: number of latent groups.
            - alpha_0, beta_0, gamma_0, sigma_0, eta_0, zeta_0: initial hyperparameter 
                                                                values.
            - adj_mat: adjacency matrix of the network. 
            - infer_graph_bool: a Boolean to indicate if the graph structure is to be 
                                inferred from the data.
            - infer_num_groups_bool: a Boolean whether we implement a nonparametric approach
                                   on the burn-in to estimate the number of groups.
            - num_var_groups: the number of groups in the variational approximation to the 
                              posterior from the nonparametric approach.
            - nu: the parameters of the gamma and beta distributions a priori.
            - int_length: the time between updates.
            - T_max: the upper bound of the full observation window (assuming starting 
                     from 0).
        """
        ## Run necessary checks on inputs
        if (adj_mat is not None) & (infer_graph_bool):
            raise ValueError("""You can't supply an adjacency matrix and set
                            infer_graph_bool = True.""")
        if (adj_mat is None) & (not infer_graph_bool):
            raise ValueError("""You must supply an adjacency matrix if
                            infer_graph_bool = False.""")
        if (adj_mat is not None) & (not infer_graph_bool):
            if infer_num_groups_bool:
                pass
            elif adj_mat.shape != (num_nodes,num_nodes):
                raise ValueError("""The shape of the supplied adjacency
                                 matrix must match the number of nodes.""")
        if infer_num_groups_bool:
           if num_var_groups is None:
                raise ValueError("""You must supply a number of groups for the 
                                 variational approximation.
                                 """)
           if infer_graph_bool:
               raise ValueError("""There is currently no functionality for both inferring
                             the graph structure and estimating the number of groups.
                             Please set one of 'est_num_groups' or 'infer_graph_bool'
                             to False.
                             """)
           if alpha_0 is None:
               raise ValueError("""You must supply a float for alpha""")
           elif not isinstance(alpha_0, float):
               alpha_0 = alpha_0.__float__()
           if beta_0 is None:
               raise ValueError("""You must supply a float for beta""")
           elif not isinstance(beta_0, float):
               beta_0 = beta_0.__float__()
           if nu_0 is None:
               raise ValueError("""You must supply a float for nu""")
           elif not isinstance(nu_0, float):
               nu_0 = nu_0.__float__()
                           
        if adj_mat is not None:
            self.adj_mat = adj_mat
        
        ## Booleans
        self.infer_graph_bool = infer_graph_bool
        self.infer_num_groups_bool = infer_num_groups_bool

        ## Network parameters
        self.num_nodes = num_nodes; self.num_groups = num_groups
        self.T_max = T_max; self.sampled_network = sampled_network
        
        ## Sampling parameters
        self.intervals = np.arange(int_length, T_max + int_length, int_length)
        self.int_length = int_length
        # Arrays to store the effective counts and observation time.
        self.eff_count = np.zeros((self.num_nodes, self.num_nodes))
        self.eff_obs_time = np.zeros((self.num_nodes, self.num_nodes))
        self.eff_obs_time.fill(self.int_length)
        self.int_length_temp = self.int_length

        ## Algorithm parameters 
        if infer_graph_bool:
            if sigma_0 is None:
                raise ValueError("Supply sigma_0.")
            if eta_0 is None:
                raise ValueError("Supply eta_0.")
            if zeta_0 is None:
                raise ValueError("Supply zeta_0.")
            self.sigma_0 = sigma_0
            self.eta_prior = np.tile([eta_0], 
                                    (num_groups, num_groups))
            self.zeta_prior = np.tile([zeta_0], 
                                    (num_groups, num_groups))
            self.sigma = np.zeros((self.num_nodes, self.num_nodes))

        if alpha_0 is None:
            raise ValueError("Supply alpha_0.")
        if beta_0 is None:
            raise ValueError("Supply beta_0.")
        
        if infer_num_groups_bool:
            if nu_0 is None:
                raise ValueError("Supply nu_0.")
            self.num_var_groups = num_var_groups
            self.alpha_prior = np.tile([alpha_0], 
                                    (num_var_groups, num_var_groups))
            self.beta_prior = np.tile([beta_0], 
                                    (num_var_groups, num_var_groups))
            self.nu_prior = np.tile([nu_0], 
                                    (num_var_groups, ))
            self.omega_prior = np.ones((num_var_groups, ))

        if not infer_num_groups_bool:
            if gamma_0 is None:
                raise ValueError("Supply gamma_0.")
            self.alpha_prior = np.tile([alpha_0], 
                                    (num_groups, num_groups))
            self.beta_prior = np.tile([beta_0], 
                                    (num_groups, num_groups))
            self.gamma_prior = gamma_0
            self.tau = np.zeros((self.num_nodes, self.num_groups))
        
    def _compute_eff_count(self, update_time: float):
        """
        A method to compute the effective count on each edge. This is simply 
        the number of observations on an edge from update_time - int_length to
        update_time. Parameters:
            - update_time: time at which we run the update.  
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
                                (update_time - self.int_length <= np_edge)
                                &
                                (np_edge < update_time)
                            ]
                        )
                    )

    def _update_q_a(self):

        def log_joint(x, a, z, rho, lambda_, pi):

            # for i in range(self.num_nodes):
            #     for j in range(self.num_nodes):
            #         for k in range(self.num_groups):
            #             for m in range(self.num_groups):
            #                 term1 += (a[i,j] * z[i,k] * z[j,m] * (
            #                     x[i,j] * np.log(lambda_[k,m]) - 
            #                     self.int_length * lambda_[k,m]
            #                     ) +
            #                     z[i,k] * z[j,m] * (a[i,j] * np.log(rho[k,m]) +
            #                             (1 - a[i,j]) * np.log(1 - rho[k,m]))
            #                 )

            term1a = np.einsum('ij,ik,jm,ij,km', a, z, z, x, np.log(lambda_))
            term1b = np.einsum('ij,ik,jm,km', a, z, z, -self.int_length * lambda_)
            term1c = np.einsum('ik,jm,ij,km', z, z, a, np.log(rho))
            term1d = np.einsum('ik,jm,ij,km', z, z,(1 - a), np.log(1 - rho))
            term1 = term1a + term1b + term1c + term1d

            # term2 = 0
            # for k in range(self.num_groups):
            #     term2 += (z[:,k] * np.log(pi[k])).sum()
            #     term2 += (self.gamma[k] - 1) * np.log(pi[k]) + gammaln(self.gamma[k])
            # term2 += gammaln(np.sum(self.gamma))
            
            term2 = np.matmul(z, np.log(pi)).sum()
            term2 += ((self.gamma - 1) * np.log(pi) + gammaln(self.gamma)).sum()
            term2 += gammaln(np.sum(self.gamma))

            # term3 = 0
            # for k in range(self.num_groups):
            #     for m in range(self.num_groups):
            #         term3 += (
            #             gammaln(self.eta[k,m] + self.zeta[k,m]) - gammaln(self.eta[k,m]) -
            #             gammaln(self.zeta[k,m]) + (self.zeta[k,m] - 1) * np.log(rho[k,m]) +
            #             (self.zeta[k,m] - 1) * np.log(1 - rho[k,m]) + 
            #             self.alpha[k,m] * np.log(self.beta[k,m]) - gammaln(self.alpha[k,m]) + 
            #             (self.alpha[k,m] - 1) * np.log(lambda_[k,m]) - self.beta[k,m] * lambda_[k,m]
                    # )
            term3 = (
                gammaln(self.eta + self.zeta) - gammaln(self.eta) -
                gammaln(self.zeta) + (self.zeta - 1) * np.log(rho) +
                (self.zeta - 1) * np.log(1 - rho) + 
                self.alpha * np.log(self.beta) - gammaln(self.alpha) + 
                (self.alpha - 1) * np.log(lambda_) - self.beta * lambda_
            ).sum()
        

            return term1 + term2 + term3
        
        def q_z(z_val, tau):
            return tau[z_val]
        
        def q_a(x_bool, a_val, sigma):
            if x_bool:
                return 1
            else:
                pmf_val = bernoulli.pmf(a_val, sigma)
                return pmf_val

        def q_pi(pi_val, gamma_):
            pdf_val = dirichlet.pdf(pi_val, gamma_)
            return pdf_val

        def q_lambda(lambda_val, alpha, beta):
            pdf_val = gamma.pdf(lambda_val, a=alpha, scale=1/beta)
            return pdf_val
        
        def q_rho(rho_val, nu, zeta):
            pdf_val = beta_.pdf(rho_val, a=nu, b=zeta)
            return pdf_val
        
        def compute_h(x_val, z_val, a_val, pi_val, lambda_val, rho_val,
                      tau, sigma, gamma_, alpha, beta, nu, zeta):
            
            h = log_joint(x_val, a_val, z_val, rho_val, lambda_val, 
                          pi_val)

            h -= np.sum(np.log(np.array([
                q_z(z_val, tau), q_a(a_val, sigma), q_pi(pi_val, gamma_),
                q_lambda(lambda_val, alpha, beta),
                q_rho(rho_val, nu, zeta)
            ])))

            return h
        
        def compute_grad_log_q(a_val, sigma):
            if (a_val == 1) & (sigma == 1):
                return 1
            elif (sigma == 0) & (a_val == 0):
                return 0
            else:
                grad_log_q = (a_val / sigma - 1) / (1 - sigma)
                return grad_log_q
        
        def approx_grab_LB(x_bool, N_samp, sigma):

            if x_bool:
                a_samp = np.ones((N_samp, ))
            else:
                bernoulli_dist = bernoulli(sigma)
                a_samp = bernoulli_dist.rvs(N_samp)
            

                


    def _update_q_a_old(self):
        """
        A method to compute the CAVI approximation for the posterior of $a$. This
        is only run if infer_graph_structure = True.
        """
        def edge_prob(i,j):
            r_sum = 0
            for k in range(self.num_groups):
                for m in range(self.num_groups):
                    r_sum += (
                        self.tau[i,k] * self.tau[j,m] * (
                            self.eff_count[i,j] * (
                                digamma(self.alpha[k,m])
                                -
                                np.log(self.beta[k,m])
                            ) + 
                            (digamma(self.eta[k,m]) 
                            -
                            digamma(self.eta[k,m] + self.zeta[k,m]))
                            -
                            (digamma(self.zeta[k,m])
                            -
                            digamma(self.zeta[k,m] + self.eta[k,m]))
                            -
                            (self.int_length * 
                            self.alpha[k,m] / self.beta[k,m])
                        )
                    )

            return r_sum

        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i == j:
                    pass
                else:
                    r_sum = edge_prob(i,j)
                    self.sigma[i,j] = 1 / (1 + np.exp(-r_sum))

    def _update_q_z(self):
        """
        A method to compute the CAVI approximation to the posterior of z.
        This is computed differently depending on the value of 
        infer_graph_structure and infer_number_groups. 
        """
        ## Functions to compute terms for fixed-point equations. This is 
        ## different for each condition.
        
        # Structured in this way to prevent multiple bool checks
        if self.infer_num_groups_bool:
            term1 = digamma(self.omega[0]) - digamma(self.nu[0] + self.omega[0])
            term1 = np.append(term1, 
                             (digamma(self.omega[1:]) - digamma(self.nu[1:] + self.omega[1:]) + 
                            np.cumsum(digamma(self.nu[:-1]) - digamma(self.nu[:-1] + self.omega[:-1]))))
            term2 = np.einsum('ij,jm,ij,km -> ik', self.adj_mat, self.tau_prior, self.eff_count, 
                              digamma(self.alpha) - np.log(self.beta))
            term3 = np.einsum('ij,jm,km -> ik', self.adj_mat, self.tau_prior, 
                              -self.int_length * self.alpha/self.beta)
            term4 = np.einsum('ji,jm,ji,mk -> ik', self.adj_mat, self.tau_prior, self.eff_count, 
                              digamma(self.alpha) - np.log(self.beta))
            term5 = np.einsum('ji,jm,mk -> ik', self.adj_mat, self.tau_prior, 
                              -self.int_length * self.alpha/self.beta)
            tau_temp = term1 + term2 + term3 + term4 + term5
        elif self.infer_graph_bool:
            term1 = digamma(gamma) - digamma(gamma.sum())
            term2a = np.einsum('jm,ij,km -> ik', self.tau_prior, -self.int_length * self.sigma, 
                               self.alpha / self.beta)
            term2b = np.einsum('jm,ji,mk -> ik', self.tau_prior, -self.int_length * self.sigma, 
                               self.alpha / self.beta)
            term3a = np.einsum('jm,ij,ij,km -> ik', self.tau_prior, self.eff_count, self.sigma, 
                            digamma(self.alpha) - np.log(self.beta))
            term3b = np.einsum('jm,ji,ji,mk -> ik', self.tau_prior, self.eff_count, self.sigma, 
                            digamma(self.alpha) - np.log(self.beta))
            term4a = np.einsum('jm,ij,km -> ik', self.tau_prior, self.sigma, 
                            digamma(self.eta) - digamma(self.eta + self.zeta))
            term4b = np.einsum('jm,ji,mk -> ik', self.tau_prior, self.sigma, 
                            digamma(self.eta) - digamma(self.eta + self.zeta))
            term5a = np.einsum('jm,ij,km -> ik', self.tau_prior, 1 - self.sigma, 
                            digamma(self.zeta) - digamma(self.eta + self.zeta))
            term5b = np.einsum('jm,ji,mk -> ik', self.tau_prior, 1 - self.sigma, 
                            digamma(self.zeta) - digamma(self.eta + self.zeta))
            tau_temp = (term1 + term2a + term2b + term3a + term3b + term4a + term4b + 
                        term5a + term5b)
        else:
            term1 = digamma(self.gamma) - digamma(self.gamma.sum())
            term2 = np.einsum('ij,jm,ij,km -> ik', self.adj_mat, self.tau_prior, self.eff_count, 
                              digamma(self.alpha) - np.log(self.beta))
            term3 = np.einsum('ij,jm,km -> ik', self.adj_mat, self.tau_prior, 
                              -self.int_length * self.alpha/self.beta)
            term4 = np.einsum('ji,jm,ji,mk -> ik', self.adj_mat, self.tau_prior, self.eff_count, 
                              digamma(self.alpha) - np.log(self.beta))
            term5 = np.einsum('ji,jm,mk -> ik', self.adj_mat, self.tau_prior, 
                              -self.int_length * self.alpha/self.beta)
            tau_temp = term1 + term2 + term3 + term4 + term5


        # Convert to exponential and normalise using logsumexp
        self.tau = np.exp(tau_temp - logsumexp(tau_temp, axis=1)[:,None])
        self.tau_prior = self.tau.copy()

    def _update_q_pi(self):
        """
        A method to compute the CAVI approximation to the posterior of pi.
        This is only run if infer_graph_structure = True.
        """
        self.gamma = (
            self.delta_pi * (self.gamma_prior - 1) + 
            self.tau.sum(axis=0) + 1
        )
        
    def _update_q_rho(self):
        """
        A method to compute the CAVI approximation to the posterior of rho.
        This is only run if infer_graph_structure = True.
        """
        self.eta = (self.delta_rho * (self.eta_prior - 1) + 
                    self.tau.T @ self.sigma @ self.tau + 1) 
        self.zeta = (self.delta_rho * (self.zeta_prior -1 ) + 
                     self.tau.T @ (1 - self.sigma) @ self.tau + 1) 

    def _update_q_lam(self):
        """
        A method to compute the CAVI approximation to the posterior of lambda.
        This is run differently depending on the value of infer_graph_bool.
        """
        if not self.infer_graph_bool:
            self.sigma = self.adj_mat
        had_prod_x_sig = self.eff_count * self.sigma
        self.alpha = (self.delta_lam * (self.alpha_prior - 1) + 
                    self.tau.T @ had_prod_x_sig @ self.tau + 1)
        self.beta = (self.delta_lam * self.beta_prior + 
                    self.int_length * self.tau.T @ self.sigma @ self.tau)
        
        # Adjust delta matrix for empty groups
        empty_groups_bool = self.tau.T @ self.sigma @ self.tau < 0.1
        self.delta_lam[empty_groups_bool] = 1
        self.delta_lam[~empty_groups_bool] = self.delta_lam_BFF
            
    def _update_q_u(self):
        """
        """
        ## Compute omega
        self.omega = self.tau.sum(axis=0) + self.omega_prior
        
        ## Calculate nu
        sum_term = np.zeros((self.num_nodes, ))
        self.nu = np.zeros((self.num_var_groups, ))
        for j in range(self.num_var_groups):
            sum_term[:] = self.tau[:,(j+1):].sum(axis=1) 
            self.nu[j] = sum_term.sum() + self.nu_prior[j]


    def _MAD_KL_outlier_detector(self, alpha, beta, max_lag, cutoff):
        """
        The function takes data contains all points up to and including
        the current update time, assuming that the burn-in points aren't 
        included. 
        """
        ## Compute the KL-divergences off all lags up to current lag
        kl_lag_list = list()
        kl_curr_datum_list = list()
        for lag in range(1, max_lag + 1):
            alpha_1 = alpha[max_lag:]; alpha_2 = alpha[(max_lag - lag):-lag]
            beta_1 = beta[max_lag:]; beta_2 = beta[(max_lag - lag):-lag]
            kl_lag = (
                alpha_2 * np.log(beta_1 / beta_2) - 
                gammaln(alpha_1) + gammaln(alpha_2) +
                (alpha_1 - alpha_2) * digamma(alpha_1) -
                (beta_1 - beta_2) * alpha_1 / beta_1
            )
            if lag == max_lag:
                kl_curr_datum_list.append(kl_lag[-1])
                kl_lag_list.append(kl_lag[:-1])
            else:
                kl_curr_datum_list.append(kl_lag[-(max_lag - lag)])
                kl_lag_list.append(kl_lag[:(max_lag - lag)])
        kl_curr_datum = np.array(kl_curr_datum_list)
        kl_lag = np.concatenate(kl_lag_list)

        ## Compute the current MAD and the deviation of the current datum
        # Current MAD (excluding current datum)
        curr_MAD = median_abs_deviation(kl_lag)
        # Deviation of current datum (for each lag up to max_lag)
        MAD_deviation_lags = []
        for i in range(max_lag):
            MAD_deviation_lags.append(
                np.abs(kl_curr_datum[i] - np.median(kl_lag)) / curr_MAD
                )
        # Flag as a change point if all lags are greater than the cutoff
        if np.all(np.array(MAD_deviation_lags) > cutoff):
            return 1
        else: 
            return 0

    def run_full_var_bayes(self, delta_pi: float=1, delta_lam: float=1, 
                           delta_rho:float=1, n_cavi: int=2, 
                           cp_burn_steps: int=10, cp_stationary_steps: int=10,
                           cp_kl_lag_steps: int=2, cp_kl_thresh: float=10, 
                           cp_rate_wait: float=0.5, ARLO_bool: bool=False):
        """
        A method to run the variational Bayesian update in its entirety.
        Parameters:
            - delta_pi, delta_rho, delta_lam: decay values.
            - n_cavi: the number of CAVI iterations at each run.
            - cp_burn_steps: the number of steps we allow up until we start to track 
                             change point metrics.
            - cp_kl_lag_steps: maximum lag (IN NUMBER OF STEPS) we consider for 
                               the KL flag.
            - cp_kl_thresh: the threshold for the MAD-KL flag.
            - cp_rate_wait: the assumed time of stationarity between changes
                            to the rate of the process.
        """
        ## Transform steps to time
        cp_burn_time = cp_burn_steps * self.int_length
        cp_kl_lag_time = cp_kl_lag_steps * self.int_length

        ## Decay rates for the prior
        self.delta_pi = delta_pi
        self.delta_rho = delta_rho
        self.delta_lam_BFF = delta_lam
        if self.infer_num_groups_bool:
            self.delta_lam = np.ones((self.num_var_groups, 
                                      self.num_var_groups)) * delta_lam
            
        else:
            self.delta_lam = np.ones((self.num_groups, self.num_groups)) * delta_lam

        ## Empty arrays for storage
        if self.infer_graph_bool:
            self.eta_store = np.zeros((len(self.intervals) + 1, 
                                       self.num_groups, 
                                       self.num_groups))
            self.zeta_store = np.zeros((len(self.intervals) + 1, 
                                        self.num_groups, 
                                        self.num_groups))
        if self.infer_num_groups_bool:
            self.omega_store = np.zeros((len(self.intervals) + 1,
                                          self.num_var_groups))
            self.nu_store = np.zeros((len(self.intervals) + 1,
                                          self.num_var_groups)) 
            self.tau_store = np.zeros((len(self.intervals) + 1, 
                                        self.num_nodes, 
                                        self.num_var_groups))
            self.alpha_store = np.zeros((len(self.intervals) + 1, 
                                        self.num_var_groups, 
                                        self.num_var_groups))
            self.beta_store = np.zeros((len(self.intervals) + 1, 
                                        self.num_var_groups, 
                                        self.num_var_groups))
        else:
            self.gamma_store = np.zeros((len(self.intervals) + 1, 
                                        self.num_groups))
        
            self.tau_store = np.zeros((len(self.intervals) + 1, 
                                        self.num_nodes, 
                                        self.num_groups))
            self.alpha_store = np.zeros((len(self.intervals) + 1, 
                                        self.num_groups, 
                                        self.num_groups))
            self.beta_store = np.zeros((len(self.intervals) + 1, 
                                        self.num_groups, 
                                        self.num_groups))

        ## Initialise relevant parameters if needed
        if self.infer_num_groups_bool:
            self.tau_prior = (
                np.array([1 / self.num_var_groups] * 
                         (self.num_nodes * self.num_var_groups))
                .reshape((self.num_nodes, self.num_var_groups))
            )
        else:
            self.tau_prior = (
                np.array([1 / self.num_groups] * (self.num_nodes * self.num_groups))
                .reshape((self.num_nodes, self.num_groups))
            )
        # Store parameter value
        self.tau_store[0,:,:] = self.tau_prior

        # Set parameter value
        self.alpha = self.alpha_prior
        self.beta = self.beta_prior

        # Store parameter value
        self.alpha_store[0,:,:] = self.alpha
        self.beta_store[0,:,:] = self.beta

        if self.infer_num_groups_bool:
            # Set parameter value
            self.nu = self.nu_prior
            self.omega = self.omega_prior

            # Store parameter value
            self.nu_store[0,:] = self.nu_prior
            self.omega_store[0,:] = self.omega_prior
        else:
            if self.infer_graph_bool:
                # Set parameter value
                self.eta = self.eta_prior
                self.zeta = self.zeta_prior

                # Store parameter value
                self.eta_store[0,:,:] = self.eta
                self.zeta_store[0,:,:] = self.zeta

            # Set parameter value
            self.gamma = self.gamma_prior

            # Store parameter value
            self.gamma_store[0,:] = self.gamma

        ## List for storing flagged changes
        self.group_changes_list = []
        self.rate_changes_list = []
        prev_change_time = 0

        ## Run the VB inference procedure            
        for it_num, update_time in enumerate(self.intervals):
            ## Run remaining runs
            print(f"...Iteration: {it_num + 1} of {len(self.intervals)}...")

            ## Reset the interval length
            self.int_length = self.int_length_temp

            ## Compute counts in the interval
            self._compute_eff_count(update_time)

            ## Update estimates (run CAVI n_cavi times)
            if self.infer_graph_bool:
                cavi_count = 0
                while cavi_count < n_cavi:
                    self._update_q_a()
                    self._update_q_z()
                    self._update_q_pi()
                    self._update_q_lam()
                    self._update_q_rho()
                    cavi_count += 1
            elif self.infer_num_groups_bool:
                cavi_count = 0
                while cavi_count < n_cavi:
                    self._update_q_z()    
                    self._update_q_u()
                    self._update_q_lam()
                    cavi_count += 1
            else:
                cavi_count = 0
                while cavi_count < n_cavi:
                    self._update_q_z()
                    self._update_q_pi()
                    self._update_q_lam()
                    cavi_count += 1

            ## Update priors
            self.tau_prior = self.tau.copy()
            self.alpha_prior = self.alpha.copy()
            self.beta_prior = self.beta.copy()
            if self.infer_num_groups_bool:
                self.nu_prior = self.nu.copy()
                self.omega_prior = self.omega.copy()
            else:
                if self.infer_graph_bool:
                    self.eta_prior = self.eta.copy()
                    self.zeta_prior = self.zeta.copy()
                self.gamma_prior = self.gamma.copy()
                
            ## Detect if any change points
            # Zero indexing
            if (it_num + 1) == cp_burn_steps:
                # Current predicted group values
                pred_groups_old = self.tau.argmax(axis=1)
            elif (it_num + 1) > cp_burn_steps:
                ## Group changes
                # Current predicted group values
                pred_groups_curr = self.tau.argmax(axis=1)
                num_group_changes = (
                    (pred_groups_curr != pred_groups_old).sum()
                )
                changing_nodes = np.argwhere(
                    pred_groups_curr != pred_groups_old
                ).flatten()
                pred_groups_old = pred_groups_curr.copy()
                self.group_changes_list.append([update_time, 
                                                num_group_changes,
                                                changing_nodes]
                                                )

                ## Rate changes
                # Only start to track changes after cp_burn_steps + max_lag runs
                if (it_num + 1) > (cp_burn_steps + cp_kl_lag_steps):
                    if self.infer_num_groups_bool:
                        num_check_groups = self.num_var_groups
                    else:
                        num_check_groups = self.num_groups
                    for j in range(num_check_groups):
                        for k in range(num_check_groups):
                            alpha_burned = (
                                self.alpha_store[cp_burn_steps:(it_num + 1), j, k]
                            )
                            beta_burned = (
                                self.beta_store[cp_burn_steps:(it_num + 1), j, k]
                            )
                            cp_flag = self._MAD_KL_outlier_detector(
                                        alpha_burned, beta_burned, 
                                        cp_kl_lag_steps, cp_kl_thresh
                                        )
                            if cp_flag:
                                curr_change_time = update_time
                                if curr_change_time - prev_change_time > cp_rate_wait:
                                    if (update_time - self.int_length > 
                                        ((cp_stationary_steps + cp_burn_steps) * self.int_length)):
                                        self.rate_changes_list.append(
                                            [update_time, update_time - self.int_length, j, k]
                                            )
                                        if ARLO_bool:
                                            return update_time - self.int_length
                                        else:
                                            prev_change_time = curr_change_time

            ## Store estimates
            self.tau_store[it_num + 1,:,:] = self.tau
            self.alpha_store[it_num + 1,:,:] = self.alpha
            self.beta_store[it_num + 1,:,:] = self.beta
            if self.infer_num_groups_bool:
                self.nu_store[it_num + 1,:] = self.nu
                self.omega_store[it_num + 1,:] = self.omega
            else:
                if self.infer_graph_bool:
                    self.eta_store[it_num + 1,:,:] = self.eta
                    self.zeta_store[it_num + 1,:,:] = self.zeta
                self.gamma_store[it_num + 1,:] = self.gamma
