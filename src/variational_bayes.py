from typing import Dict
import numpy as np
from scipy.special import gammaln, digamma, logsumexp
from scipy.stats import median_abs_deviation, dirichlet, bernoulli
from scipy.stats import beta 
from scipy.stats import gamma 

class VariationalBayes:

    def __init__(self, sampled_network: Dict[int, Dict[int, list]]=None,
                 num_nodes: int=None, num_groups: int=None, alpha_0: float=None, 
                 beta_0: float=None, gamma_0: np.array=None, adj_mat: np.array=None,
                 infer_graph_bool: bool=False, num_groups_prime: int=None, sigma_0: np.array=None, 
                 eta_0: float=None, zeta_0: float=None, xi_0: float=None, 
                 infer_num_groups_bool: bool=False, num_var_groups: int=None, nu_0: float=None, 
                 int_length: float=1, T_max: float=50) -> None:
        """
        A class to run a full variational Bayesian inference procedure for a network
        point process with latent group structure. Parameters:
            - sampled_network: the network of point patterns. This should be a dictionary
                               of dictionaries, where the key of the outer dictionary is
                               the source node, with value being another dictionary with
                               key as destination node and value a list of arrival times.
            - num_nodes: number of nodes in the network.
            - num_groups: number of latent groups.
            - alpha_0, beta_0, gamma_0, xi_0, sigma_0, eta_0, zeta_0: initial hyperparameter 
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
        self.num_nodes = num_nodes; self.num_groups = num_groups; 
        self.num_groups_prime = num_groups_prime; self.T_max = T_max
        
        self.sampled_network = sampled_network

        ## Sampling parameters
        self.intervals = np.arange(int_length, T_max + int_length, int_length)
        self.int_length = int_length
        # Arrays to store the effective counts and observation time.
        self.eff_count = np.zeros((self.num_nodes, self.num_nodes))
        self.full_count = np.zeros((self.num_nodes, self.num_nodes))
        self.eff_obs_time = np.zeros((self.num_nodes, self.num_nodes))
        self.eff_obs_time.fill(self.int_length)

        ## Algorithm parameters 
        if infer_graph_bool:
            if sigma_0 is None:
                raise ValueError("Supply sigma_0.")
            if eta_0 is None:
                raise ValueError("Supply eta_0.")
            if zeta_0 is None:
                raise ValueError("Supply zeta_0.")
            self.eta_prior = np.tile([eta_0], 
                                    (num_groups_prime, num_groups_prime))
            self.zeta_prior = np.tile([zeta_0], 
                                    (num_groups_prime, num_groups_prime))
            # self.eta_prior = eta_0
            # self.zeta_prior = zeta_0
            self.sigma = np.ones((self.num_nodes, self.num_nodes)) * sigma_0

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
            self.xi_prior = xi_0
            self.tau = np.zeros((self.num_nodes, self.num_groups))
            self.tau_prime = np.zeros((self.num_nodes, self.num_groups_prime))
        
    def _compute_eff_count(self, update_time: float):
        """
        A method to compute the effective count on each edge. This is simply 
        the number of observations on an edge from update_time - int_length to
        update_time. Parameters:
            - update_time: time at which we run the update.  
        """
        self.update_time = update_time
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
        self.full_count = self.full_count + self.eff_count

    def _update_q_a(self):
        """
        Function to compute the exact posterior for a.
        """
        tau_max_idx = self.tau.argmax(axis=1)
        tau_prime_max_idx = self.tau_prime.argmax(axis=1)

        z_val = np.zeros((self.num_nodes, self.num_groups))
        z_prime_val = np.zeros((self.num_nodes, self.num_groups_prime))
        
        z_val[np.arange(self.num_nodes), tau_max_idx] = 1
        z_prime_val[np.arange(self.num_nodes), tau_prime_max_idx] = 1

        rho_val = self.eta / (self.eta + self.zeta)
        lambda_val = self.alpha / self.beta

        # Create matrix of relevant lambda and rho values
        lam_big = np.zeros((self.num_nodes, self.num_nodes))
        rho_big = np.zeros((self.num_nodes, self.num_nodes))
        rows, cols = np.meshgrid(np.arange(self.num_nodes), 
                                 np.arange(self.num_nodes), 
                                 indexing='ij')
        
        lam_big = (lambda_val[tau_max_idx[rows.ravel()],
                                  tau_max_idx[cols.ravel()]].
                                  reshape(self.num_nodes, self.num_nodes))
        self.lam_big_store += lam_big

        rho_big = (rho_val[tau_prime_max_idx[rows.ravel()],
                                  tau_prime_max_idx[cols.ravel()]].
                                  reshape(self.num_nodes, self.num_nodes))  
        self.sigma = (rho_big * np.exp(-self.int_length * self.lam_big_store) / 
                      (1 - rho_big + rho_big * np.exp(-self.int_length * self.lam_big_store)))

        # True if there is at least one observation
        self.sigma[self.full_count > 0] = 1

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
            digamma_diff = (self.delta_z * (
                digamma(self.omega) - digamma(self.omega + self.nu) +
                (np.cumsum(digamma(self.nu) - digamma(self.omega + self.nu)) -
                 digamma(self.nu) - digamma(self.omega + self.nu))
            ))
            digamma_diff_alpha_log_beta = np.diag(digamma(self.alpha) - np.log(self.beta))
            alpha_beta_ratio = self.int_length * np.diag(self.alpha / self.beta)
            for i in range(self.num_nodes):
                tau_temp_i = np.zeros((self.num_var_groups, ))
                for k in range(self.num_var_groups):
                    tau_temp_i[k] = (digamma_diff[k] + 
                                   self.eff_count[i,i] * digamma_diff_alpha_log_beta[k] - 
                                   alpha_beta_ratio[k])
                    for m in range(self.num_var_groups):
                        if m == k:
                            mask = np.ones((self.num_nodes,), dtype=bool)
                            mask[i] = False # Exclude i == j and k == m
                            tau_temp_i[k] += (
                                (self.tau_prior[mask,m] * self.adj_mat[i,mask] * 
                                (self.eff_count[i,mask] * (
                                digamma(self.alpha[k,m]) - np.log(self.beta[k,m])) - 
                                self.int_length * self.alpha[k,m] / self.beta[k,m])).sum() +
                                (self.tau_prior[mask,m] * self.adj_mat[mask,i] *
                                (self.eff_count[mask,i] * (
                                digamma(self.alpha[m,k]) - np.log(self.beta[m,k])) - 
                                self.int_length * self.alpha[m,k] / self.beta[m,k])).sum()
                                )
                        else:
                            tau_temp_i[k] += (
                                (self.tau_prior[:,m] * self.adj_mat[i,:] * 
                                (self.eff_count[i,:] * (
                                digamma(self.alpha[k,m]) - np.log(self.beta[k,m])) - 
                                self.int_length * self.alpha[k,m] / self.beta[k,m])).sum() +
                                (self.tau_prior[:,m] * self.adj_mat[:,i] *
                                (self.eff_count[:,i] * (
                                digamma(self.alpha[m,k]) - np.log(self.beta[m,k])) - 
                                self.int_length * self.alpha[m,k] / self.beta[m,k])).sum()
                                )
                # Renormalise the row
                tau_temp_i_norm = np.exp(tau_temp_i - logsumexp(tau_temp_i))
                self.tau_prior[i,:] = tau_temp_i_norm

        elif self.infer_graph_bool:
            digamma_diff = self.delta_z * (digamma(self.gamma) - digamma(self.gamma.sum()))
            digamma_diff_alpha_log_beta = np.diag(digamma(self.alpha) - np.log(self.beta))
            alpha_beta_ratio = self.int_length * np.diag(self.alpha / self.beta)
            for i in range(self.num_nodes):
                tau_temp_i = np.zeros((self.num_groups, ))
                for k in range(self.num_groups):
                    tau_temp_i[k] = (digamma_diff[k] + self.sigma[i,i] * ( 
                                   self.eff_count[i,i] * digamma_diff_alpha_log_beta[k] - 
                                   alpha_beta_ratio[k])
                    )
                    for m in range(self.num_groups):
                        if m == k:
                            mask = np.ones((self.num_nodes,), dtype=bool)
                            mask[i] = False # Exclude i == j and k == m
                            tau_temp_i[k] += (
                                (self.tau_prior[mask,m] * 
                                 (self.sigma[i,mask] * self.eff_count[i,mask] * (
                                     digamma(self.alpha[k,m]) - np.log(self.beta[k,m])) + 
                                 self.sigma[mask,i] * self.eff_count[mask,i] * (
                                     digamma(self.alpha[m,k]) - np.log(self.beta[m,k])) -
                                 self.int_length * (
                                     self.sigma[i,mask] * self.alpha[k,m] / self.beta[k,m] +
                                     self.sigma[mask,i] * self.alpha[m,k] / self.beta[m,k])
                                 )
                                ).sum()
                            )
                        else:
                            tau_temp_i[k] += (
                                (self.tau_prior[:,m] * 
                                 (self.sigma[i,:] * self.eff_count[i,:] * (
                                     digamma(self.alpha[k,m]) - np.log(self.beta[k,m])) + 
                                 self.sigma[:,i] * self.eff_count[:,i] * (
                                     digamma(self.alpha[m,k]) - np.log(self.beta[m,k])) -
                                 self.int_length * (
                                     self.sigma[i,:] * self.alpha[k,m] / self.beta[k,m] +
                                     self.sigma[:,i] * self.alpha[m,k] / self.beta[m,k])
                                 )
                                ).sum()
                            )
                # Renormalise the row
                tau_temp_i_norm = np.exp(tau_temp_i - logsumexp(tau_temp_i))
                self.tau_prior[i,:] = tau_temp_i_norm
        else:
            digamma_diff = self.delta_z * (digamma(self.gamma) - digamma(self.gamma.sum()))
            digamma_diff_alpha_log_beta = np.diag(digamma(self.alpha) - np.log(self.beta))
            alpha_beta_ratio = self.int_length * np.diag(self.alpha / self.beta)
            for i in range(self.num_nodes):
                tau_temp_i = np.zeros((self.num_groups, ))
                for k in range(self.num_groups):
                    tau_temp_i[k] = (digamma_diff[k] + 
                                   self.eff_count[i,i] * digamma_diff_alpha_log_beta[k] - 
                                   alpha_beta_ratio[k])
                    for m in range(self.num_groups):
                        if m == k:
                            mask = np.ones((self.num_nodes,), dtype=bool)
                            mask[i] = False # Exclude i == j and k == m
                            tau_temp_i[k] += (
                                (self.tau_prior[mask,m] * self.adj_mat[i,mask] * 
                                (self.eff_count[i,mask] * (
                                digamma(self.alpha[k,m]) - np.log(self.beta[k,m])) - 
                                self.int_length * self.alpha[k,m] / self.beta[k,m])).sum() +
                                (self.tau_prior[mask,m] * self.adj_mat[mask,i] *
                                (self.eff_count[mask,i] * (
                                digamma(self.alpha[m,k]) - np.log(self.beta[m,k])) - 
                                self.int_length * self.alpha[m,k] / self.beta[m,k])).sum()
                                )
                        else:
                            tau_temp_i[k] += (
                                (self.tau_prior[:,m] * self.adj_mat[i,:] * 
                                (self.eff_count[i,:] * (
                                digamma(self.alpha[k,m]) - np.log(self.beta[k,m])) - 
                                self.int_length * self.alpha[k,m] / self.beta[k,m])).sum() +
                                (self.tau_prior[:,m] * self.adj_mat[:,i] *
                                (self.eff_count[:,i] * (
                                digamma(self.alpha[m,k]) - np.log(self.beta[m,k])) - 
                                self.int_length * self.alpha[m,k] / self.beta[m,k])).sum()
                                )
                # Renormalise the row
                tau_temp_i_norm = np.exp(tau_temp_i - logsumexp(tau_temp_i))
                self.tau_prior[i,:] = tau_temp_i_norm

    def _update_q_z_prime(self):
        """
        A method to compute the CAVI approximation to the posterior of z'.
        """
        digamma_diff = digamma(self.xi) - digamma(self.xi.sum())
        digamma_diff_1 = np.diag(digamma(self.eta) - digamma(self.eta + self.zeta))
        digamma_diff_2 = np.diag(digamma(self.zeta) - digamma(self.eta + self.zeta))
        for i in range(self.num_nodes):
            tau_temp_i = np.zeros((self.num_groups_prime, ))
            for k in range(self.num_groups_prime):
                tau_temp_i[k] = (digamma_diff[k] + 
                                self.sigma[i,i] * digamma_diff_1[k] + 
                                (1 - self.sigma[i,i]) * digamma_diff_2[k])
                for m in range(self.num_groups_prime):
                    if m == k:
                        mask = np.ones((self.num_nodes,), dtype=bool)
                        mask[i] = False # Exclude i == j and k == m
                        tau_temp_i[k] += (
                            (self.tau_prime_prior[mask,m] * (
                                self.sigma[i,mask] * (
                                    digamma(self.eta[k,m]) - digamma(self.eta[k,m] + self.zeta[k,m])
                                ) +
                                (1 - self.sigma[i,mask]) * (
                                    digamma(self.zeta[k,m]) - digamma(self.eta[k,m] + self.zeta[k,m])
                                ) +
                                self.sigma[mask,i] * (
                                    digamma(self.eta[m,k]) - digamma(self.eta[m,k] + self.zeta[m,k])
                                ) +
                                (1 - self.sigma[mask,i]) * (
                                    digamma(self.zeta[m,k]) - digamma(self.eta[m,k] + self.zeta[m,k])
                                )
                            ) 
                            ).sum()
                        )
                    else:
                        tau_temp_i[k] += (
                            (self.tau_prime_prior[:,m] * (
                                self.sigma[i,:] * (
                                    digamma(self.eta[k,m]) - digamma(self.eta[k,m] + self.zeta[k,m])
                                ) +
                                (1 - self.sigma[i,:]) * (
                                    digamma(self.zeta[k,m]) - digamma(self.eta[k,m] + self.zeta[k,m])
                                ) +
                                self.sigma[:,i] * (
                                    digamma(self.eta[m,k]) - digamma(self.eta[m,k] + self.zeta[m,k])
                                ) +
                                (1 - self.sigma[:,i]) * (
                                    digamma(self.zeta[m,k]) - digamma(self.eta[m,k] + self.zeta[m,k])
                                )
                            ) 
                            ).sum()
                        )
            # Renormalise the row
            tau_temp_i_norm = np.exp(tau_temp_i - logsumexp(tau_temp_i))
            self.tau_prime_prior[i,:] = tau_temp_i_norm

    def _update_q_pi(self):
        """
        A method to compute the CAVI approximation to the posterior of pi.
        This is only run if infer_graph_structure = True.
        """
        self.gamma = (
            self.delta_pi * (self.gamma_prior - 1) + 
            self.tau.sum(axis=0) + 1
        )
    
    def _update_q_mu(self):
        """
        A method to compute the CAVI approximation to the posterior of pi.
        This is only run if infer_graph_structure = True.
        """
        self.xi = (
            self.xi_prior + self.tau_prime.sum(axis=0) 
        )

    def _update_q_rho(self):
        """
        A method to compute the CAVI approximation to the posterior of rho.
        This is only run if infer_graph_structure = True.
        """
        self.eta = (self.delta_rho * (self.eta_prior - 1) + 
                    np.einsum('ik,jm,ij', self.tau_prime, self.tau_prime, self.sigma) + 1
                    )
        self.zeta = (self.delta_rho * (self.zeta_prior -1 ) + 
                     np.einsum('ik,jm,ij', self.tau_prime, self.tau_prime, 1 - self.sigma) + 1
                     ) 

    def _update_q_lam(self):
        """
        A method to compute the CAVI approximation to the posterior of lambda.
        This is run differently depending on the value of infer_graph_bool.
        """
        if not self.infer_graph_bool:
            self.sigma = self.adj_mat
        self.alpha = (self.delta_lam * (self.alpha_prior - 1) + 
                    np.einsum('ik,jm,ij,ij', self.tau, self.tau,
                            self.sigma, self.eff_count) + 1
        )
        self.beta = (self.delta_lam * self.beta_prior + 
                    self.int_length * np.einsum('ik,jm,ij', self.tau, self.tau,
                                                self.sigma)
        )
        
        # Adjust delta matrix for empty groups
        empty_groups_bool = self.tau.T @ self.sigma @ self.tau < 0.1
        self.delta_lam[empty_groups_bool] = 1
        self.delta_lam[~empty_groups_bool] = self.delta_lam_BFF
            
    def _update_q_u(self):
        """
        A method to compute the CAVI approximation to the posterior of each u_i.
        """
        ## Compute omega
        self.omega = (
            self.delta_u * (self.omega_prior - 1) +
            self.delta_z * self.tau.sum(axis=0) + 1
        )
        
        ## Calculate nu
        sum_term = np.zeros((self.num_nodes, ))
        self.nu = np.zeros((self.num_var_groups, ))
        for j in range(self.num_var_groups):
            sum_term[:] = self.tau[:,(j+1):].sum(axis=1) 
            self.nu[j] = (
                self.delta_z * sum_term.sum() + 
                self.delta_u * (self.nu_prior[j] - 1) + 1
            )

    def _MAD_JS_outlier_detector(self, tau, max_lag, L):
        """
        Function to detect nodes changing groups.
        The function takes data contains all points up to and including
        the current update time, assuming that the burn-in points aren't 
        includedm and outputs which nodes have changed groups.
        """
        ## Compute the JS-divergences off all lags up to current lag
        # List to store all the JS-divergences
        js_lag_list = list()
        # List to store the current JS-divergence
        js_curr_datum_list = list()

        curr_pred = tau[-1,:,:].argmax(axis=1)
        pred_change_bool = np.ones(shape=(self.num_nodes,), dtype=bool)
        for lag in range(1, max_lag + 1):
            tau_1 = tau[max_lag:, :]; tau_2 = tau[(max_lag - lag):-lag, :]

            curr_lag_pred = tau[-(lag + 1),:,:].argmax(axis=1)
            pred_change_bool = ((curr_pred != curr_lag_pred) & pred_change_bool)

            tau_1_safe = np.where(tau_1 == 0, 1e-10, tau_1)
            tau_2_safe = np.where(tau_2 == 0, 1e-10, tau_2)

            log_term_1 = (
                tau_1_safe * np.log(2 * tau_1_safe / (tau_1_safe + tau_2_safe))
            )
            log_term_2 = (
                tau_2_safe * np.log(2 * tau_2_safe / (tau_1_safe + tau_2_safe))
            )

            log_term_1[log_term_1 == 0] = 1e-300
            log_term_2[log_term_2 == 0] = 1e-300
            log_term_1 = np.where(tau_1 == 0, 1e-300, log_term_1)
            log_term_2 = np.where(tau_2 == 0, 1e-300, log_term_2)

            js_lag = (
                np.sum(log_term_1 + log_term_2, axis = 2) / 2 
            )
            js_curr_datum_list.append(js_lag[-1, :])
            js_lag_list.append(js_lag[:-1, :])

        js_curr_datum = np.array(np.log(np.abs(js_curr_datum_list)))
        js_lag = np.array(np.log(np.abs((js_lag_list)))).flatten()

        ## Compute the current MAD and the deviation of the current datum
        # Current MAD (excluding current datum)
        curr_MAD = median_abs_deviation(js_lag)
        # Deviation of current datum (for each lag up to max_lag)
        MAD_deviation_lags = []
        for i in range(max_lag):
            MAD_deviation_lags.append(
                np.abs(js_curr_datum[i, :] - np.median(js_lag)) / curr_MAD
                )
        # Flag as a change point if all lags are greater than the cutoff
        return (np.all((np.array(MAD_deviation_lags) > L), axis=0) & pred_change_bool)
    
    def compute_cal_Y_rates(self, kappa_kl, B2, cal_X_alpha, cal_X_beta, k, m):
        """
        Function that returns a populated cal_Y array.
        """
        length = int(kappa_kl * (2 * B2 - kappa_kl - 1) / 2)
        cal_Y_rates_km = np.zeros((length, ))
        idx = -1
        for s in range(1, kappa_kl + 1):
            for l in range(s, B2):
                idx += 1
                alpha_1 = cal_X_alpha[l,k,m]
                alpha_2 = cal_X_alpha[l-s,k,m]
                beta_1 = cal_X_beta[l,k,m]
                beta_2 = cal_X_beta[l-s,k,m]
                cal_Y_rates_km[idx] = (
                    alpha_2 * np.log(beta_1 / beta_2) - 
                    gammaln(alpha_1) + gammaln(alpha_2) +
                    (alpha_1 - alpha_2) * digamma(alpha_1) -
                    (beta_1 - beta_2) * alpha_1 / beta_1
                )
                        
        return cal_Y_rates_km

    def run_full_var_bayes(self, delta_pi: float=1, delta_u: float=1, delta_lam: float=1, 
                           delta_rho: float=1, delta_z:float=1, n_cavi: int=2, num_fp_its: int = 3,
                           B1: int=10, B2: int=10,
                           kappa_kl: int=2, L_kl: float=10, reset_stream: bool=True,
                           kappa_js: int=2, L_js: float=3.5):
        """
        A method to run the variational Bayesian update in its entirety.
        Parameters:
            - delta_pi, delta_u, delta_rho, delta_lam, delta_rho: BFF values.
            - n_cavi: the number of CAVI iterations at each run.
            - num_fp_its: the number of iterations of the fixed point equations.
            - B1: the number of steps we allow up until we start to track 
                             change point metrics.
            - B2: the assumed time of stationarity between changes
                            to the rate of the process.
            - kappa_kl: maximum lag (IN NUMBER OF STEPS) we consider for 
                               the KL flag (kappa for rates).
            - L_kl: the threshold for the MAD-KL flag (L for rates).
            - reset_stream: Boolean for whether stream for rate changes is reset after
                            a change is flagged.
            - kappa_js: maximum lag (IN NUMBER OF STEPS) we consider for 
                             the JS flag (kappa for memberships).
            - L_js: the threshold for the MAD-JS flag (L for memberships).
        """
        # ==============================================    
        # Initialise empty arrays and save parameteters
        # ==============================================
        if self.infer_num_groups_bool:
            num_check_groups = self.num_var_groups
        else:
            num_check_groups = self.num_groups
        
        ## BFF values for tempering the prior
        self.delta_u = delta_u
        self.delta_pi = delta_pi
        self.delta_rho = delta_rho
        self.delta_lam_BFF = delta_lam
        self.delta_z = delta_z
        if self.infer_num_groups_bool:
            self.delta_lam = np.ones((self.num_var_groups, 
                                      self.num_var_groups)) * delta_lam
        else:
            self.delta_lam = np.ones((self.num_groups, 
                                      self.num_groups)) * delta_lam

        if self.infer_graph_bool:
            self.lam_big_store = np.zeros((self.num_nodes, self.num_nodes))

        ## Arrays for storing estimates
        if self.infer_num_groups_bool:
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
            self.tau_store = np.zeros((len(self.intervals) + 1, 
                                        self.num_nodes, 
                                        self.num_groups))
            self.tau_prime_store = np.zeros((len(self.intervals) + 1, 
                                             self.num_nodes, 
                                             self.num_groups_prime))
            self.alpha_store = np.zeros((len(self.intervals) + 1, 
                                        self.num_groups, 
                                        self.num_groups))
            self.beta_store = np.zeros((len(self.intervals) + 1, 
                                        self.num_groups, 
                                        self.num_groups))
            
        self.group_memberships = np.zeros((len(self.intervals) + 1 - B1,
                                          self.num_nodes))

        if self.infer_num_groups_bool:
            self.tau_prior = (
                np.array([1 / self.num_var_groups] * 
                         (self.num_nodes * self.num_var_groups))
                .reshape((self.num_nodes, self.num_var_groups))
            )
        else:
            if self.infer_graph_bool:
                self.tau_prime_prior = (
                np.array([1 / self.num_groups_prime] * (self.num_nodes * self.num_groups_prime))
                .reshape((self.num_nodes, self.num_groups_prime))
            )
            self.tau_prior = (
                np.array([1 / self.num_groups] * (self.num_nodes * self.num_groups))
                .reshape((self.num_nodes, self.num_groups))
            )
        
        # Set initial hyperparameter values
        self.alpha = self.alpha_prior
        self.beta = self.beta_prior
        if self.infer_num_groups_bool:
            self.nu = self.nu_prior
            self.omega = self.omega_prior
        else:
            if self.infer_graph_bool:
                self.eta = self.eta_prior
                self.zeta = self.zeta_prior
                self.xi = self.xi_prior
            self.gamma = self.gamma_prior

        # Store parameter values
        self.alpha_store[0,:,:] = self.alpha
        self.beta_store[0,:,:] = self.beta
        self.tau_store[0,:,:] = self.tau_prior
            
        ## List for storing flagged changes
        self.group_changes_list = []
        self.rate_changes_list = []
        
        ## Run the VB inference procedure 
        wait = np.full((num_check_groups, num_check_groups), False) 
        wait_count = np.zeros((num_check_groups, num_check_groups))            
        for it_num, update_time in enumerate(self.intervals):
            print(f"...Iteration: {it_num + 1} of {len(self.intervals)}...", end='\r')
    
            # ==============================================
            # Run the variational Bayes inference procedure
            # ==============================================
            ## Compute counts in the interval
            self._compute_eff_count(update_time)
 
            ## Update estimates (run CAVI n_cavi times)
            # Inferring the graph structure
            if self.infer_graph_bool:
                cavi_count = 0
                while cavi_count < n_cavi:
                    for rep in range(num_fp_its):
                        self._update_q_z()
                    self.tau = self.tau_prior.copy()
                    for rep in range(num_fp_its):
                        self._update_q_z_prime()
                    self.tau_prime = self.tau_prime_prior.copy()
                    self._update_q_pi()
                    self._update_q_mu()
                    self._update_q_lam()
                    self._update_q_rho()
                    self._update_q_a()
                    cavi_count += 1

            # Inferring the number of groups
            elif self.infer_num_groups_bool:
                cavi_count = 0
                while cavi_count < n_cavi:
                    for rep in range(num_fp_its):
                        self._update_q_z()    
                    self.tau = self.tau_prior.copy()
                    self._update_q_u()
                    self._update_q_lam()
                    cavi_count += 1
            
            # Known graph structure and number of groups
            else:
                cavi_count = 0
                while cavi_count < n_cavi:
                    for rep in range(num_fp_its):
                        self._update_q_z()
                    self.tau = self.tau_prior.copy()
                    self._update_q_pi()
                    self._update_q_lam()
                    cavi_count += 1

            # ========================
            # Check for change points
            # ========================
            if (it_num + 1) == B1:
                ## Store initial estimates of group memberships
                self.group_memberships[0, :] = self.tau.argmax(axis=1)
            if (B1 < (it_num + 1)) & ((it_num + 1) < B1 + B2 + 1):
                # Stationarity assumed, so keep previous membership
                self.group_memberships[it_num + 1 - B1, :] = (
                    self.group_memberships[it_num - B1, :]
                )
            if ((it_num + 1) == (B1 + B2 + 1)):
                ## Initialise sets for storing the distributions
                # Arrays to store values for checking (\mathcal{X}_{km})
                cal_X_alpha = np.zeros((B2, num_check_groups, num_check_groups))
                cal_X_beta = np.zeros((B2, num_check_groups, num_check_groups))
                # Array for storing KL divergences (\mathcal{Y})
                length = int(kappa_kl * (2 * B2 - kappa_kl - 1) / 2)
                cal_Y_rates = np.zeros((length, num_check_groups, num_check_groups))
                # Initialise cal_X_alpha and cal_X_beta
                for k in range(num_check_groups):
                    for m in range(num_check_groups):
                        # Up to but not including latest point
                        cal_X_alpha[:,k,m] = self.alpha_store[B1:it_num,k,m].copy()
                        cal_X_beta[:,k,m] = self.beta_store[B1:it_num,k,m].copy()
                        cal_Y_rates[:,k,m] = self.compute_cal_Y_rates(kappa_kl, B2, 
                                                                cal_X_alpha, cal_X_beta,
                                                                k, m)
                cal_X_tau = self.tau_store[B1:it_num].copy()
                cal_Y_groups = self.compute_cal_Y_groups(kappa_js, B2, cal_X_tau)
                
                # Stores how many consecutive outliers we've seen
                outlier_counter_rates = np.zeros((self.num_nodes, self.num_nodes)) 
                outlier_counter_groups = np.zeros((self.num_nodes, ))
                
            if (it_num + 1) > (B1 + B2 + 1):
                # =======================
                # Check for rate changes
                # =======================
                for k in range(num_check_groups):
                    for m in range(num_check_groups):
                        if not wait[k,m]: # Check if we need to wait to collect samples after change
                            # Compute KL between current estimate and latest of cal_Y
                            curr_KL_diff = (
                                cal_X_alpha[-1, k, m] * np.log(self.beta[k, m] / cal_X_beta[-1, k, m])
                                - gammaln(self.alpha[k, m])
                                + gammaln(cal_X_alpha[-1, k, m])
                                + (self.alpha[k, m] - cal_X_alpha[-1, k, m]) * digamma(self.alpha[k, m])
                                - (self.beta[k, m] - cal_X_beta[-1, k, m]) * self.alpha[k, m] / self.beta[k, m]
                            )
                            # Current MAD (excluding current datum)
                            curr_MAD = median_abs_deviation(cal_Y_rates[:,k,m])
                            # Deviation of current datum
                            MAD_deviation = (
                                    np.abs(curr_KL_diff - np.median(cal_Y_rates[:,k,m])) / curr_MAD
                                )
                            
                            if MAD_deviation > L_kl: # At least an outlier
                                outlier_counter_rates[k,m] += 1 
                        
                                if outlier_counter_rates[k,m] == kappa_kl: # A change point
                                    self.rate_changes_list.append(
                                        [self.intervals[it_num + 1 - kappa_kl], k, m]
                                        )
                                    outlier_counter_rates[k,m] = 0
                                    wait[k,m] = True
                                    wait_count[k,m] = 0
                            else:
                                outlier_counter_rates[k,m] = 0 # Reset counter and recompute cal_X and cal_Y
                                
                                cal_X_alpha[:-1,k,m] = cal_X_alpha[1:,k,m].copy() # Shift all entries
                                cal_X_alpha[-1,k,m] = self.alpha[k,m].copy() # Adjust final value
                                cal_X_beta[:-1,k,m] = cal_X_beta[1:,k,m].copy() 
                                cal_X_beta[-1,k,m] = self.beta[k,m].copy()
                            
                                cal_Y_rates[:,k,m] = self.compute_cal_Y_rates(kappa_kl, B2, 
                                                                        cal_X_alpha, cal_X_beta,
                                                                        k, m)
                        
                        else: # Have detected a change and are collecting samples
                            cal_X_alpha[int(wait_count[k,m]),k,m] = (
                                self.alpha_store[int(it_num - kappa_kl), k, m]
                            )
                            cal_X_beta[int(wait_count[k,m]),k,m] = (
                                self.beta_store[int(it_num - kappa_kl), k, m]
                            )
                            wait_count[k,m] += 1
                            
                            if wait_count[k,m] == B2:
                                # Now compute cal_Y
                                cal_Y_rates[:,k,m] = self.compute_cal_Y_rates(kappa_kl, B2, 
                                                                        cal_X_alpha, cal_X_beta,
                                                                        k, m)
                                wait[k,m] = False
                                wait_count[k,m] = 0

                # =============================
                # Check for membership changes
                # =============================
                self.group_memberships[it_num + 1 - B1, :] = (
                    self.group_memberships[it_num - B1, :]
                )
                tau_burned = self.tau_store[(it_num + 1 - B2):(it_num + 1), :, :]
                mem_cp_flag = self._MAD_JS_outlier_detector(
                                        tau_burned, 
                                        kappa_js, L_js
                                        )
                changed_nodes = np.where(mem_cp_flag)[0]
                unchanged_nodes = np.where(~mem_cp_flag)[0]
                if len(changed_nodes) > 0:
                    self.group_changes_list.append(
                        [update_time,
                        changed_nodes]
                    )
                    self.group_memberships[it_num + 1 - B1, changed_nodes] = (
                        self.tau.argmax(axis=1)[changed_nodes]
                    )

            # ==============
            # Update priors 
            # ==============
            self.tau_prior = self.tau.copy()
            self.alpha_prior = self.alpha.copy()
            self.beta_prior = self.beta.copy()
            if self.infer_num_groups_bool:
                self.nu_prior = self.nu.copy()
                self.omega_prior = self.omega.copy()
            else:
                if self.infer_graph_bool:
                    self.tau_prime_prior = self.tau_prime.copy()
                    self.eta_prior = self.eta.copy()
                    self.zeta_prior = self.zeta.copy()
                    self.xi_prior = self.xi.copy()
                self.gamma_prior = self.gamma.copy()

            ## Store estimates
            self.tau_store[it_num + 1,:,:] = self.tau
            self.tau_prime_store[it_num + 1,:,:] = self.tau_prime
            self.alpha_store[it_num + 1,:,:] = self.alpha
            self.beta_store[it_num + 1,:,:] = self.beta