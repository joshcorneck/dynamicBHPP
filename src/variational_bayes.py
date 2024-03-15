from typing import Dict
import numpy as np
from scipy.special import gammaln, digamma, logsumexp
from scipy.stats import median_abs_deviation

class VariationalBayes:

    def __init__(self, sampled_network: Dict[int, Dict[int, list]]=None, 
                 num_nodes: int=None, num_groups: int=None, alpha_0: float=None, 
                 beta_0: float=None, gamma_0: np.array=None, adj_mat: np.array=None,
                 infer_graph_bool: bool=False, sigma_0: np.array=None, eta_0: np.array=None, 
                 zeta_0: np.array=None, infer_num_groups_bool: bool=False, 
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
            self.eta_prior = eta_0
            self.zeta_prior = zeta_0
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
        """
        A method to compute the CAVI approximation for the posterior of $a$. This
        is only run if infer_graph_structure = True.
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
                            - self.int_length * 
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
        A method to compute the CAVI approximation to the posterior of z.
        This is computed differently depending on the value of 
        infer_graph_structure and infer_number_groups. 
        """
        ## Functions to compute terms for fixed-point equations. This is 
        ## different for each condition.
        
        # If we need to infer the graph structure
        def sum_term_infer_graph(i, j_prime, k):
            """
            Function to compute sum over groups within exponential (when we DO need
            to infer the graph structure).
            Parameters:
                - i, k: node and groups indices of responsibility to calculate.
                - j_prime: index of node to check that i is connected to.
            """
            term = self.tau_prior[j_prime,:] * (
                self.eff_count[i,j_prime] * self.sigma[i,j_prime] * (
                    digamma(self.alpha[k,:]) - np.log(self.beta[k,:])
                ) +
                self.eff_count[j_prime,i] * self.sigma[j_prime,i] * (
                    digamma(self.alpha[:,k]) - np.log(self.beta[:,k])
                ) -
                
                self.sigma[i,j_prime] * 
                self.alpha[k,:] / self.beta[k,:] -
                self.sigma[j_prime,i] * 
                self.alpha[:,k] / self.beta[:,k] + 
                
                self.sigma[i,j_prime] * (
                    digamma(self.eta[k,:]) - 
                    digamma(self.eta[k,:] + self.zeta[k,:])
                ) +
                (1 - self.sigma[i,j_prime]) * (
                    digamma(self.zeta[k,:]) - 
                    digamma(self.eta[k,:] + self.zeta[k,:])
                ) + 

                self.int_length * (
                    self.sigma[j_prime,i] * (
                        digamma(self.eta[:,k]) - 
                        digamma(self.eta[:,k] + self.zeta[:,k])
                    ) +
                    (1 - self.sigma[j_prime,i]) * (
                        digamma(self.zeta[:,k]) - 
                        digamma(self.eta[:,k] + self.zeta[:,k])
                    )
                )
            )

            return term.sum()

        # If we don't need to infer the graph structure
        def sum_term_known_graph(i, j_prime, k):
            """
            Function to compute sum over groups within exponential (when we DON'T need
            to infer the graph structure) and are NOT inferring groups.
            Parameters:
                - i, k: node and groups indices of responsibility to calculate.
                - j_prime: index of node to check that i is connected to.
            """

            term = 0

            ## Two if statements: one for (i,j_prime) \in E and one for (j_prime,i) \in E.
            if self.adj_mat[i,j_prime] == 1:
                # This computes the interior sum of the exponential over m \in Q for 
                # edge (i,j_prime) if this is in the edge set. 
                term_out = self.tau_prior[j_prime,:] * (
                    self.eff_count[i,j_prime] * (
                        digamma(self.alpha[k,:]) - np.log(self.beta[k,:])
                    ) -
                    self.int_length * 
                    (self.alpha[k,:] / self.beta[k,:])
                )

                term += term_out.sum()

            if self.adj_mat[j_prime,i] == 1:
                # This computes the interior sum of the exponential over m \in Q for 
                # edge (j_prime,i) if this is in the edge set.
                term_in = self.tau_prior[j_prime,:] * (
                    self.eff_count[j_prime,i] * (
                        digamma(self.alpha[:,k]) - np.log(self.beta[:,k])
                    ) -
                    self.int_length * 
                    (self.alpha[:,k] / self.beta[:,k])
                )

                term += term_in.sum()


            return term
        
        # Function to compute tau when IN the burn-in period.
        def compute_tau_infer_groups(sum_func, num_nodes, num_var_groups):
            # Empty array to store updates.
            tau_temp = np.zeros((num_nodes, num_var_groups))

            # Iterate over each i \in V (each node in node set).
            for i in range(num_nodes):
                for k in range(num_var_groups):
                    temp_sum = (
                        digamma(self.omega[k]) -
                        digamma(self.nu[k] + 
                                self.omega[k]) + 
                        np.sum(
                            digamma(self.nu[:k]) -
                            digamma(self.nu[:k] + 
                                    self.omega[:k]))
                    )
                    for j_prime in range(self.num_nodes):
                        temp_sum += sum_func(i, j_prime, k)
                    tau_temp[i,k] = temp_sum
            
            return tau_temp

        # Function to compute tau when NOT IN the burn-in period.
        def compute_tau_known_groups(sum_func, num_nodes, num_groups, gamma):
            tau_temp = np.zeros((num_nodes, num_groups))
            # Compute tau_{ik} for each i \in V and k \in Q.
            for i in range(num_nodes):
                for k in range(num_groups):
                    temp_sum = ((digamma(gamma[k]) - 
                                 digamma(gamma.sum()))
                    )
                    for j_prime in range(self.num_nodes):
                        temp_sum += sum_func(i, j_prime, k)
                    tau_temp[i,k] = temp_sum
            
            return tau_temp
        
        # Structured in this way to prevent multiple bool checks
        if self.infer_num_groups_bool:
            tau_temp = compute_tau_infer_groups(sum_term_known_graph, self.num_nodes,
                                           self.num_var_groups)
        elif self.infer_graph_bool:
            tau_temp = compute_tau_known_groups(sum_term_infer_graph, self.num_nodes,
                                   self.num_groups, self.gamma)
        else:
            tau_temp = compute_tau_known_groups(sum_term_known_graph, self.num_nodes,
                                   self.num_groups, self.gamma)

        # Convert to exponential and normalise using logsumexp
        self.tau = np.exp(tau_temp - logsumexp(tau_temp, axis=1)[:,None])

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
            kl_curr_datum_list.append(kl_lag[-1])
            kl_lag_list.append(kl_lag[:-1])
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
                           cp_burn_steps: float=10, cp_kl_lag_steps: int=2,
                           cp_kl_thresh: float=10, cp_rate_wait: float=0.5):
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
        self.delta_lam = delta_lam

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
                self.sigma_prior = self.sigma_0
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
                    for j in range(self.num_groups):
                        for k in range(self.num_groups):
                            # +1 on the index to include current point
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
                                    self.rate_changes_list.append(
                                        [update_time, j, k]
                                        )
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

            print(f"Group props: {self.tau.mean(axis=0)}")

            # Update delta_lam
            # self.delta_lam = 

            
