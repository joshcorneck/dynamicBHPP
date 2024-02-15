from typing import Dict
import numpy as np
from scipy.special import gammaln, digamma, logsumexp

class VariationalBayes:

    def __init__(self, sampled_network: Dict[int, Dict[int, list]]=None, 
                 num_nodes: int=None, num_groups: int=None, alpha_0: np.array=None, 
                 beta_0: np.array=None, gamma_0: np.array=None, adj_mat: np.array=None,
                 infer_graph_bool: bool=False, sigma_0: np.array=None, eta_0: np.array=None, 
                 zeta_0: np.array=None, est_num_groups_bool: bool=False, burn_in: int=None,
                 num_var_groups: int=None, alpha: float=None, beta: float=None,
                 nu: float=None, int_length: float=1, T_max: float=100) -> None:
        
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
            - est_num_groups_bool: a Boolean whether we implement a nonparametric approach
                                   on the burn-in to estimate the number of groups.
            - num_var_groups: the number of groups in the variational approximation to the 
                              posterior from the nonparametric approach.
            - alpha, beta, nu: the parameters of the gamma and beta distributions a priori.
            - int_length: the time between updates.
            - T_max: the upper bound of the full observation window (assuming starting 
                     from 0).
        """
        # Run necessary checks on inputs
        if (adj_mat is not None) & (infer_graph_bool):
            raise ValueError("""You can't supply an adjacency matrix and set
                            infer_graph_bool = True.""")
        if (adj_mat is None) & (not infer_graph_bool):
            raise ValueError("""You must supply an adjacency matrix if
                            infer_graph_bool = False.""")
        if (adj_mat is not None) & (not infer_graph_bool):
            if adj_mat.shape != (num_nodes,num_nodes):
                raise ValueError("""The shape of the supplied adjacency
                                 matrix must match the number of nodes.""")
        if est_num_groups_bool:
           if burn_in is None:
               raise ValueError("""You must supply a burn-in by passing an integer
                                value for burn_in""")
           self.burn_in = burn_in
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
           if alpha is None:
               raise ValueError("""You must supply a float for alpha""")
           elif not isinstance(alpha, float):
               alpha = alpha.__float__()
           if beta is None:
               raise ValueError("""You must supply a float for beta""")
           elif not isinstance(beta, float):
               beta = beta.__float__()
           if nu is None:
               raise ValueError("""You must supply a float for nu""")
           elif not isinstance(nu, float):
               nu = nu.__float__()
                           
        if adj_mat is not None:
            self.adj_mat = adj_mat
        
        # Booleans
        self.infer_graph_bool = infer_graph_bool
        self.est_num_groups_bool = est_num_groups_bool

        # Network parameters
        self.num_nodes = num_nodes; self.num_groups = num_groups
        self.T_max = T_max; self.sampled_network = sampled_network

        # Sampling            
        # for i in range(self.num_groups):
            #     for j in range(self.num_groups):
            #         self.KL_div[it_num,i,j] = self._KL_div_gammas(
            #                                         self.alpha_store[it_num,i,j],
            #                                         self.alpha[i,j],
            #                                         self.beta_store[it_num,i,j],
            #                                         self.beta[i,j]
            #                                         )
        
        if est_num_groups_bool:
            self.intervals = np.arange(int_length, T_max, int_length)
        else:
            self.intervals = np.arange(int_length, T_max, int_length)
        self.int_length = int_length
        self.eff_count = np.zeros((self.num_nodes, self.num_nodes))
        self.eff_obs_time = np.zeros((self.num_nodes, self.num_nodes))
        self.eff_obs_time.fill(self.int_length)

        # Algorithm parameters - conditional on the type of procedure
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

        if est_num_groups_bool:
            self.num_var_groups = num_var_groups
            self.alpha = alpha; self.beta = beta; self.nu = nu

        if not est_num_groups_bool:
            self.alpha_prior = alpha_0
            self.beta_prior = beta_0
            self.gamma_prior = gamma_0
            self.tau = np.zeros((self.num_nodes, self.num_groups))
        
    def _compute_eff_count(self, update_time: float, burn_in_bool: bool=False):
        """
        A method to compute the effective count on each edge. This is simply 
        the number of observations on an edge from update_time - int_length to
        update_time. Parameters:
            - update_time: time at which we run the update.  
            - burn_in_bool: flag for if to run on the burn-in.
        """
        if burn_in_bool:
            int_length_temp = update_time
        else:
            int_length_temp = self.int_length

        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if self.sampled_network[i][j] is not None:
                    np_edge = np.array(
                        self.sampled_network[i][j]
                    )
                    self.eff_count[i,j] = (
                        len(
                            np_edge[
                                (update_time - int_length_temp <= np_edge)
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

    def _update_q_z(self, burn_in_bool: bool=False):
        """
        A method to compute the CAVI approximation to the posterior of z.
        This is computed differently depending on the value of 
        infer_graph_structure. Parameters:
            - burn_in_bool: flag for whether we are running the burn-in.
        """
        # Function to compute sum over groups within exponential (when we need
        # to infer the graph structure).
        def sum_term_infer(i_prime, j, k_prime):
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

                self.int_length * (
                    self.sigma[j,i_prime] * (
                        digamma(self.eta[:,k_prime]) - 
                        digamma(self.eta[:,k_prime] + self.zeta[:,k_prime])
                    ) +
                    (1 - self.sigma[j,i_prime]) * (
                        digamma(self.zeta[:,k_prime]) - 
                        digamma(self.eta[:,k_prime] + self.zeta[:,k_prime])
                    )
                )
            )

            return term.sum()
        
        # Function to compute sum over groups within exponential (when we DON'T need
        # to infer the graph structure).
        def sum_term(i_prime, j, k_prime):
            # If there is no edge, do not sum.
            if self.adj_mat[i_prime,j] == 1:
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
            else:
                return 0
        
        def compute_tau(sum_func, num_nodes, num_groups, delta_z,
                        gamma):
            tau_temp = np.zeros((num_nodes, num_groups))
            for i_prime in range(num_nodes):
                for k_prime in range(num_groups):
                    temp_sum = (delta_z * 
                                (digamma(gamma[k_prime]) - 
                                 digamma(gamma.sum()))
                    )
                    for j in range(self.num_nodes):
                        if j != i_prime:
                            temp_sum += sum_func(i_prime, j, k_prime)
                    tau_temp[i_prime,k_prime] = temp_sum
            
            return tau_temp
        
        # Structured in this way to prevent multiple bool checks
        if burn_in_bool:
            pass
        elif self.infer_graph_bool:
            tau_temp = compute_tau(sum_term_infer, self.num_nodes,
                                   self.num_groups, self.delta_z,
                                   self.gamma)
        else:
            tau_temp = compute_tau(sum_term, self.num_nodes,
                                   self.num_groups, self.delta_z,
                                   self.gamma)

        # Convert to exponential and normalise using logsumexp
        self.tau = np.exp(tau_temp - logsumexp(tau_temp, axis=1)[:,None])

    def _update_q_pi(self):
        """
        A method to compute the CAVI approximation to the posterior of pi.
        This is only run if infer_graph_structure = True.
        """
        self.gamma = (
            self.delta_pi * (self.gamma_prior - 1) + 
            self.delta_z * self.tau.sum(axis=0) + 1
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

    def _update_q_lam(self, burn_in=False):
        """
        A method to compute the CAVI approximation to the posterior of lambda.
        This is run differently depending on the value of infer_graph_bool.
        """
        if burn_in:
            self.sigma = np.ones((self.num_nodes, self.num_nodes))
            had_prod_x_sig = self.eff_count * self.sigma
            self.alpha = (self.alpha_prior + 
                        self.tau.T @ had_prod_x_sig @ self.tau)
            self.beta = (self.beta_prior + 
                        self.int_length * self.tau.T @ self.sigma @ self.tau)
        else:
            if not self.infer_graph_bool:
                self.sigma = np.ones((self.num_nodes, self.num_nodes))
            had_prod_x_sig = self.eff_count * self.sigma
            self.alpha = (self.delta_lam * (self.alpha_prior - 1) + 
                        self.tau.T @ had_prod_x_sig @ self.tau + 1)
            self.beta = (self.delta_lam * self.beta_prior + 
                        self.int_length * self.tau.T @ self.sigma @ self.tau)
    def _update_q_u(self):
        """
        """


    def _KL_div_gammas(self, a1, a2, b1, b2):
        """
        A methdod to compute the KL divergence between the approximate posteriors
        of lambda at two distinct update points.
        Parameters:
            - a1, b1: the rate and scale of the approx posterior from t-1.
            - a2, b2: the rate and scale of the approx posterior from t.
        """
        return (
            a2 * np.log(b1 / b2) - gammaln(a1) + gammaln(a2) +
            (a1 - a2) * digamma(a1) - (b1 - b2) * a1 / b1
        )

    def run_full_var_bayes(self, delta_z: float=1, delta_pi: float=1, 
                           delta_lam: float=1, delta_rho:float=1,
                           n_cavi: int=2, n_cavi_burn: int=5):
        """
        A method to run the variational Bayesian update in its entirety.
        Parameters:
            - delta_z, delta_pi, delta_rho, delta_lam: decay values.
            - n_cavi: the number of CAVI iterations at each run.
        """
        ## Decay rates for the prior
        self.delta_z = delta_z
        self.delta_pi = delta_pi
        self.delta_rho = delta_rho
        self.delta_lam = delta_lam

        ## Empty arrays for storage
        # Arrays that are needed only for graph inference
        if self.infer_graph_bool:
            self.sigma_store = np.zeros((len(self.intervals) + 1, 
                                         self.num_nodes, 
                                         self.num_nodes))
            self.eta_store = np.zeros((len(self.intervals) + 1, 
                                       self.num_groups, 
                                       self.num_groups))
            self.zeta_store = np.zeros((len(self.intervals) + 1, 
                                        self.num_groups, 
                                        self.num_groups))
        # Arrays that are needed only for estimating number of groups
        if self.est_num_groups_bool:
            self.alpha_prior = np.tile(self.alpha, 
                                       (self.num_var_groups, self.num_var_groups))
            self.beta_prior = np.tile(self.beta, 
                                      (self.num_var_groups, self.num_var_groups))
            self.nu_prior = np.tile(self.nu, (self.num_var_groups,))

        # Arrays that are always needed
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
        if not self.est_num_groups_bool:
            self.tau_prior = (
                np.array([1 / self.num_groups] * (self.num_nodes * self.num_groups))
                .reshape((self.num_nodes, self.num_groups))
            )
            if self.infer_graph_bool:
                self.sigma_prior = self.sigma_0
                self.eta = self.eta_prior
                self.zeta = self.zeta_prior
            self.gamma = self.gamma_prior
            self.alpha = self.alpha_prior
            self.beta = self.beta_prior

            if self.infer_graph_bool:
                self.sigma_store[0,:,:] = self.sigma_prior
                self.eta_store[0,:,:] = self.eta
                self.zeta_store[0,:,:] = self.zeta
            self.gamma_store[0,:] = self.gamma
            self.tau_store[0,:,:] = self.tau_prior
            self.alpha_store[0,:,:] = self.alpha
            self.beta_store[0,:,:] = self.beta

        ## Change-point indicators
        # self.KL_div = np.zeros((len(self.intervals), self.num_groups, self.num_groups))
        for it_num, update_time in enumerate(self.intervals):
            if (self.est_num_groups_bool) & (it_num == 0):
                ## Run burn-in      
                print(f"...Running burn-in computation...")

                # Compute counts in the interval
                self._compute_eff_count(update_time, burn_in_bool=True)



            else:
                ## Run remaining runs
                print(f"...Iteration: {it_num + 1} of {len(self.intervals)}...", end='\r')

                # Compute counts in the interval
                self._compute_eff_count(update_time, burn_in_bool=False)

                ## Update estimates (run CAVI n_cavi times)
                cavi_count = 0
                while cavi_count < n_cavi:
                    if self.infer_graph_bool:
                        self._update_q_a()
                    self._update_q_z()
                    cavi_count += 1

                self._update_q_pi()
                self._update_q_lam()
                if self.infer_graph_bool:
                    self._update_q_rho()
            
                # Compute the KL-divergence for each rate parameter
                # for i in range(self.num_groups):
                #     for j in range(self.num_groups):
                #         self.KL_div[it_num,i,j] = self._KL_div_gammas(
                #                                         self.alpha_store[it_num,i,j],
                #                                         self.alpha[i,j],
                #                                         self.beta_store[it_num,i,j],
                #                                         self.beta[i,j]
                #                                         )
                
                # Flag if CP has occurred
                

                # Store estimates
                if self.infer_graph_bool:
                    self.sigma_store[it_num + 1,:,:] = self.sigma
                    self.eta_store[it_num + 1,:,:] = self.eta
                    self.zeta_store[it_num + 1,:,:] = self.zeta
                self.gamma_store[it_num + 1,:] = self.gamma
                self.tau_store[it_num + 1,:,:] = self.tau
                self.alpha_store[it_num + 1,:,:] = self.alpha
                self.beta_store[it_num + 1,:,:] = self.beta

                # Update priors
                if self.infer_graph_bool:
                    self.eta_prior = self.eta.copy()
                    self.zeta_prior = self.zeta.copy()
                self.gamma_prior = self.gamma.copy()
                self.tau_prior = self.tau.copy()
                self.alpha_prior = self.alpha.copy()
                self.beta_prior = self.beta.copy()