#%%
import numpy as np
from scipy.stats import norm
from scipy.optimize import root

from fully_connected_poisson import FullyConnectedPoissonNetwork

class BaseVEM():
    
    def __init__(self, num_nodes, num_groups, T_max, sampled_network):
        """
        A class to run a variational EM algorithm to infer the latent group structure and 
        the relevant parameter values for each group.
        Parameters:
            - num_nodes: an integer for the number of num_nodes in the sampled_network.
            - num_groups: an integer for the number of latent groups.
            - T_max: the upper limit of the time interval from which we
                        sample.
            - groups: mapping of node numbers to group numbers.
            - sampled_network: a dictionary of dictionaries, where sampled_network[i][j]
                        corresponds to the arrivals to edge e_ij (i->j)
        """
        self.num_nodes = num_nodes; self.num_groups = num_groups
        self.T_max = T_max; self.sampled_network = sampled_network

    @staticmethod
    def _set_tau_equations_log(log_tau, num_nodes, num_groups, int_length, lam_minus, 
                           pi_minus, N_count):
        """
        Defines the set of equations we need to solve. We assume tau
        is organised as (tau_11,...,tau_1K, tau_21,...,tau_2K,...,tau_NK). 
        Parameters are described in _solve_tau_equations_fixed_point method.
        """
        fnc_return = []
        def log_f_km_Nij(k, m, i, j):
            return np.log(lam_minus[k,m]) * N_count[i][j] - int_length[i,j] * lam_minus[k,m]
        

        # Iterate over node and group numbers to create the system of 
        # equations to solve
        for i in range(num_nodes):
            eqns_i = [0] * num_groups
            for k in range(num_groups):
                # Compute the double sum inside of each equation
                if pi_minus[k] == 0:
                    eqn_ik = np.log(1e-10)
                else: 
                    eqn_ik = np.log(pi_minus[k])
                for j in range(num_nodes):
                    if j == i:
                        pass
                    else:
                        for m in range(num_groups):
                            eqn_ik += (
                                (log_f_km_Nij(k,m,i,j) + log_f_km_Nij(m,k,j,i)) * 
                                np.exp(log_tau[num_groups * j + m])
                            )
                eqns_i[k] = eqn_ik

            eqns_i = np.array(eqns_i); c = eqns_i.max()
            eqns_i = (
                eqns_i - (c + np.log(np.sum(np.exp(eqns_i - c))))
                )
   
            fnc_return.append(eqns_i)

        # Flatten the output return
        fnc_return = [eqn for sublist in fnc_return for eqn in sublist]

        return np.array(fnc_return)

    def _pi_and_lambda(self, tau_ell, eff_obs_time, N_count, Bayes):
        """
        
        """   
        # Compute current pi estimate
        if Bayes:
            # Sample group assignments
            group_assigments = (
                np.array([np.random.choice(
                    np.arange(self.num_groups), p=tau_i) for tau_i in tau_ell])
            )
            _, group_counts = np.unique(group_assigments, return_counts=True)

            # Update alpha and pi using conjugate of dirichlet prior
            self.alpha = (self.alpha + group_counts) / (self.num_nodes + self.alpha.sum())
            pi_ell = np.random.dirichlet(self.alpha) 
            
        else:
            pi_ell = np.mean(tau_ell, axis=0)
        
        # Compute current lambda estimate
        numerator = (tau_ell.T @ N_count @ tau_ell)
        denominator = (tau_ell.T @ eff_obs_time @ tau_ell)
        mask = (denominator != 0)
        lam_ell = np.zeros_like(numerator, dtype=float)
        lam_ell[mask] = numerator[mask] / denominator[mask]
        lam_ell[lam_ell == 0] = 1e-10

        return pi_ell, lam_ell
    
    def _run_base_VEM(self, pi_ell_minus, lam_ell_minus, n_EM_its, 
                      n_fp_its, int_length, N_count, jitter_sd=None, 
                      tau_prev=None, Bayes=False):
        """
        Parameters:
         - pi_ell_minus: initialisation for pi vector.
         - lam_ell_minus: intialisation for the lambda matrix.
         - n_fp_its: number of iterations of the the number of times we 
                     run the fixed point iteration scheme on each EM run.
         - n_EM_its: numeber of iterations of the full EM scheme. 
         - int_length: the width of the interval we run the scheme on.

        """

        for i in range(n_EM_its): 
            # print(f"EM-run: {i+1}")      
            ## STEP 1 - Compute tau_ell ##
            tau_eqns = lambda x: self._set_tau_equations_log(x, self.num_nodes, 
                                                            self.num_groups,
                                                            int_length, lam_ell_minus,
                                                            pi_ell_minus, N_count)
            # Solve for the current value of tau
            # soln = root(tau_eqns, np.log([1/self.num_groups]*(self.num_nodes*self.num_groups)))
            # soln_tau = soln.x
            # print(soln_tau)

            if tau_prev is not None:
                # Initialise tau as a jittered version of the previous values.
                # Reshape the previous value to jitter amongs rows and then flatten 
                jitter = jitter_sd * np.random.normal(size=(self.num_nodes, self.num_groups))
                jittered_array = np.abs(tau_prev + jitter)
                row_norms = np.linalg.norm(jittered_array, axis=1, ord=1)
                soln_log_tau = np.log(jittered_array / row_norms[:, np.newaxis]).flatten()
            else:
                # If it's the first run, or no previous vector supplied then uniform prior
                soln_log_tau = np.log([1 / self.num_groups] * (self.num_nodes * self.num_groups))

            k = 0
            while k < n_fp_its:
                soln_log_tau = tau_eqns(soln_log_tau)
                k += 1
            soln_tau = np.exp(soln_log_tau)
            mask = soln_tau < 1e-10
            soln_tau[mask] = 0
            tau_ell = soln_tau.reshape((self.num_nodes, self.num_groups))

            ## STEP 2 - Compute pi_ell and lam_ell ##
            pi_ell, lam_ell = self._pi_and_lambda(tau_ell, int_length, N_count, Bayes)
            pi_ell_minus = pi_ell; lam_ell_minus = lam_ell

        return (tau_ell, pi_ell, lam_ell)


class ExponentialRW(BaseVEM):
    def __init__(self, num_nodes, num_groups, T_max,
                 sampled_network, time_step, xi1, xi2,
                 eta_base):
        super().__init__(num_nodes, num_groups, T_max, 
                         sampled_network)
        
        # Parameters for VEM 
        self.time_step = time_step
        self.intervals = np.arange(time_step, T_max, time_step)
        self.int_length = self.intervals[1]

        # Parameters for eta update
        self.xi1 = xi1; self.xi2 = xi2; self.eta_base = eta_base

    def _compute_eff_count_r(self, N_eff_count_r_minus, s_curr, 
                               s_prev, eta_r) -> dict:
        """
        Computes N_ij^eta for step r. This uses the previous effective
        count and the current eta value. Parameters:
            - N_eff_count_r_minus: a np.array of the previous effective counts.
            - s_curr, s_prev: the current and previous update point.
            - eta_r: the current eta value.
        """
        N_eff_count_r = np.zeros((self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i == j:
                    pass
                else:
                    # Get numpy array of edge arrival times and extract
                    # the new arrivals
                    np_e_ij = (
                        np.array(self.sampled_network[i][j])
                    )
                    mask = (s_prev < np_e_ij) & (np_e_ij <= s_curr)
                    new_arrivals_ij = np_e_ij[mask]

                    if len(new_arrivals_ij) != 0:
                        # Update effective edge count
                        N_eff_count_r[i,j] = (
                            np.exp(-eta_r[i,j] * (s_curr - s_prev)) * N_eff_count_r_minus[i,j] +
                            np.exp(-eta_r[i,j] * (s_curr - new_arrivals_ij)).sum()
                        )
                    else:
                        N_eff_count_r[i,j] = (
                            np.exp(-eta_r[i,j] * (s_curr - s_prev)) * N_eff_count_r_minus[i,j]
                        )
        
        return N_eff_count_r
    
    def _compute_eta(self, eta_r_minus, d):
        """
        Function to compute eta^r_{ij}. This is computed iteratively.  
        """
        eta_r = np.ones((self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i == j:
                    pass
                else: 
                    eta_r[i,j] = (
                        self.eta_base + self.xi1 * eta_r_minus[i,j] + 
                        self.xi2 * (d[i] + d[j])
                    )
        return eta_r
    
    def _compute_eff_obs_time(self, I_r_minus, s_curr, s_prev, eta_r):
        """
        A method to compute the effective observation time. This is I_r, and
        is computed iteratively.
        """
        I_r = (np.exp(-eta_r * (s_curr - s_prev)) * I_r_minus + 
               (1 - np.exp(-eta_r * (s_curr - s_prev))) / eta_r)
        for i in range(len(I_r)):
            I_r[i,i] = 0

        return I_r
    
    def _compute_d(self, tau_minus, tau_minus_2, p=2):

        d = np.zeros((self.num_nodes, ))
        for i in range(self.num_nodes):
            d[i] = (
                (
                abs(
                    tau_minus[(self.num_groups * i):(self.num_groups * (i + 1))] -
                    tau_minus_2[(self.num_groups * i):(self.num_groups * (i + 1))]
                ) ** p 
                ).sum() ** (1 / p)
            )

        return d

    def run_VEM(self, n_EM_its, n_fp_its, jitter_sd, Bayes=False):
        """
        Run the VEM procedure. It will iterate over each of the update
        points sequentially.
        """
        # # Get dictionary of effective counts at each sampling point
        # eff_count_dict = self._compute_full_eff_count(eta)

        # # Compute effective observation time
        # full_eff_obs_times = self._compute_full_eff_obs_time(eta)

        # Empty dictionaries for storing parameters
        tau_store, pi_store, lam_store = dict(), dict(), dict()
        group_preds_store = dict()

        # Instantiate initial values
        s_prev = 0
        I_r_minus = np.zeros((self.num_nodes, self.num_nodes))
        eta_r = np.tile(self.eta_base, (self.num_nodes, self.num_nodes)) 
        eta_r_minus = np.tile(self.eta_base, (self.num_nodes, self.num_nodes)) 
        N_eff_count_r_minus = np.zeros((self.num_nodes, self.num_nodes))
        tau_r_minus_2 = np.zeros((self.num_groups * self.num_nodes, ))
        tau_r_minus = np.zeros((self.num_groups * self.num_nodes, ))

        print("Beginning procedure...")
        for it_num, s_curr in enumerate(self.intervals):
            print(f"...Iteration: {it_num + 1} of {len(self.intervals)}...")

            # Compute eta_r (the current decay rate)
            if (it_num != 0) & (it_num != 1):
                # Compute d^r-2,r-1 values
                d_r_minus_2_r_minus = self._compute_d(tau_r_minus, tau_r_minus_2)
                # print(d_r_minus_2_r_minus)
                eta_r = self._compute_eta(eta_r_minus, d_r_minus_2_r_minus)

            # Compute the effective counts and the effective observation time
            I_r = self._compute_eff_obs_time(I_r_minus, s_curr, s_prev, eta_r)
            N_eff_count_r = self._compute_eff_count_r(N_eff_count_r_minus, s_curr,
                                                      s_prev, eta_r)

            if it_num == 0:
                # Initialise parameters randomly if it's the first run.
                # If using Bayesian, initialise the alpha vector
                if Bayes:
                    self.alpha = np.array([1] * self.num_groups)
                    pi_r_minus = np.random.dirichlet(self.alpha)
                else:
                    pi_r_minus = np.array([1/self.num_groups] * self.num_groups)
                lam_r_minus = np.random.uniform(low=0, high=10, 
                                                    size=(self.num_groups, self.num_groups))
                tau_r = (np.array([1 / self.num_groups] * (self.num_nodes * self.num_groups)).
                        reshape((self.num_nodes, self.num_groups)))
            else:
                # If not first run, then update current previous parameter estimates
                # with estimates from previous run
                pi_r_minus = pi_r
                lam_r_minus = lam_r

            # Compute parameter estimates
            (tau_r, pi_r, lam_r) = (
                        self._run_base_VEM(pi_r_minus, lam_r_minus, 
                                           n_EM_its, n_fp_its, I_r, N_eff_count_r,
                                           jitter_sd, tau_r, Bayes)
            )

            # Extract group predictions
            group_preds = np.argmax(tau_r, axis=1)

            # Store the parameters
            tau_store[it_num] = tau_r 
            pi_store[it_num] = pi_r; lam_store[it_num] = lam_r
            group_preds_store[it_num] = group_preds

            # Update for computing d_i^r-2,r-2
            if (it_num == 0):
                tau_r_minus_2 = tau_r.flatten()
            elif (it_num == 1):
                tau_r_minus = tau_r.flatten()
            else:
                tau_r_minus_2 = tau_r_minus
                tau_r_minus = tau_r.flatten()

            # Update values
            s_prev = s_curr; I_r_minus = I_r; eta_r_minus = eta_r
            N_eff_count_r_minus = N_eff_count_r

        return (tau_store, pi_store, lam_store, group_preds_store)


class TopHatVEM(BaseVEM):
    def __init__(self, num_nodes, num_groups, T_max,
                 sampled_network, num_ints):
        super().__init__(num_nodes, num_groups, T_max, 
                         sampled_network)
        self.num_ints = num_ints
        self.intervals = np.linspace(0, T_max, num_ints + 1)
        self.int_length = self.intervals[1]

    def _compute_top_hat_counts(self) -> dict:
        """
        Produces a dictionary of numpy arrays containing the counts
        on each edge.
        """
        count_dict = dict()
        for int_num in range(self.num_ints):
            counts = (
                np.zeros((self.num_nodes, self.num_nodes))
            )
            # Upper and lower interval bounds
            t_min = self.intervals[int_num]
            t_max = self.intervals[int_num + 1]
            for i in range(self.num_nodes):
                for j in range(self.num_nodes):
                    if i == j:
                        counts[i,j] = 0
                    else:
                        numpy_edge = (
                            np.array(self.sampled_network[i][j])
                        )
                        counts[i,j] = (
                            len(
                            numpy_edge[
                                (t_min <= numpy_edge)
                                &
                                (numpy_edge < t_max)
                                ]
                            )
                        )
            count_dict[int_num] = counts

            self.count_dict = count_dict

    def run_VEM(self, n_EM_its, n_fp_its):
        """
        Run the top-hat VEM procedure. It will iterate over each 
        of the intervals.
        """
        self._compute_top_hat_counts()

        tau_store, pi_store, lam_store = dict(), dict(), dict()
        group_preds_store = dict()

        for int_num in range(self.num_ints):
            print(f"Iteration: {int_num+1} of {self.num_ints}")
            N_count = self.count_dict[int_num]
            if int_num == 0:
                    pi_ell_minus = np.array([1/self.num_groups] * self.num_groups)
                    lam_ell_minus = np.random.uniform(low=0, high=10, 
                                                      size=(self.num_groups, self.num_groups))
            else:
                pi_ell_minus = pi
                lam_ell_minus = lam
            (tau, pi, lam) = (
                        self._run_base_VEM(pi_ell_minus, lam_ell_minus, 
                                           n_EM_its, n_fp_its,
                                           self.int_length, N_count)
            )
            group_preds = np.argmax(tau, axis=1)

            tau_store[int_num] = tau
            pi_store[int_num] = pi
            lam_store[int_num] = lam
            group_preds_store[int_num] = group_preds

        return (tau_store, pi_store, lam_store, group_preds_store)

