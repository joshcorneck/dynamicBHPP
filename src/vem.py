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
    def _set_tau_equations(log_tau, num_nodes, num_groups, int_length, lam, pi, 
                           N_count):
        """
        Defines the set of equations we need to solve. We assume tau
        is organised as (tau_11,...,tau_1K, tau_21,...,tau_2K,...,tau_NK). 
        Parameters are described in _solve_tau_equations_fixed_point method.
        """
        fnc_return = []
        def log_f_km_Nij(k, m, i, j):
            return np.log(lam[k,m]) * N_count[i][j] - int_length * lam[k,m]
        

        # Iterate over node and group numbers to create the system of 
        # equations to solve
        for i in range(num_nodes):
            eqns_i = [0] * num_groups
            for k in range(num_groups):
                # Compute the double sum inside of each equation
                if pi[k] != 0:
                    eqn_ik = np.log(pi[k])
                    for j in range(num_nodes):
                        if j == i:
                            pass
                        else:
                            for m in range(num_groups):
                                eqn_ik += (
                                    (log_f_km_Nij(k,m,i,j) + log_f_km_Nij(m,k,j,i)) * 
                                    np.exp(log_tau[num_groups * j + m])
                                )
                else:
                    eqn_ik = 0
                eqns_i[k] = eqn_ik

            eqns_i = np.array(eqns_i); c = eqns_i.max()
            eqns_i = (
                eqns_i - (c + np.log(np.sum(np.exp(eqns_i - c))))
                )
   
            fnc_return.append(eqns_i)

        # Flatten the output return
        fnc_return = [eqn for sublist in fnc_return for eqn in sublist]

        return np.array(fnc_return)
    
    def _pi_and_lambda(self, tau_ell, int_length, N_count):
        """
        
        """   
        # Compute current pi estimate
        pi_ell = np.mean(tau_ell, axis=0)
        
        # Compute current lambda estimate
        
        # Efficiently compute tau_iktau_jm and store - can be used for 
        # numerator and denominator 
        lam_ell = np.zeros(shape=(self.num_groups, self.num_groups))
        for k in range(self.num_groups):
            tau_k_2d = tau_ell[:,k].reshape((self.num_nodes,1))
            for m in range(self.num_groups):
                tau_m_2d = tau_ell[:,m].reshape((1,self.num_nodes))
                # Here tau_prod_array[i,j] = tau_ik*tau_jm
                tau_prod_array = tau_k_2d * tau_m_2d
                if np.sum(tau_prod_array) < 1e-10:
                    lam_ell[k,m] = 1e-10
                else:
                    lam_ell[k,m] = (
                        np.sum(tau_prod_array * N_count) /
                        (int_length * np.sum(tau_prod_array))
                    )
                    # This can happen in rare cases that the tau_prod_array
                    # is zero except for diagonal entries
                    if lam_ell[k,m] == 0:
                        lam_ell[k,m] = 1e-10

        return pi_ell, lam_ell
    
    def _run_base_VEM(self, pi_ell_minus, lam_ell_minus, n_EM_its, 
                      n_fp_its, int_length, N_count):
        """
        Parameters:
         - pi_ell_minus: initialisation for pi vector.
         - lam_ell_minus: intialisation for the lambda matrix.
         - n_fp_its: number of iterations of the the number of times we 
                     run the fixed point iteration scheme on each EM run.
         - n_EM_its: numeber of iterations of the full EM scheme. 
         - int_length: the width of the interval we run the scheme on.

        """

        ## STEP 1 - Initialise parameters ##
        # pi_ell_minus = np.array([1/self.num_groups] * self.num_groups)
        # lam_ell_minus = np.random.uniform(low=0, high=10,
        #                           size=(self.num_groups, self.num_groups))

        for i in range(n_EM_its): 
            # print(f"EM-run: {i+1}")      
            ## STEP 2 - Compute tau_ell ##
            tau_eqns = lambda x: self._set_tau_equations(x, self.num_nodes, 
                                                         self.num_groups,
                                                         int_length, lam_ell_minus,
                                                         pi_ell_minus, N_count)
            
            # soln = root(tau_eqns, np.log([1/self.num_groups]*(self.num_nodes*self.num_groups)),
            #             method='hybr')
            # soln_log_tau = soln.x
            soln_log_tau = np.log([1/self.num_groups]*(self.num_nodes*self.num_groups))
            for k in range(n_fp_its):
                soln_log_tau = tau_eqns(soln_log_tau)
            soln_tau = np.exp(soln_log_tau)
            mask = soln_tau < 1e-10
            soln_tau[mask] = 0
            tau_ell = soln_tau.reshape((self.num_nodes, self.num_groups))

            ## STEP 3 - Compute pi_ell and lam_ell ##
            pi_ell, lam_ell = self._pi_and_lambda(tau_ell, int_length,
                                                  N_count)
            pi_ell_minus = pi_ell; lam_ell_minus = lam_ell

        return (tau_ell, pi_ell, lam_ell)


class ExponentialRW(BaseVEM):
    def __init__(self, num_nodes, num_groups, T_max,
                 sampled_network, time_step):
        super().__init__(num_nodes, num_groups, T_max, 
                         sampled_network)
        self.time_step = time_step
        self.intervals = np.arange(time_step, T_max, time_step)
        self.int_length = self.intervals[1]

    def _compute_full_eff_count(self, eta):

        eff_count_dict = dict()
        s_prev = 0; N_count = np.zeros((self.num_nodes, self.num_nodes))
        for num, s_curr in enumerate(self.intervals):
            N_count_new = self._compute_reweighting(
                N_count, s_curr, s_prev, eta 
            )
            eff_count_dict[num] = N_count_new
            N_count = N_count_new.copy()
            s_prev = s_curr

        return eff_count_dict

    def _compute_reweighting(self, N_count, s_curr, s_prev, eta) -> dict:
        """
        Produces a dictionary of numpy arrays containing the counts
        on each edge.
        """
        N_count_new = np.zeros((self.num_nodes, self.num_nodes))
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

                    # Update effective edge count
                    N_count_new[i,j] = (
                        np.exp(-eta*(s_curr - s_prev)) * N_count[i,j] +
                        np.exp(-eta*(s_curr - new_arrivals_ij)).sum()
                    )
                    
        return N_count_new

    def _compute_full_eff_obs_time(self, eta):

        full_eff_obs_time = []
        for s_curr in self.intervals:
            full_eff_obs_time.append(self._compute_eff_obs_time(s_curr, eta))

        return full_eff_obs_time

    def _compute_eff_obs_time(self, s_curr, eta):
        
        return (1 - np.exp(-eta * s_curr)) / eta

    def run_VEM(self, n_EM_its, n_fp_its, eta):
        """
        Run the top-hat VEM procedure. It will iterate over each 
        of the intervals.
        """
        # Get dictionary of effective counts at each sampling point
        eff_count_dict = self._compute_full_eff_count(eta)

        # Compute effective observation time
        full_eff_obs_times = self._compute_full_eff_obs_time(eta)

        # Empty dictionaries for storing parameters
        tau_store, pi_store, lam_store = dict(), dict(), dict()
        group_preds_store = dict()

        for it_num in range(len(self.intervals)):
            print(f"Iteration: {it_num+1} of {len(self.intervals)}")

            # Extract effective interval length and effective counts
            eff_int_length = full_eff_obs_times[it_num]
            eff_N_count = eff_count_dict[it_num]

            if it_num == 0:
                    # Initialise parameters randomly if it's the first run.
                    pi_ell_minus = np.array([1/self.num_groups] * self.num_groups)
                    lam_ell_minus = np.random.uniform(low=0, high=10, 
                                                      size=(self.num_groups, self.num_groups))
            else:
                # If not first run, then update current previous parameter estimates
                # with estimates from previous run
                pi_ell_minus = pi
                lam_ell_minus = lam

            # Compute parameter estimates
            (tau, pi, lam) = (
                        self._run_base_VEM(pi_ell_minus, lam_ell_minus, 
                                           n_EM_its, n_fp_its, eff_int_length, eff_N_count)
            )

            # Extract group predictions
            group_preds = np.argmax(tau, axis=1)

            # Store the parameters
            tau_store[it_num] = tau
            pi_store[it_num] = pi
            lam_store[it_num] = lam
            group_preds_store[it_num] = group_preds

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

