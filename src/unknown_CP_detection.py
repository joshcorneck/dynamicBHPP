#%%
import numpy as np
from scipy.stats import norm

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
    def _set_tau_equations_fixed_point(log_tau, num_nodes, num_groups, int_length, 
                                       lam, pi, N_count):
        """
        Defines the set of equations we need to solve. We assume tau
        is organised as (tau_11,...,tau_1K, tau_21,...,tau_2K,...,tau_NK). 
        Parameters are described in _solve_tau_equations_fixed_point method.
        """
        fnc_return = []
        
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
                                    np.exp(log_tau[num_groups * j + m]) * 
                                    (N_count[i][j] * np.log(lam[k,m]) +
                                    N_count[j][i] * np.log(lam[m,k]) -
                                    int_length * (lam[k,m] + lam[m,k]))
                                )
                else:
                    eqn_ik = 0
                eqns_i[k] = eqn_ik

            # Logsumexp trick to normalise. This normalises the sum of tau_ik
            eqns_i = np.array(eqns_i); c = eqns_i.max()
            eqns_i = (
                eqns_i - (c + np.log(np.sum(np.exp(eqns_i - c))))
                )
   
            fnc_return.append(eqns_i.tolist())

        # Flatten the output return
        fnc_return = [eqn for sublist in fnc_return for eqn in sublist]

        return np.array(fnc_return)

    def _solve_tau_equations_fixed_point(self, n_its, lam_ell, pi_ell, int_length,
                                        N_count):
        """
        Defines a method for solving for tau using fixed point iteration.
        Parameters:
            - n_its: number of iterations for the fixed-point iterations.
            - lam_ell: current value of lambda estimates.
            - pi_ell: current estimate of pi.
            - int_length: the interval over which samples are observed.
            - N_count: numpy array containing the counts on each edge.
        """
        # Set of equations that we need to solve
        target_eqns = lambda x: self._set_tau_equations_fixed_point(x, self.num_nodes, 
                                          self.num_groups, int_length, 
                                          lam_ell, pi_ell, N_count)
        
        # Empty array to store the log-scale solutions
        log_tau = np.log([1/self.num_groups]*(self.num_nodes*self.num_groups))

        # Use fixed-point iteration to solve the system of equations
        for i in range(n_its):
            log_tau = target_eqns(log_tau)

        # Convert solutions to original scale
        tau = np.exp(log_tau.reshape((self.num_nodes, self.num_groups)))

        return tau
    
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
    
    def _run_base_VEM(self, pi_ell_minus, lam_ell_minus, n_fp_its, n_EM_its, 
                      int_length, N_count):
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
            tau_ell = self._solve_tau_equations_fixed_point(n_fp_its, 
                                lam_ell_minus, pi_ell_minus, int_length,
                                N_count)

            ## STEP 3 - Compute pi_ell and lam_ell ##
            pi_ell, lam_ell = self._pi_and_lambda(tau_ell, int_length,
                                                  N_count)
            pi_ell_minus = pi_ell; lam_ell_minus = lam_ell

        return (tau_ell, pi_ell, lam_ell)



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

    def run_VEM(self, n_fp_its, n_EM_its):
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
                        self._run_base_VEM(pi_ell_minus, lam_ell_minus, n_fp_its, 
                                           n_EM_its, self.int_length, N_count)
            )
            group_preds = np.argmax(tau, axis=1)

            tau_store[int_num] = tau
            pi_store[int_num] = pi
            lam_store[int_num] = lam
            group_preds_store[int_num] = group_preds

        return (tau_store, pi_store, lam_store, group_preds_store)


class GaussianVEM(BaseVEM):
    def __init__(self, num_nodes, num_groups, T_max,
                 sampled_network, num_ints):
        super().__init__(num_nodes, num_groups, T_max, 
                         sampled_network)
        self.num_ints = num_ints
        self.intervals = np.linspace(0, T_max, num_ints + 1)
        self.int_length = self.intervals[1]

    def _compute_Gaussian_counts(self,scale) -> dict:
        """
        Produces a dictionary of numpy arrays containing the counts
        on each edge.
        """
        count_dict = dict()
        for int_num in range(self.num_ints):
            counts = (
                np.zeros((self.num_nodes, self.num_nodes))
            )
            for i in range(self.num_nodes):
                for j in range(self.num_nodes):
                    if i == j:
                        counts[i,j] = 0
                    else:
                        numpy_edge = (
                            np.array(self.sampled_network[i][j])
                        )
                        counts[i,j] = (
                            norm.pdf(
                            numpy_edge, loc=self.intervals[int_num], scale=scale
                            ).sum()
                        )
            count_dict[int_num] = counts

            self.count_dict = count_dict

    def run_VEM(self, n_fp_its, n_EM_its, scale):
        """
        Run the top-hat VEM procedure. It will iterate over each 
        of the intervals.
        """
        self._compute_Gaussian_counts(scale)

        tau_store, pi_store, lam_store = dict(), dict(), dict()
        group_preds_store = dict()

        for int_num in range(self.num_ints):
            print(f"Iteration: {int_num}")
            N_count = self.count_dict[int_num]
            if int_num == 0:
                    pi_ell_minus = np.array([1/self.num_groups] * self.num_groups)
                    lam_ell_minus = np.random.uniform(low=0, high=10, 
                                          size=(self.num_groups, self.num_groups))
            else:
                pi_ell_minus = pi
                lam_ell_minus = lam
            (tau, pi, lam) = (
                        self._run_base_VEM(pi_ell_minus, lam_ell_minus, n_fp_its, 
                                           n_EM_its, self.int_length, N_count)
            )
            group_preds = np.argmax(tau, axis=1)

            tau_store[int_num] = tau
            pi_store[int_num] = pi
            lam_store[int_num] = lam
            group_preds_store[int_num] = group_preds

        return (tau_store, pi_store, lam_store, group_preds_store)

#%%
FCP = FullyConnectedPoissonNetwork(num_nodes=30, num_groups=2, T_max=100,
                                   lam_matrix=np.array([[4, 1],
                                                        [2, 7]]))
sampled_network, groups_in_regions = FCP.sample_network(change_point=True, num_cps=1)
#%%
THV = TopHatVEM(30, 2, 100, sampled_network, 100)
tau_store, pi_store, lam_store, group_preds_store = THV.run_VEM(5, 5)
#%%
import matplotlib.pyplot as plt

plt.plot(np.arange(100), [tau_store[k][29,0] for k in tau_store])
# %%
plt.plot(np.arange(100), [lam_store[k][0,0] for k in lam_store])

plt.plot(np.arange(100), [lam_store[k][1,0] for k in lam_store])

plt.plot(np.arange(100), [lam_store[k][1,1] for k in lam_store])

plt.plot(np.arange(100), [lam_store[k][0,1] for k in lam_store])

# %%
