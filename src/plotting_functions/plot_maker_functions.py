import numpy as np
import matplotlib.pyplot as plt

def posterior_rate_mean(num_groups: int, alpha: np.array, beta: np.array,
                        true_rates: np.array, int_length: float=1, 
                        T_max: float=100, save: bool=True,
                        file_path: str=''):
    """
    """
    for j in range(num_groups):
        for k in range(num_groups):
                plt.plot(np.arange(int(T_max/int_length) - 1),
                        [alpha[i,j,k] / beta[i,j,k] 
                         for i in np.arange(int(T_max/int_length) - 1)]);
                plt.axhline(true_rates[j,k], linestyle='--', 
                            linewidth=1, color='gray');
    
    plt.xlabel("Update time"); plt.ylabel("Posterior rate mean");
    if save:
        plt.savefig(file_path + ".pdf")

def node_membership_probability(tau: np.array, change_node: int,
                                change_times: float, group: int,
                                int_length: float=1, T_max: float=100, 
                                save: bool=True, file_path: str=''):
    """
    """
    plt.plot(np.arange(0, T_max, int_length), tau[:,change_node,group]);
    for change_time in change_times:
        plt.axvline(x=change_time, color='r', linestyle='--', linewidth=1);
    plt.xlabel("Update time"); plt.ylabel(f"Probability of group {group}");
    if save:
        plt.savefig(file_path + ".pdf")

def posterior_adjacency_mean(num_groups: int, eta: np.array, zeta: np.array,
                             true_con_prob: np.array, int_length: float=1, 
                             T_max: float=100, save: bool=True,
                             file_path: str=''):
    """
    """
    for j in range(num_groups):
        for k in range(num_groups):
                plt.plot(np.arange(int(T_max/int_length) - 1),
                        [eta[i,j,k] / zeta[i,j,k] 
                            for i in np.arange(int(T_max/int_length) - 1)]);
                plt.axhline(true_con_prob[j,k], linestyle='--', 
                            linewidth=1, color='gray');
    plt.xlabel("Update time"); plt.ylabel("Posterior connection probability mean");
    if save:
        plt.savefig(file_path + ".pdf")

def global_group_probability(n: np.array, true_pi: np.array, int_length: float=1,
                                  T_max: float=100, save: bool=True, file_path: str=''):
    """
    """
    n = n / n.sum(axis=1, keepdims=1)
    plt.plot(np.arange(0, T_max, int_length), n);
    for i in range(len(true_pi)):
        plt.axhline(true_pi[i], linestyle='--', linewidth=1, color='gray');
    plt.xlabel("Update time"); plt.ylabel("Global group membership probability");
    if save:
        plt.savefig(file_path + ".pdf")
