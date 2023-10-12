#%%
import numpy as np
import pandas as pd

from fully_connected_poisson import FullyConnectedPoissonNetwork

class OnlineExponentialFCP:

    def __init__(self, num_nodes, num_groups, T_max, sampled_network) -> None:
        self.num_nodes = num_nodes
        self.num_groups = num_groups
        self.T_max = T_max
        self.sampled_network = sampled_network

    def _effective_count_update(self, node_out, node_in, N_prev, s_curr, s_prev):
        arrival_times = np.array(self.sampled_network[node_out][node_in])
        new_arrivals = arrival_times[(s_prev < arrival_times) &
                                     (arrival_times <= s_curr)]
        N_curr = (np.exp(-self.eta * (s_curr - s_prev)) * N_prev +
                  np.exp(-self.eta * (s_curr - new_arrivals)).sum())
        
        return N_curr

    def _effective_observation_time(self, s_curr):

        M_curr = (1 - np.exp(-self.eta * s_curr)) / self.eta

        return M_curr
    
    def _lambda_estimate(self, node_out, node_in, N_prev, s_curr, s_prev):
        
        N_curr = self._effective_count_update(node_out, node_in, N_prev,
                                                    s_curr, s_prev)
        M_curr =  self._effective_observation_time(s_curr)

        lambda_curr = N_curr / M_curr              

        return lambda_curr, N_curr

    def run(self, eta, time_step):

        self.eta = eta

        update_times = np.arange(time_step, self.T_max + time_step, 
                                 time_step)
        node_range = np.arange(self.num_nodes)
        lambda_time = np.zeros((len(update_times), self.num_nodes, 
                                self.num_nodes))
        N_array = np.zeros((self.num_nodes, self.num_nodes))

        # Iterate over update times
        s_prev = 0
        for k, s_curr in enumerate(update_times):
            for i in node_range:
                for j in node_range:
                    if not (i == j):
                        lambda_time[k,i,j], N_curr = (
                            self._lambda_estimate(i, j, N_array[i,j],
                                                  s_curr, s_prev)
                        )
                        N_array[i,j] = N_curr
                    
            s_prev = s_curr
        
        return lambda_time
    
#%%
FCP = FullyConnectedPoissonNetwork(num_nodes=5, num_groups=2, T_max=100,
                                   lam_matrix=np.array([[5, 1],
                                                        [3, 10]]))
sampled_network, groups_in_regions = FCP.sample_network(change_point=True, num_cps=2)
#%%
node_in = 0; node_out = 2
num_nodes = 5; num_groups = 2; T_max = 100
OEFCP = OnlineExponentialFCP(num_nodes, num_groups, T_max, sampled_network)
eta = 0.2; time_step = 0.1
lambda_time = OEFCP.run(eta, time_step)
print(f"CP time: {FCP.change_point_time}")
# plt.plot(np.arange(T_max/time_step), 
#          (np.cumsum(lambda_time[:,node_in, node_out]) / 
#           np.arange(1, T_max/time_step + 1)))
plt.plot(np.arange(T_max/time_step), lambda_time[:, node_in, node_out])

quantiles_high = []
quantiles_low = []
for i in range(1, len(lambda_time[:,0,0]) + 1):
    subset = lambda_time[:i, node_in, node_out]

    quantile_high = np.quantile(subset, 0.975)
    quantile_low = np.quantile(subset, 0.025)

    quantiles_high.append(quantile_high)
    quantiles_low.append(quantile_low)

plt.plot(np.arange(T_max/time_step), quantiles_high, color='red')
plt.plot(np.arange(T_max/time_step), quantiles_low, color='red')
# %%
groups_in_regions

# %%
@np.vectorize
def term(k):
    return ((1 - np.exp(- 2 * 1 * 2 * k)) / (1 - np.exp(- 1 * 2 * k)) ** 2)

plt.plot(np.arange(100), np.cumsum(term(np.arange(1, 101))) / np.arange(1, 101) ** 2)
# %%
