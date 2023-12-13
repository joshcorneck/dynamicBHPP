#%%
from xmlrpc.client import Boolean
import numpy as np
from more_itertools import flatten

class BasePoissonNetwork:
    """
    The base class for networks (complete or incomplete) with 
    Poisson processes living on the edges. The num_nodes are assumed to 
    be partitioned into groups, with the Poisson process on each edge
    being dependent upon the group identity of the num_nodes it joins.

    Parameters:
        - num_nodes: an integer for the number of num_nodes in the network.
        - num_groups: an integer for the number of latent groups.
        - T_max: the upper limit of the time interval from which we
                 sample.
    """

    def __init__(self, num_nodes: int, num_groups: int, T_max: float) -> None:
        self.num_nodes = num_nodes
        self.num_groups = num_groups
        self.T_max = T_max

    def _sample_change_point_time(self, num_cps: int):
        """
        Randomly generate times at which change points occur.
        Parameters:
            - num_cps: the number of change points.
        """
        self.change_point_time = np.random.uniform(low=0, high=self.T_max, 
                                                   size=num_cps)
        self.change_point_time.sort()

    def _select_changing_node(self, num_cps: int):
        """
        Pick the node that changes group.
        Parameters:
            - num_cps: the number of change points.
        """
        self.changing_node = np.random.randint(low=0, high=self.num_nodes,
                                               size=num_cps)

    def _assign_groups(self, random_groups: Boolean, membership_change: Boolean, num_cps: int = None,
                       group_assignment_type: str = 'sequential', group_sizes: np.array = None, 
                       change_point_times: np.array = None, changing_nodes: np.array = None):
        """
        A method to assign each node to the group. This will produce num_cps + 1 sets of group
        assignments.
        Parameters:
            - random_groups: Boolean for whether we randomly assign group membership split.
            - membership_change: Boolean for whether we have a change in the group memberships.
            - num_cps: the number of change points we observe.
            - group_assignment_type: how we split the group memberships over the nodes. This
                                     can be any of "sequential",  "alternate" or "random".
            - group_sizes: the number of nodes in each group (must sum to num_nodes).
            - change_point_times: the times of the changes (must have length equal to num_cps).
            - changing_nodes: the nodes that change at each change point (must have length equal to num_cps).
        """
        self.membership_change = membership_change

        # If not specified, nodes will be assigned to groups in such a way
        # that the number of nodes in each group is roughly equal.
        if random_groups:
            # Assign num_nodes to groups
            group_sizes = np.array(
                self._random_partition(self.num_nodes, self.num_groups)
            )
            groups = np.array(
                list(flatten([[i]*j for i,j in enumerate(group_sizes)]))
            )
        else:
            if group_sizes is None:
                raise ValueError("""Please supply group sizes.""")
            elif group_sizes.sum() != self.num_nodes:
                return ValueError("""The group partition must sum to the number
                                  of nodes.""")
            elif len(group_sizes) != self.num_groups:
                return ValueError("""The number of groups in the allocation doesn't 
                                  match the specified number of groups.""")
            groups = np.array(
                list(flatten([[i]*j for i,j in enumerate(group_sizes)]))
            )

        # Order the nodes to match the desired split of group assignments
        if group_assignment_type == 'sequential':
            pass
        elif group_assignment_type == 'alternate':
            # Get the number of unique elements and the counts
            unique_elements, counts = np.unique(groups, return_counts=True)

            # Create a repeating array, where we repeat the minimum number of times
            min_count = np.min(counts)
            repeating_array = np.tile(unique_elements, min_count)

            # Append the remaining counts of the other groups
            remaining_elements = []
            for i in range(len(unique_elements)):
                num_to_append = counts[i] - min_count
                remaining_elements.append([i] * num_to_append)
            remaining_elements = np.array(list(flatten(remaining_elements)))

            groups = np.concatenate((repeating_array, remaining_elements)).astype(int)

        elif group_assignment_type == 'random':
            np.random.shuffle(groups)
        else:
            raise ValueError("""Please supply group_assignment_type from the
                             list ['sequential', 'alternate', 'random']""")

        # If a membership change point(s) is specified, then we sample the group assignments
        # in each region around the change points.
        if membership_change:
            if num_cps is None:
                raise ValueError("""Please enter an integer number of change
                                 points.""")
            # Run through options of whether the times and nodes are given
            if change_point_times is None:
                # Sample the change point times and the nodes that change.
                self._sample_change_point_time(num_cps)
            else:
                if len(change_point_times) != num_cps:
                    raise ValueError("""Please supply enough change point times
                                    to match the specified number of change points""")
                else:
                    self.change_point_time = change_point_times
                    self.change_point_time.sort()
                    
            if changing_nodes is None:
                self._select_changing_node(num_cps)
            elif len(changing_nodes) != num_cps:
                raise ValueError("""Please supply enough changing nodes
                                to match the specified number of change points""")
            else:
                self.changing_node = changing_nodes

            # Create a list of lists in which we have the sequential group
            # assignments.
            groups_in_regions = []
            groups_in_regions.append(groups)
            # Copy orginal group array
            groups_cps = groups.copy()
            # Iterate over change points
            for cp in range(num_cps):
                old_group = groups_cps[self.changing_node[cp]]
                new_group = old_group
                while new_group == old_group:
                    new_group = np.random.randint(0,self.num_groups)
                groups_cps[self.changing_node[cp]] = new_group 
                groups_in_regions.append(groups_cps.copy())

            return groups_in_regions
        
        else:
            return groups
        
    def _sample_edge(self, lam: float, t_start: float, t_end: float) -> list:
        """
        A method to sample the Poisson process on a given edge between
        two time points. The two time points are variables to allow
        for simulation of change points.
        Parameters: 
            - lam: the intensity value of the given Poisson process.
            - t_start: start of the simulation window.
            - t_end: end of the simulation window.
        """
        num_arrivals = np.random.poisson(lam * (t_end - t_start))
        arrivals = np.random.uniform(t_start, t_end, size=(num_arrivals, ))
        arrivals.sort()

        return arrivals.tolist()


class CompletePoissonNetwork(BasePoissonNetwork):
    """
        A class to simulate a fully connected network with 
        Poisson processes living on the edges. The num_nodes are assumed to 
        be partitioned into groups, with the Poisson process on each edge
        being dependent upon the group identity of the num_nodes it joins.

        Parameters:
            - num_nodes: an integer for the number of num_nodes in the network.
            - num_groups: an integer for the number of latent groups.
            - T_max: the upper limit of the time interval from which we
                        sample.
            - lam_matrix: a symmetric np.array of shape (groups, groups)
                            containing the intensity values for the
                            Poisson process. Entry ij is the intensity 
                            value for edge e_ij.
    """
    def __init__(self, num_nodes: int, num_groups: int, T_max: float, lam_matrix: np.array = None):
        super().__init__(num_nodes, num_groups, T_max)

        if lam_matrix is None:
            # Sample a matrix
            lam_matrix = np.random.uniform(0.2, 2, size=(num_groups, num_groups))

        self.lam_matrix = lam_matrix

    def _random_partition(self, number: int, parts: int) -> list:
        """
        A recursive function to partition a number in parts. This
        is used to randomly assign num_nodes to groups.
        """
        if parts == 1:
            return [number]
        max_value = number - parts + 1
        part = np.random.randint(low=1, high=max_value)
        remaining = number - part
        return [part] + self._random_partition(remaining, parts - 1)
    
    def sample_network(self, random_groups: Boolean, membership_change: Boolean, num_cps: int,
                       group_assignment_type: str = 'sequential', group_sizes: np.array = None, 
                       change_point_times: np.array = None, changing_nodes: np.array = None) -> dict(dict()):
        """
        A method to sample the full network.
        Parameters:
            - random_groups: Boolean for whether we randomly assign group membership split.
            - membership_change: Boolean for whether we have a change in the group memberships.
            - num_cps: the number of change points we observe.
            - group_assignment_type: how we split the group memberships over the nodes. This
                                     can be any of "sequential",  "alternate" or "random".
            - group_sizes: the number of nodes in each group (must sum to num_nodes).
            - change_point_times: the times of the changes (must have length equal to num_cps).
            - changing_nodes: the nodes that change at each change point (must have length equal to num_cps).
        """
        # Assign nodes to groups
        groups_in_regions = self._assign_groups(random_groups, membership_change, 
                                                num_cps, group_assignment_type, 
                                                group_sizes, change_point_times, 
                                                changing_nodes)

        # Sample the network
        network = dict()
        for i in range(self.num_nodes):
            # Dictionary to store arrivals relating to node i
            network[i] = dict()
            for j in range(self.num_nodes):
                network[i][j] = []
                if membership_change:
                    # Run from start to first CP
                    # Map node i and j to their groups
                    group_i = groups_in_regions[0][i]
                    group_j = groups_in_regions[0][j]
                    if j == i:
                        network[i][j] = None
                    else:

                        network[i][j] += (
                            self._sample_edge(self.lam_matrix[group_i,group_j],
                                    t_start=0, t_end=self.change_point_time[0])
                        )
                    # Iterate over CPs
                    for cp in range(1, num_cps):
                        # Map node i and j to their groups
                        group_i = groups_in_regions[cp][i]
                        group_j = groups_in_regions[cp][j]
                        if j == i:
                            network[i][j] = None
                        else:

                            network[i][j] += (
                                self._sample_edge(self.lam_matrix[group_i,group_j],
                                        t_start=self.change_point_time[cp-1], 
                                        t_end=self.change_point_time[cp])
                            )
                    # From final CP to the end
                    if num_cps == 1:
                        cp = 0
                    group_i = groups_in_regions[cp+1][i]
                    group_j = groups_in_regions[cp+1][j]
                    if j == i:
                        network[i][j] = None
                    else:

                        network[i][j] += (
                            self._sample_edge(self.lam_matrix[group_i,group_j],
                                    t_start=self.change_point_time[num_cps-1], t_end=self.T_max)
                        )

                else:
                    # Map node i and j to their groups
                    group_i = groups_in_regions[i]
                    group_j = groups_in_regions[j]
                    if j == i:
                        network[i][j] = None
                    else:
                        network[i][j] += (
                            self._sample_edge(self.lam_matrix[group_i,group_j],
                                              t_start=0, t_end=self.T_max)
                        )

        return (network, groups_in_regions)
    

class IncompletePoissonNetwork(BasePoissonNetwork):
    """
    A class to simulate a fully connected network with 
    Poisson processes living on the edges. The num_nodes are assumed to 
    be partitioned into groups, with the Poisson process on each edge
    being dependent upon the group identity of the num_nodes it joins.

    Parameters:
        - random_groups: Boolean for whether we randomly assign group membership split.
        - membership_change: Boolean for whether we have a change in the group memberships.
        - num_cps: the number of change points we observe.
        - group_assignment_type: how we split the group memberships over the nodes. This
                                    can be any of "sequential",  "alternate" or "random".
        - group_sizes: the number of nodes in each group (must sum to num_nodes).
        - change_point_times: the times of the changes (must have length equal to num_cps).
        - changing_nodes: the nodes that change at each change point (must have length equal to num_cps).
    """
    def __init__(self, num_nodes: int, num_groups: int, T_max: float, rho_matrix: np.array,
                 lam_matrix: np.array = None):
        super().__init__(num_nodes, num_groups, T_max)

        if lam_matrix is None:
            # Sample a matrix
            lam_matrix = np.random.uniform(0.2, 2, size=(num_groups, num_groups))

        self.lam_matrix = lam_matrix
        self.rho_matrix = rho_matrix

    def _sample_adjancency_matrix(self, groups: int):
        """
        Samples an adjacency matrix from the supplied rho_matrix.
        Parameters:
            - groups: the node memberships in each region.
        """

        # Extract the groups for each node
        if not self.change_point:
            group_assignments = groups
        else:
            group_assignments = groups[0]

        # Create an n x n matrix where the entries are the relevant 
        # rho_matrix values and sample the adjacency matrix
        temp_mat = self.rho_matrix[group_assignments]
        edge_probs = temp_mat[:, group_assignments]
        np.fill_diagonal(edge_probs, 0)
        self.adjacency_matrix = (
            (np.random.rand(self.num_nodes, self.num_nodes) < edge_probs).astype(int)
        )

    def sample_network(self, random_groups: Boolean, membership_change: Boolean, num_cps: int,
                       group_assignment_type: str = 'sequential', group_sizes: np.array = None, 
                       change_point_times: np.array = None, changing_nodes: np.array = None) -> dict(dict()):
        """
        A method to sample the full network.
        Parameters:
            - random_groups: Boolean for whether we randomly assign group membership split.
            - membership_change: Boolean for whether we have a change in the group memberships.
            - num_cps: the number of change points we observe.
            - group_assignment_type: how we split the group memberships over the nodes. This
                                     can be any of "sequential",  "alternate" or "random".
            - group_sizes: the number of nodes in each group (must sum to num_nodes).
            - change_point_times: the times of the changes (must have length equal to num_cps).
            - changing_nodes: the nodes that change at each change point (must have length equal to num_cps).
        """

        # Assign nodes to groups
        groups_in_regions = self._assign_groups(random_groups, membership_change, 
                                                num_cps, group_assignment_type, 
                                                group_sizes, change_point_times, 
                                                changing_nodes)
        
        # Sample adjacency matrix
        self._sample_adjancency_matrix(groups_in_regions)

        # Sample the network
        network = dict()
        for i in range(self.num_nodes):
            # Dictionary to store arrivals relating to node i
            network[i] = dict()
            for j in range(self.num_nodes):
                network[i][j] = []
                if self.adjacency_matrix[i,j] == 0:
                    network[i][j] = None
                else:
                    if membership_change:
                        # Run from start to first CP
                        # Map node i and j to their groups
                        group_i = groups_in_regions[0][i]
                        group_j = groups_in_regions[0][j]

                        network[i][j] += (
                            self._sample_edge(self.lam_matrix[group_i,group_j],
                                    t_start=0, t_end=self.change_point_time[0])
                        )
                        # Iterate over CPs
                        for cp in range(1, num_cps):
                            # Map node i and j to their groups
                            group_i = groups_in_regions[cp][i]
                            group_j = groups_in_regions[cp][j]
                            if j == i:
                                network[i][j] = None
                            else:

                                network[i][j] += (
                                    self._sample_edge(self.lam_matrix[group_i,group_j],
                                            t_start=self.change_point_time[cp-1], 
                                            t_end=self.change_point_time[cp])
                                )
                        # From final CP to the end
                        if num_cps == 1:
                            cp = 0
                        group_i = groups_in_regions[cp+1][i]
                        group_j = groups_in_regions[cp+1][j]

                        network[i][j] += (
                            self._sample_edge(self.lam_matrix[group_i,group_j],
                                    t_start=self.change_point_time[num_cps-1], t_end=self.T_max)
                        )

                    else:
                        # Map node i and j to their groups
                        group_i = groups_in_regions[i]
                        group_j = groups_in_regions[j]

                        network[i][j] += (
                            self._sample_edge(self.lam_matrix[group_i,group_j],
                                            t_start=0, t_end=self.T_max)
                        )

        return (network, groups_in_regions)
