import numpy as np
from more_itertools import flatten

class BaseHomogeneousPoissonNetwork:
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

    def __init__(self, num_nodes, num_groups, T_max) -> None:
        self.num_nodes = num_nodes
        self.num_groups = num_groups
        self.T_max = T_max

    def _sample_change_point_time(self, num_cps):
        """
        Randomly generate times at which change points occur.
        """
        self.change_point_time = np.random.uniform(low=0, high=self.T_max, 
                                                   size=num_cps)
        self.change_point_time.sort()

    def _select_changing_node(self, num_cps):
        """
        Pick the node that changes group.
        """
        self.changing_node = np.random.randint(low=0, high=self.num_nodes,
                                               size=num_cps)

    def _assign_groups(self, random_groups, change_point, num_cps,
                       group_assignment_type='sequential', group_sizes = None, 
                       change_point_times=None, changing_nodes=None):
        """
        A method to assign each node to the group. This will produce two sets of group
        assignments: one pre-changepoint and one after. num_cps
        """
        # If specified, nodes will be assigned to groups in such a way
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

        # Sort the group assigments
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

        # If a change point(s) is specified, then we sample the group assingments
        # in each region around the change points.
        if change_point:
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
        
    def _sample_edge(self, lam, t_start, t_end) -> list:
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


class CompleteHomogeneousPoissonNetwork(BaseHomogeneousPoissonNetwork):
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
    def __init__(self, num_nodes, num_groups, T_max, lam_matrix=None):
        super().__init__(num_nodes, num_groups, T_max)

        if lam_matrix is None:
            # Sample a matrix
            lam_matrix = np.random.uniform(0.2, 2, size=(num_groups, num_groups))

        self.lam_matrix = lam_matrix

    def _random_partition(self, number, parts) -> list:
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
    
    def sample_network(self, random_groups, change_point, num_cps,
                       group_assignment_type='sequential', group_sizes=None, 
                       change_point_times=None, changing_nodes=None) -> dict(dict()):
        """
        A method to sample the full network.
        """
        # Assign nodes to groups
        groups_in_regions = self._assign_groups(random_groups, change_point, 
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
                if change_point:
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
    

# class IncompletePoissonNetwork()