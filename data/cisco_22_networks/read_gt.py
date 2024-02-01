#!/usr/bin/python

# Simple code to read a ground truth (gt) file (groupings of nodes),
# and output basic statistics (number of groups, a histogram, etc).

# Example runs:

# python read_gt.py dir_g21_small_workload_with_gt/groupings.gt.txt 

# num gt sets=23  size(node_to_gt)=59
# gt sizes histo: Counter({2: 8, 3: 8, 1: 4, 4: 2, 7: 1})
# group sizes descending: [7, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1]

# Another run (read a ground truth grouping file from the extra graph
# directory:
#
# python read_gt.py dir_g22_extra_graph_with_gt/candidate_gt.minPrefix5.txt 

# ...

import sys, gzip
import numpy as np

from collections import defaultdict, Counter


def read_gt(gt_file):
    f = open(gt_file)
    node_gt = {}
    gt_to_nodes = defaultdict(set)
    i = 0
    for line in f:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        i += 1
        parts = line.split(',')
        for nid in parts:
           node_gt[nid] = i
           gt_to_nodes[i].add(nid)
    f.close()
    print( '\n# num gt sets=%d  size(node_to_gt)=%d' % (len(gt_to_nodes), len(node_gt)))
    sizes = Counter()
    ss = []
    for i, s in gt_to_nodes.items():
        sizes[len(s)] += 1
        ss.append(len(s))
    print('# gt sizes histo:', sizes)
    ss.sort()
    ss.reverse()
    print('# group sizes descending: %s\n' % ss)
    sys.stdout.flush()
    return node_gt, gt_to_nodes


#######


if __name__ == '__main__':

    gt_file = sys.argv[1]

    node_gt, gt_nodes = read_gt(gt_file)
