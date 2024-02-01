#!/usr/bin/python

# Read (de-identified) edges files (port csv format), and output
# basic statistics (number of nodes in the graph, undirected edges, directed edges,
# etc.).   Example runs given below.

# Two specific functions when the corresponding if-condition is set
# tot True:

# 1) report on directed/undirected degree, number of unique (service)
#    ports, service/provide (indegree) and client/consumer (outdegree)
#    stats, etc (see the main body and
#    report_degree_and_port_stats(...)  ) ..

# 2) longevity: report on edge longevity related stats (set the last
#         condition in main to True, and see the function
#         longevity_histograms(...)  )


# Example runs:

# Read 2 edge files from the directory:  dir_20_graphs
# python read_graphs.py dir_20_graphs 2

# final output (graph name, number of nodes, number of unidrected edges, etc):
# 
# Graph, num nodes, undirected edges, directed edges, port-differentiated-directed edges
# g2 58234 515850 578776 1629927
# g4 35613 39457 39510 39687
# g6 15927 47280 48362 124520
# ....

# Read all files
# python read_graphs.py dir_20_graphs

# Another directory
# python read_graphs.py dir_g21_small_workload_with_gt/dir_no_packets_etc/


############

import sys, gzip
from os import listdir
import numpy as np
from collections import defaultdict, Counter

######  Reading related.

# Expects each line (tab separated columns): graph-id, node1, node2,
# then csv of port info.
def read_edges_with_ports_to_stats(edges_file,
                                   wload_to_graph=None,  wload_to_port_info=None,
                                   wload_to_directed_longevity=None):
    if wload_to_graph is None:  # Otherwise, add to existing graph
        wload_to_graph = {}
    if wload_to_port_info is None:  # Otherwise, add to existing graph
        wload_to_port_info = {}
    directed_longevity = defaultdict(Counter)
    port_to_freq = Counter()
    # Assume gzip file.
    with gzip.open(edges_file, mode='rt') as fopen: # open in rt=read-text mode
        for line in fopen:
            if line.startswith('#'): # skip comment lines
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            wload_id = parts[0]
            v1 = parts[1]  # node1, consumer / client
            v2 = parts[2]
            
            if wload_id not in wload_to_graph:
                wload_to_graph[ wload_id ] = defaultdict(Counter)
            v_to_u = wload_to_graph[wload_id]
            port_info = parts[3]
            ports = port_info.split( ',' )
            stats = wload_to_port_info.get(wload_id, None)
            if stats is None:
                stats = defaultdict(set)
                wload_to_port_info[ wload_id ] = stats

            ports_added = False
            for port_tuple in ports:
                #if 0 and '-' not in port_tuple:
                #    continue
                if 'p' not in port_tuple:
                    continue
                port_part = port_tuple.split('-')[0]
                if port_part == '':
                    continue
                port_to_freq[port_part] += 1
                stats[port_part].add((v1, v2))
                ports_added = True
                directed_longevity[wload_id][(v1, v2, port_part)] = 1
                
            if ports_added:
                v_to_u[v1][v2] += 1
                v_to_u[v2][v1] += 1
        
    pairs = [(x[0], x[1]) for x in port_to_freq.items()]
    pairs.sort(key=lambda x: - x[1]) # descending
    
    #top=5
    #print('\n# top %d most common ports: %s' % (top, str(pairs[:top])))
    
    if wload_to_directed_longevity is None:
        wload_to_directed_longevity  = directed_longevity
    else:
        for wload, triples in directed_longevity.items():
            for trip in triples:
                wload_to_directed_longevity[wload][trip] += 1
    
    return wload_to_graph, wload_to_port_info, wload_to_directed_longevity

###

# Read and aggregate from possibly multiple edge files.
def read_edges_with_ports_to_stats_multiple_files(path, maxnum=None, print_it=0):
    wload_to_gr = {} # map workload or specific graph, eg 'g1', to its graph of edges
    wload_to_stats = {} # representation by ports first.
    
    # For each file, a directed port-differentiated edges appears in,
    # this goes up by 1 (persistence or longevity of an edge).
    wload_to_directed_longevity = defaultdict(Counter)
    
    print( '\n# reading multiple files.. path is:', path)
    marker = 'gz' # or 'edge' 
    names_only = [f for f in listdir(path) if marker in f   ]
    fnames = [path + '/' + f for f in listdir(path) if marker in f   ] # includes path
    if len(fnames) == 0: # go one level deeper
        dirs = [path + '/' + fdir for fdir in listdir(path) if 'dir' in f   ]
        fnames = []
        names_only = []
        i = 0
        for fdir in dirs:
            i += 1
            fnames += [fdir + '/' + f for f in listdir(fdir) if marker in f  ]
            names_only += [f for f in listdir(fdir) if marker in f  ]
            #if i >= 2: # if we stop at 2 on 20 graphs, this is day 1 and day 2
            #    break

    if len(fnames) == 0: # go one level deeper
        print("\n# No edge files found..\n")
        return {}, {}, {}
    
    fnames.sort()
    if maxnum is not None:
        fnames = fnames[:maxnum]

    if len(names_only) < 10:
        print('\n# All edge file names are:\n%s' % (names_only))
    else:
        print('\n# Some edge file names are:\n%s ...' % (names_only[:10]))

    print('\n# Num files to process: %d, 1st few are:%s\n' % (len(fnames), fnames[:10]))
    
    i = 0
    for fname in fnames:
        i += 1
        if print_it:
            print( '\n# %d. Reading from fname %s' % (i,  fname))
        sys.stdout.flush()
        wload_to_gr, wload_to_stats, wload_to_directed_longevity = read_edges_with_ports_to_stats(
            fname, wload_to_graph=wload_to_gr, wload_to_port_info=wload_to_stats,
            wload_to_directed_longevity=wload_to_directed_longevity)
        sizes = ([len(x) for x in wload_to_gr.values()])
        if print_it:
            print('\n# numgraphs read=%d, min nodes=%d, mean=%.1f, med=%.1f, max=%.1f\n' % (
                len(sizes), np.min(sizes), np.mean(sizes), np.median(sizes), np.max(sizes)))
    print('\n# Done reading from %d file(s).\n' % i)
    return wload_to_gr, wload_to_stats, wload_to_directed_longevity

#############

# Report on the number of nodes and edges (treated as undirected and
# ignoring ports, directed, and finally directed and port
# differentiated). So if there is an edge u to v and there is another
# edge v to u. Then the number of undirected is 1, the number of
# directed is 2 (both ignore ports).
def report_num_nodes_and_edges(workloads, wload_to_directed_longevity):
    if workloads is None or len(workloads) == 0:
        return
    print('\nGraph, num nodes, undirected edges, directed edges, ' +
          'port-differentiated-directed edges' )
    for w in workloads:
        num_trips = 0 # num directed edges (v1, v2, port)
        udirected = set()
        directed = set()
        outv = set()
        allv = set()
        for trip, count in wload_to_directed_longevity[w].items():
            num_trips += 1
            u, v = trip[0], trip[1]
            directed.add((u, v))
            if u <= v:
                udirected.add((u, v))
            else:
                udirected.add((v, u))
            outv.add(u)
            allv.add(u)
            allv.add(v)
        print('%s %d %d %d %d' % (
            w, len(allv), len(udirected), len(directed), num_trips))
    print('\n')

###############################
#################
# 
#  More statistics reporting.
#

def report_degree_and_port_stats(graphs, stats):
    # Sort by number of nodes..
    pairs = [(x[0], len(x[1])) for x in graphs.items()]
    pairs.sort(key=lambda x: -x[1]) # descending
    i = 0
    for wload, size in pairs:
        i += 1
        num_ports = len( stats[wload] )
        report_node_by_degree(
            stats[wload], wload, len(graphs[wload]))
            
def report_node_by_degree(
        port_to_list, wload, num_nodes=0):
    # For computing number of nodes providing one or more service
    # ports.
    provider_to = defaultdict(set)
    # Same for client of (or consumer of.
    consumer_of = defaultdict(set)
    dir_graph  = defaultdict(set) # directed graph
    undir_graph  = defaultdict(Counter) # undirected graph

    # by_provided = defaultdict(dict)
    by_consumed = defaultdict(dict)
          
    uedges = set()
    all_nodes = set()
    numports2 = 0
    for port, vlist in port_to_list.items():
        if len(vlist) > 1:
          numports2 += 1
        for v1, v2 in vlist:
          all_nodes.add(v1)
          all_nodes.add(v2)
            
          provider_to[v2].add(port)
          consumer_of[v1].add(port)

          #if port not in by_provided[v2]:
          #  by_provided[v2][port] = set()
          if port not in by_consumed[v1]:
            by_consumed[v1][port] = set()
          
          # by_provided[v2][port].add(v1)
          by_consumed[v1][port].add(v2)
          
          dir_graph[v1].add(v2) # directed graph
          if v1 < v2:
              uedges.add((v1, v2)) 
          else:
              uedges.add((v2, v1))
          undir_graph[v1][v2]=1
          undir_graph[v2][v1]=1

    numports = len(port_to_list)
    two_way = 0 # directed 2-cycles (u to v and v to u)
    nself = 0 # any self-arcs?
    for u, neibs in dir_graph.items():
        for v in neibs:
            if u == v:
                nself += 1
                continue
            if v in dir_graph:
                if u in dir_graph[v]:
                    two_way += 1

    inAndOutAtLeastOne = 0
    for node in provider_to:
        if node in consumer_of:
            inAndOutAtLeastOne += 1

    # Num directed and undirected edges
    numde = sum( [len(x[1]) for x in dir_graph.items() ]  )
    numue =  len( uedges  )  # undirected

    udegs = [len(x) for x in undir_graph.values() ]
    medd = np.median(udegs)
    maxd = np.max(udegs)
    d2 = len(list(filter(lambda x: x > 1, udegs)))
        
    nump = len(provider_to) # providing at least one port
    numc = len(consumer_of) # consuming at least one port
        
    thresh=1
    ps = [len(x) for x in provider_to.values() ]
    # ps = [x[1] for x in pairsp] # for median and max
    medp = np.median(ps)
    maxp = np.max(ps)
    nump2 = len(list(filter(lambda x: x > thresh, ps)))
        
    pairsc = [(x[0], len(x[1])) for x in consumer_of.items() ]
    cs = [x[1] for x in pairsc] # for median and max
    medc = np.median(cs)
    maxc = np.max(cs)
    numc2 = len(list(filter(lambda x: x[1] > thresh, pairsc)))

    print('\nGraph=%s num nodes=%d, num undirected edges=%d' % (wload, len(all_nodes), numue))
    print('(undirected) num nodes with degree 2+=%d, median degree=%d, max degree=%d' % (
        d2, medd, maxd))
    print('Num. of (unique service) ports in graph =%d, on 2+ edges=%d' % (numports, numports2))
    print('Num. nodes providing a port=%d, 2+ ports=%d median=%d max=%d' % (
        nump, nump2, medp, maxp))
    print('Num. nodes client of a port=%d, 2+ ports=%d median=%d max=%d' % (
        numc, numc2, medc, maxc))
    print('Num. nodes with positive indegree and outdegree = %d' % (inAndOutAtLeastOne))
    print('Num. of self-arcs: %d' % (nself))
    print('Num. directed edges: %d' % numde)
    print('Num. of directed 2-cycles: %d\n' % (two_way))


####################

# Explore longevity or persistence, or short (eg, occurring in one edge file only) vs
# long-lived edges (conversations or triples).  May output  in
# Latex table format! ..  (number of edges seen once, twice, and in all
# or max-count files).
#
# NOTE: need to run it on at least 3 edge files (as written!)
#
# NOTE 2: you can provide a count_thresh (3rd parameter) and count and
# proportion of conversations above this is reported.
#
def longevity_histograms(workloads, v_to_directed_longevity, count_thresh=None):
    for w in workloads:
        num_trips = 0 # number of port-differentiated directed edges.
        cnt_to_cnt = Counter()
        cnt_to_ports = defaultdict(Counter)
        cnt_to_servers = defaultdict(Counter)
        cnt_to_clients = defaultdict(Counter)
        max_cnt = 0
        cnt_exceeds = 0 # count those above given count_thresh
        for trip, count in v_to_directed_longevity[w].items():
            num_trips += 1
            cnt_to_cnt[count] += 1
            port_part = trip[2]
            cnt_to_ports[count][port_part] += 1
            cnt_to_servers[count][trip[1]] += 1
            cnt_to_clients[count][trip[0]] += 1
            max_cnt = max(max_cnt, count)
            if count_thresh is not None:
                if count >= count_thresh:
                    cnt_exceeds += 1

        print('\n# Max count (of most seen edge) is: %d\n' % max_cnt)
        pairs = list(cnt_to_cnt.items())
        # descending on first
        pairs.sort(key=lambda x: - x[0])        
        if 0: # plain print the contents
            print(w, num_trips, pairs)
        else:
            # Insert number of directed port-differentiated edges
            # (triples).
            strw = w + ' & %d ' % num_trips

            cnt = 0 # seen once
            r = 0
            if pairs[-1][0] == 1:
                cnt = pairs[-1][1]
                r = int(100.0 * cnt / num_trips)
            # strw = strw + "& %d, %d\\%s " % (cnt, r, '%')
            # Percentage seen once.
            strw = strw + "& %d\\%s " % (r, '%') # latex table format!

            cnt = 0 # seen twice
            r = 0
            if pairs[-2][0] == 2:
                cnt = pairs[-2][1]
                r = int(100.0 * cnt / num_trips)
            if r > 0:
                strw = strw + "& %d\\%s " % (r, '%') # latex table format!
            else:
                strw = strw + "& %d\\%s (%d) " % (r, '%', cnt) # latex table format!
                

            cnt = 0
            r = 0
            if pairs[0][0] == max_cnt: # seen in all max_cnt times
                cnt = pairs[0][1]
                r = int(100.0 * cnt / num_trips)
            strw = strw + "& %d, %d\\%s " % (cnt, r, '%') # latex table format!

            # num unique ports with highest persistent edge count
            num_ports = len(cnt_to_ports[max_cnt])
            # num unique server nodes and client nodes associated with
            # highest count edges.
            num_servers = len(cnt_to_servers[max_cnt])
            num_clients = len(cnt_to_clients[max_cnt])
            strw = strw + "& %d, %d, %d " % (
                num_ports, num_servers, num_clients)
            
            strw = strw + "\\\\ \\hline "
            print(strw)

            if count_thresh is not None:
                # For 
                r = int(100.0 * cnt_exceeds / num_trips)
                print('# Count and percentage of edges (conversations) >= %d: %d or %.3f\%' % (
                    count_thresh, cnt_exceeds, r))

            
###

if __name__ == '__main__':

    # path to edges file(s)
    path = sys.argv[1]
    path += '/'
    maxnum = None # read all files if None..
    if len(sys.argv) > 2:
        maxnum = int(sys.argv[2]) # read maxnum many edge files.
    graphs, stats, wload_to_directed_longevity = read_edges_with_ports_to_stats_multiple_files(
        path, maxnum=maxnum )

    pairs = [(x[0], len(x[1])) for x in graphs.items()]
    # Sort in descending number of nodes.
    pairs.sort(key=lambda x: - x[1])
    workloads = [x[0] for x in pairs] # They are ordered now.
    report_num_nodes_and_edges(workloads, wload_to_directed_longevity)


    if 1:
        report_degree_and_port_stats(graphs, stats)
    
    if 0:  # Explore longevity of edges? (how many edges files each observed edge
           # falls into).
        longevity_histograms(workloads, wload_to_directed_longevity, count_thresh=50)
