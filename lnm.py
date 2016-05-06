
import re
import numpy as np
from graph import Graph, immutable_array

def adjacent_variations(var):
    """ Given a variation represented as a bit string, return all variations
        one step above in the lattice """
    idx = (var == 0).view(np.ndarray)
    I = np.identity(len(var), dtype='b')
    data = var + I[idx,:]
    return data

class LNM(object):
    def __init__(self, graph):
        self.graph = graph

    def _compute_lnm_time(self, variation, L):
        graph = self.graph
        reachable = (p.payload for p in graph.within_distance(variation, L))
        return reduce(np.minimum, reachable)

    def compute_lnm_graph(self, L):
        graph = self.graph
        res = ((key, self._compute_lnm_time(key, L)) for key in graph.iterkeys())
        return Graph(res, adjacent_variations)

def compute_lnm_times(graph, L=0):
    return LNM(graph).compute_lnm_graph(L)

def sanitize(variations):
    result = []
    for var in variations:
        clean = re.sub("[^01]", "", var)
        result.append(immutable_array(map(int, clean), dtype='b'))
    return result

def fromkeyvals_transpose(keys, *args):
    args = immutable_array(zip(*args))
    return Graph.fromkeyvals(keys, args, adjacent_variations)

def fromkeyvals(keys, args):
    return Graph.fromkeyvals(keys, args, adjacent_variations)

def read_data(fname):
    variations = np.genfromtxt(fname, usecols=(0,), dtype=None)
    times = np.genfromtxt(fname, usecols=(1,2,3), dtype='d')
    keys = sanitize(variations)
    return Graph.fromkeyvals(keys, times, adjacent_variations)

# if __name__ == '__main__':
    # data = read_data('results_tetris.txt')
    # graph1 = compute_lnm_times(data, L=1)
    # graph2 = compute_lnm_times(data, L=2)
    # assert set(graph1.keys()) == set(graph2.keys()) and len(graph1) == len(graph2)
    # for key in graph1.iterkeys():
        # print graph1[key] - graph2[key]
        # # assert np.all(graph1[key] == graph2[key])

# variation = np.genfromtxt("results_tetris.txt", usecols=(0,), dtype=None)
# time = np.genfromtxt("results_tetris.txt", usecols=(1,2,3), dtype='i')

# print variation
# print time
