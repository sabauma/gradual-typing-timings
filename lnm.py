
import numpy as np

def adjacent_variations(variation):
    """ Given a variation represented as a bit string, return all variations
        one step above in the lattice """
    var = np.array(variation, dtype='b')
    I = np.identity(len(var), dtype='b')
    swaps = I[var == 0,:]
    yield variation
    for i in range(swaps.shape[0]):
        yield tuple(variation + swaps[i,:])

def compute_lnm_time(variation, graph, L, entry=0):
    """ variation: tuple bit string of current variation
        graph:     dict mapping variation to timings
        L:         integer value """
    if L == 0:
        return graph[variation][entry]
    l = L - 1
    return min((compute_lnm_time(var, graph, l) for var in adjacent_variations(variation)))

class LNM(object):
    def __init__(self, graph):
        self.graph = graph
        self.cache = {}

    def compute_lnm_time(self, variation, L, entry=0):
        """ variation: tuple bit string of current variation
            graph:     dict mapping variation to timings
            L:         integer value """
        if L == 0:
            return self.graph[variation][entry]
        key = (variation, L, entry)
        lup = self.cache.get(key, None)
        if lup is not None:
            return lup
        l = L - 1
        result = min((self.compute_lnm_time(var, l, entry=entry)
                     for var in adjacent_variations(variation)))
        self.cache[key] = result
        return result

def sanitize(variations):
    return [tuple(map(lambda x: bool(int(x)), var[9:])) for var in variations]

def read_data(fname):
    variations = np.genfromtxt(fname, usecols=(0,), dtype=None)
    times = np.genfromtxt(fname, usecols=(1,2,3), dtype='d')
    keys = sanitize(variations)
    return dict(zip(keys, times))

def compute_lnm_times(graph, L=0):
    lnm = LNM(graph)
    new_graph = {}
    for key, entries in graph.iteritems():
        new_times = [None] * len(entries)
        for entry in range(len(entries)):
            new_times[entry] = lnm.compute_lnm_time(key, L, entry)
        new_graph[key] = np.array(new_times)
    return new_graph

def mkgraph(keys, vals):
    return dict(zip(keys, vals))

def ungraph(graph):
    keys = sorted(graph.keys())
    return (keys, np.array([graph[key] for key in keys]))

if __name__ == '__main__':
    data = read_data('results_tetris.txt')
    graph1 = compute_lnm_times(data, L=1)
    graph2 = compute_lnm_times(data, L=2)
    assert set(graph1.keys()) == set(graph2.keys()) and len(graph1) == len(graph2)
    for key in graph1.iterkeys():
        print graph1[key] - graph2[key]
        # assert np.all(graph1[key] == graph2[key])

# variation = np.genfromtxt("results_tetris.txt", usecols=(0,), dtype=None)
# time = np.genfromtxt("results_tetris.txt", usecols=(1,2,3), dtype='i')

# print variation
# print time