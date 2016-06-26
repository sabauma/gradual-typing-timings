
import numpy as np
from collections import OrderedDict

def immutable_array(data, *args, **kwargs):
    array = np.array(data, *args, **kwargs)
    return array.view(immutable_ndarray)

class immutable_ndarray(np.ndarray):

    def __array_finalize__(self, obj):
        self.setflags(write=False)

    def __nonzero__(self):
        """ This is necessarry to use immutable_ndarray as dict keys. Kinda gross """
        return all(self)

    def __hash__(self):
        return hash(self.data)

class Node(object):
    __slots__ = ('name', 'adjacent', 'payload')
    def __init__(self, name, adjacent, payload):
        self.name     = name
        self.adjacent = adjacent
        self.payload  = payload

    def within_distance(self, distance):
        todo = {self}
        for _ in xrange(distance):
            todo = todo.union(neighbor for node in todo for neighbor in node.adjacent)
        return list(todo)

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return "Node(%r, %r, %r)" % (self.name, self.adjacent, self.payload)

class NodeCounter(object):
    def __init__(self):
        self.counter = 0
        self.cache   = {}

    def getnid(self, node):
        nid = self.cache.get(node, None)
        if nid is None:
            self.cache[node] = nid = self.counter
            self.counter += 1
        return nid

class Graph(object):

    def __init__(self, graph):
        assert isinstance(graph, OrderedDict)
        self.graph = graph

    def traverse(self, func):
        newgraph = Graph.memo_nodes()
        for key, node in self.graph.iteritems():
            newnode = newgraph(key)
            newnode.payload = func(node)
            newnode.adjacent = [newgraph(n.name) for n in node.adjacent]
        return Graph(newgraph.memo_table())

    def within_distance(self, node, distance):
        return self.graph[node].within_distance(distance)

    def ungraph(self):
        """
        Produce the new data set in the order given by the initial input data.
        """
        graph = self.graph
        keys  = graph.keys()
        return (keys, immutable_array([graph[key].payload for key in keys]))

    def networkx_graph(self):
        import networkx as nx
        g = nx.DiGraph()
        counter = NodeCounter()
        for node in self.graph.itervalues():
            nid = counter.getnid(node)
            g.add_node(nid)
            for neighbor in node.adjacent:
                nextid = counter.getnid(neighbor)
                g.add_edge(nid, nextid)
        return g

    def iterkeys(self):
        return self.graph.iterkeys()

    @staticmethod
    def memo_nodes(storage=None):
        if storage is None:
            storage = OrderedDict()
        def func(key):
            node = storage.get(key, None)
            if node is None:
                node = Node(key, None, None)
                storage[key] = node
            return node
        func.storage = storage
        func.memo_table = lambda: storage.copy()
        return func

    @staticmethod
    def fromfunc(keyvals, adjacent):
        memo = Graph.memo_nodes()
        for k, v in keyvals:
            node = memo(k)
            node.payload  = v
            node.adjacent = map(memo, adjacent(k))
        return Graph(memo.memo_table())

    @staticmethod
    def fromkeyvals(keys, vals, adjacent):
        from itertools import izip
        return Graph.fromfunc(izip(keys, vals), adjacent)

ex = {'00': 1, '11': 2, '01': 3, '10': 4}

# if __name__ == '__main__':
    # import networkx as nx
    # g = Graph.fromdict(ex)
    # print nx.draw(g.display())

