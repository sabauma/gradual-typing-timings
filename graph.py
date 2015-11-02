
import numpy as np

class Node(object):
    __slots__ = ('name', 'adjacent', 'payload')
    def __init__(self, name, adjacent, payload):
        self.name     = name
        self.adjacent = adjacent
        self.payload  = payload

    def within_distance(self, distance):
        todo = {self}
        for _ in xrange(distance):
            todo = todo.union((neighbor for node in todo for neighbor in node.adjacent))
        return list(todo)

    def __str__(self):
        return self.name

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
    def __init__(self, keyvals, adjacent):
        self.graph = {}
        for k, v in keyvals:
            node = self.get_cached_node(k)
            node.payload = v
            node.adjacent = map(self.get_cached_node, adjacent(k))

    def get_cached_node(self, nodeid):
        node = self.graph.get(nodeid, None)
        if node is None:
            node = Node(nodeid, None, None)
            self.graph[nodeid] = node
        return node

    def within_distance(self, node, distance):
        return self.graph[node].within_distance(distance)

    def ungraph(self):
        graph = self.graph
        keys = sorted(graph.keys())
        return (keys, np.array([graph[key].payload for key in keys]))

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
    def fromiter(it):
        return Graph(it)

    @staticmethod
    def fromkeyvals(keys, vals, adjacent):
        from itertools import izip
        return Graph(izip(keys, vals), adjacent)

    @staticmethod
    def fromdict(dict):
        return Graph(dict.iteritems())

ex = {'00': 1, '11': 2, '01': 3, '10': 4}

if __name__ == '__main__':
    import networkx as nx
    g = Graph.fromdict(ex)
    print nx.draw(g.display())

