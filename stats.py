
import lnm
import numpy as np
import matplotlib.pyplot as plt

def read_raw_data(fname):
    try:
        variations = np.genfromtxt(fname, usecols=(0,), dtype=None)
        with open(fname, 'r') as infile:
            columns = (" ".join(line.split()[1:]) for line in infile)
            times   = np.genfromtxt(columns, dtype='d')
    except ValueError:
        print "failed on: ", fname
    else:
        keys = lnm.sanitize(variations)
        # In the case of a single column, the resulting array will be one dimensional
        # due to genfromtxt. In that case we pad on the extra dimension.
        if len(times.shape) == 1:
            times = times[:,np.newaxis]
        return keys, times

def make_slowdown_data(fname):
    data = np.genfromtxt(fname, usecols=(1,2,3))
    hidden_untyped = float(data[0,2])
    pycket_untyped = float(data[0,1])
    racket_untyped = float(data[0,0])
    hidden = data[:,2]
    pycket = data[:,1]
    racket = data[:,0]
    weights = np.ones(len(hidden)) / float(len(hidden))
    return racket / racket_untyped, pycket / pycket_untyped, hidden / hidden_untyped, weights

def compute_deliverable(data):
    th = sum(data < 3)
    oh = sum(np.logical_and(data >= 3, data < 10))
    return [max(data), np.mean(data), np.median(data), "%d (%0.2f%%)" % (th, th / float(len(data)) * 100), "%d (%0.2f%%)" % (oh, oh / float(len(data)) * 100)]

def make_deliverable_table(**kwargs):
    # from ipy_table   import *
    import ipy_table
    lst = [["", "Max overhead", "Mean overhead", "Median overhead", "300-deliverable", "300/1000-usable"]]
    for name, data in kwargs.iteritems():
        lst.append([name] + compute_deliverable(data))
    tbl = ipy_table.make_table(map(list, zip(*lst)))
    ipy_table.apply_theme('basic_both')
    return tbl

def slowdown_cdf(*args, **kwargs):
    weights = kwargs.get('weights', None)
    many_weights = kwargs.get('many_weights', None)
    entries = 0

    for i, (data, color, label) in enumerate(args):
        if weights is not None:
            counts, bin_edges = np.histogram(data, weights=weights, bins=len(data))
            entries = np.sum(weights)
        elif many_weights is not None:
            counts, bin_edges = np.histogram(data, weights=many_weights[i], bins=len(data))
            entries = np.sum(many_weights[i])
        else:
            counts, bin_edges = np.histogram(data, bins=len(data))
            entries = len(data)
        cdf = np.cumsum(counts)
        if not color:
            plt.plot(bin_edges[:-1], cdf, label=label)
        else:
            plt.plot(bin_edges[:-1], cdf, color=color, label=label)
    fname = "slowdown_{name}.pdf".format(**kwargs)
    plt.axvline(3, color='y')
    plt.axvline(10, color='k')
    plt.axhline(entries * 0.7, color='c', ls='--')
    plt.xlabel('slowdown factor')
    plt.xlim((1,10))
    plt.ylim((0,entries))
    plt.ylabel('number below')
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.savefig(fname)

def compute_lnm_deliverable(slowdown_graph, L):
    graph = lnm.compute_lnm_times(slowdown_graph, L)
    data = graph.ungraph()[1]
    rs = data[:,0]
    ps = data[:,1]
    hs = data[:,2]
    return rs, ps, hs, make_deliverable_table(racket=rs, pycket=ps,hidden=hs)

