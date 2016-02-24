
import numpy as np
import pylab

from scipy.stats import cumfreq
from pandas      import DataFrame
from ipy_table   import *
from lnm         import sanitize

def read_raw_data(fname):
    variations = np.genfromtxt(fname, usecols=(0,), dtype=None)
    times = np.genfromtxt(fname, usecols=(1,2,3), dtype='d')
    keys = sanitize(variations)
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
    lst = [["", "Max overhead", "Mean overhead", "Median overhead", "300-deliverable", "300/1000-usable"]]
    for name, data in kwargs.iteritems():
        lst.append([name] + compute_deliverable(data))
    tbl = make_table(map(list, zip(*lst)))
    apply_theme('basic_both')
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
            pylab.plot(bin_edges[:-1], cdf, label=label)
        else:
            pylab.plot(bin_edges[:-1], cdf, color=color, label=label)
    fname = "slowdown_{name}.pdf".format(**kwargs)
    pylab.axvline(3, color='y')
    pylab.axvline(10, color='k')
    pylab.axhline(entries * 0.7, color='c', ls='--')
    pylab.xlabel('slowdown factor')
    pylab.xlim((1,10))
    pylab.ylim((0,entries))
    pylab.ylabel('number below')
    pylab.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    pylab.savefig(fname)

def compute_lnm_deliverable(slowdown_graph, L):
    graph = lnm.compute_lnm_times(slowdown_graph, L)
    data = graph.ungraph()[1]
    rs = data[:,0]
    ps = data[:,1]
    hs = data[:,2]
    return rs, ps, hs, make_deliverable_table(racket=rs, pycket=ps,hidden=hs)

