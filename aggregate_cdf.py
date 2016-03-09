
from collections import namedtuple
from itertools   import izip

import glob
import argparse
import lnm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import operator as op
import os
import stats
import sys

import matplotlib as mpl
mpl.rc('lines', linewidth=3, color='r')
mpl.rc('font', family='Arial', size=22)

Data = namedtuple('Data', 'names times means variances')

COLORS = ['red', 'green', 'blue', 'yellow', 'orange']
LABELS = ['racket', 'pycket', 'hidden']

def print_help():
    pass

def read_data_files(pattern):
    files = glob.glob(pattern)
    print "processing {} file(s)".format(len(files))

    if not files:
        raise ValueError("cannot find any matching files")

    keys, times = zip(*[stats.read_raw_data(fname) for fname in files])
    for i in keys:
        if keys[0] != i:
            raise ValueError("inconsistent data files")

    means     = np.mean(times, axis=0)
    variances = np.var(times, axis=0)
    return Data(keys[0], np.array(times), means, variances)

def slowdown_cdf(data):

    weights   = [np.array([1.0 / float(d.means.shape[0])] * d.means.shape[0]) for d in data]
    slowdowns = [d.means / d.means[0,:] for d in data]

    weights  = reduce(np.append, weights)
    all_data = reduce(lambda a, b: np.append(a, b, axis=0), slowdowns)

    fig, ax = plt.subplots(nrows=1, ncols=1)

    N = all_data.shape[-1]
    for i in range(N):
        result = all_data[:,i]
        counts, bin_edges = np.histogram(result, bins=len(result), weights=weights)
        cdf = np.cumsum(counts)
        ax.plot(bin_edges[:-1], cdf, label=LABELS[i], color=COLORS[i])


    entries = np.sum(weights)
    plt.axvline(3, color='y')
    plt.axvline(10, color='k')
    plt.axhline(int(0.6 * entries), color='c', ls='--')
    plt.xlim((1,10))
    ax.set_xticklabels(["%dx" % (i + 1) for i in range(10)])
    # plt.ylim((0, entries))
    plt.savefig("aggregate-cdf.pdf")

    avg_slowdown = np.dot(all_data.T, weights) / np.sum(weights)
    print "Weighted Average slowdown: ", ", ".join(["%s=%f" % (LABELS[i], avg_slowdown[i]) for i in range(len(avg_slowdown))])
    avg_slowdown = np.mean(all_data, axis=0)
    print "Unweighted Average slowdown: ", ", ".join(["%s=%f" % (LABELS[i], avg_slowdown[i]) for i in range(len(avg_slowdown))])

if __name__ == '__main__':
    slowdown_cdf([read_data_files(g) for g in sys.argv[1:]])
