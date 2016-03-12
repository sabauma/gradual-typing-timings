
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
mpl.rc('figure', autolayout=True)

Data = namedtuple('Data', 'names times means variances')

COLORS = ['red', 'green', 'blue', 'yellow', 'orange']
LABELS = ['racket', 'pycket', 'hidden']

def print_help():
    pass

def read_data_files(pattern):
    files = glob.glob(pattern)
    # print "processing {} file(s)".format(len(files))

    if not files:
        raise ValueError("cannot find any matching files: %s" % pattern)

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

    entries = len(data)
    plt.axvline(3, color='y')
    plt.axvline(10, color='k')
    plt.xlim((1,10))

    ax.set_xlabel("slowdown factor")
    ax.set_ylabel("# of benchmarks")
    ax.set_xticklabels(["%dx" % (i + 1) for i in range(10)])
    plt.ylim((0, entries))
    plt.savefig("figs/aggregate-cdf.pdf")
    plt.cla()

    avg_slowdown_weighted = np.dot(weights, all_data) / float(entries)
    avg_slowdown_unweighted = np.mean(all_data, axis=0)
    for i in range(len(avg_slowdown_weighted)):
        s1 = round(avg_slowdown_weighted[i], 1)
        s2 = round(avg_slowdown_unweighted[i], 1)
        print "%s & $%0.1f\\times$ & $%0.1f\\times$ \\\\" % (LABELS[i].capitalize(), s1, s2)

    all_racket = reduce(np.append, [d.means[:,0] for d in data])
    for i in range(1, N):
        ax.scatter(all_data[:,0] / all_data[0,0], all_data[:,i] / all_data[:,0], color=COLORS[i], label=LABELS[i])


    max = int(round(np.max(all_data[:,0] / all_data[0,0]) / 10.0, 0) * 10)

    perfect = np.arange(0.0, max, 0.01)
    ax.plot(perfect, 1.0 / perfect, color='c')

    ax.set_xticks(range(0, 10, 1) + range(10, max + 10, 10))
    ax.set_xticklabels([0] + ['' for i in range(9)] + range(10, max + 10, 10))
    ax.axhline(1.0)
    plt.ylim((0, 2))
    plt.xlim((0, max))
    ax.legend(loc='best')
    ax.set_xlabel("Racket gradual typing overhead")
    ax.set_ylabel("Runtime relative to Racket")
    plt.savefig("figs/aggregate-slowdown.pdf")

if __name__ == '__main__':
    slowdown_cdf([read_data_files(g) for g in sys.argv[1:]])
