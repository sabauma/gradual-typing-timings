
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

COLORS = [(255.0 / 255.0, 90 / 255.0, 20 / 255.0), (34 / 255.0, 139 / 255.0, 34 / 255.0), (36 / 255.0, 36 / 255.0, 140 / 255.0), (218 / 255.0, 165 / 255.0, 32 / 255.0)]
LABELS = ['racket', 'baseline', 'pycket']
LINESTYLES = ['-', '--', ':']
# MARKERS = ['s', 'o', 'o']
SUFFIXES = ['Racket 6.5.0.6', 'Racket 6.2.1']

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

def slowdown_cdf(datas):

    fig, ax = plt.subplots(nrows=1, ncols=1)
    for number, data in enumerate(datas):
        weights   = [np.array([1.0 / float(d.means.shape[0])] * d.means.shape[0]) for d in data]
        slowdowns = [d.means / d.means[0,:] for d in data]

        graphs = [lnm.fromkeyvals(d.names, slowdown) for d, slowdown in zip(data, slowdowns)]
        graphs = [lnm.compute_lnm_times(g, L=1) for g in graphs]
        slowdowns1 = [g.ungraph()[1] for g in graphs]

        graphs = [lnm.fromkeyvals(d.names, slowdown) for d, slowdown in zip(data, slowdowns)]
        graphs = [lnm.compute_lnm_times(g, L=2) for g in graphs]
        slowdowns2 = [g.ungraph()[1] for g in graphs]

        weights   = reduce(np.append, weights)
        all_data  = reduce(lambda a, b: np.append(a, b, axis=0), slowdowns)
        all_data1 = reduce(lambda a, b: np.append(a, b, axis=0), slowdowns1)
        all_data2 = reduce(lambda a, b: np.append(a, b, axis=0), slowdowns2)

        entries = len(data)
        N = all_data.shape[-1]
        for i in range(N):
            if number == 0 and i == 2:
                result = all_data1[:,i]
                counts, bin_edges = np.histogram(result, bins=len(result), weights=weights)
                cdf = np.cumsum(counts) / float(entries) * 100.0
                ax.plot(bin_edges[:-1], cdf, LINESTYLES[number], label=LABELS[i], color=(0,0,0))
            elif i == 1:
                continue
            result = all_data[:,i]
            counts, bin_edges = np.histogram(result, bins=len(result), weights=weights)
            cdf = np.cumsum(counts) / float(entries) * 100.0
            ax.plot(bin_edges[:-1], cdf, LINESTYLES[number], label=LABELS[i], color=COLORS[i])

        avg_slowdown_weighted  = np.dot(weights, all_data) / float(entries)
        avg_slowdown_weighted1 = np.dot(weights, all_data1) / float(entries)
        s3 = np.dot(weights, all_data < 3.0) * 100.0  / float(np.sum(weights))
        s4 = np.dot(weights, all_data1 < 3.0) * 100.0 / float(np.sum(weights))
        s5 = np.dot(weights, all_data < 1.1) * 100.0  / float(np.sum(weights))
        s6 = np.dot(weights, all_data1 < 1.1) * 100.0 / float(np.sum(weights))
        s7 = np.dot(weights, all_data2 < 1.1) * 100.0 / float(np.sum(weights))

        def rnd(x):
            return round(x, 0)

        if number != 0:
            print "\multicolumn{8}{|c|}{%s} \\\\" % SUFFIXES[number]
            print "\\hline"
        for i in reversed(range(len(avg_slowdown_weighted))):
            if i == 1:
                continue
            s1 = round(avg_slowdown_weighted[i], 1)
            s2 = round(avg_slowdown_weighted1[i], 1)
            print "%s & $%0.1f\\times$ & $%0.1f\\times$ & $%0.0f$ & $%0.0f$ & $%0.0f$ & $%0.0f$ & $%0.0f$ \\\\" % ((LABELS[i].capitalize(), s1, s2) + tuple(map(rnd, (s3[i], s4[i], s5[i], s6[i], s7[i]))))
        print "\\hline"

    plt.axvline(3, color=COLORS[-1])
    plt.xlim((1,10))

    ax.set_xlabel("slowdown factor")
    ax.set_ylabel("% of benchmarks")
    ax.set_xticklabels(["%dx" % (i + 1) for i in range(10)])
    plt.ylim((0, 100))
    plt.savefig("figs/aggregate-cdf.pdf")

    for number, data in enumerate(datas):
        plt.cla()
        weights   = [np.array([1.0 / float(d.means.shape[0])] * d.means.shape[0]) for d in data]
        slowdowns = [d.means / d.means[0,:] for d in data]

        weights  = reduce(np.append, weights)
        all_data = reduce(lambda a, b: np.append(a, b, axis=0), slowdowns)

        all_racket = reduce(np.append, [d.means[:,0] for d in data])
        for i in range(2, N):
            ax.scatter(all_data[:,0] / all_data[0,0], all_data[:,i] / all_data[:,0], color=COLORS[i], label=LABELS[i])


        max = int(round(np.max(all_data[:,0] / all_data[0,0]) / 10.0, 0) * 10)

        perfect = np.arange(0.0, max, 0.001)[1:]
        ax.plot(perfect, 1.0 / perfect, color=COLORS[-1])

        if max > 100:
            skip = 20
        else:
            skip = 10
        ax.set_xticks(range(0, 10, 1) + range(10, max + 10, skip))
        ax.set_xticklabels([0] + ['' for i in range(9)] + range(10, max + 10, skip))
        ax.axhline(1.0, color=COLORS[0])
        plt.ylim((0, 2))
        plt.xlim((0, max))
        ax.legend(loc='best')
        ax.set_xlabel("Racket gradual typing overhead")
        ax.set_ylabel("Runtime relative to Racket")
        plt.savefig("figs/aggregate-slowdown-%d.pdf" % number)

if __name__ == '__main__':
    args = sys.argv[1:]
    outer = [[]]
    for arg in args:
        if arg == '--':
            outer.append([])
            continue
        outer[-1].append(read_data_files(arg))
    slowdown_cdf(outer)
