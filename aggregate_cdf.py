
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

COLORS = [(255.0 / 255.0, 90 / 255.0, 20 / 255.0), (36 / 255.0, 36 / 255.0, 140 / 255.0), (34 / 255.0, 139 / 255.0, 34 / 255.0), (218 / 255.0, 165 / 255.0, 32 / 255.0)]
LABELS = ['racket', 'pycket', 'baseline', 'no-callgraph']
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
    for idx, i in enumerate(keys):
        if keys[0] != i:
            print files[idx]
            raise ValueError("inconsistent data files")

    means     = np.mean(times, axis=0)
    variances = np.var(times, axis=0)
    return Data(keys[0], np.array(times), means, variances)

def pad_weights(weights, arrs):
    needed = max(*[s.shape[-1] for s in arrs])
    new_arrs, new_weights = [], []

    for weight, arr in zip(weights, arrs):
        have = arr.shape[-1]
        need = needed - have
        pad  = np.ones((arr.shape[0], need)) * -1
        arr  = np.append(arr, pad, axis=1)
        new_arrs.append(arr)

        weight = np.repeat([weight], have, axis=0).T
        pad = np.zeros((arr.shape[0], need))
        weight = np.append(weight, pad, axis=1)
        new_weights.append(weight)

    return new_weights, new_arrs

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

        largest_dims = max(*[s.shape[-1] for s in slowdowns])

        _, slowdowns = pad_weights(weights, slowdowns)
        _, slowdowns1 = pad_weights(weights, slowdowns1)
        weights, slowdowns2 = pad_weights(weights, slowdowns2)

        weights   = np.vstack(weights)
        all_data  = np.vstack(slowdowns)
        all_data1 = np.vstack(slowdowns1)
        all_data2 = np.vstack(slowdowns2)

        N = all_data.shape[-1]
        for i in range(2):
            entries = np.sum(weights[:,i])
            result = all_data[:,i]
            counts, bin_edges = np.histogram(result, bins=len(result), weights=weights[:,i])
            cdf = np.cumsum(counts) / np.sum(entries) * 100.0
            ax.plot(bin_edges[:-1], cdf, LINESTYLES[number], label=LABELS[i], color=COLORS[i])

        # add lines for L=1
        result = all_data1[:,i]
        counts, bin_edges = np.histogram(result, bins=len(result), weights=weights[:,1])
        cdf = np.cumsum(counts) / entries * 100.0
        ax.plot(bin_edges[:-1], cdf, LINESTYLES[number], label=LABELS[i], color=(0,0,0))

        total_benchmarks = np.sum(weights, axis=0)
        avg_slowdown_weighted  = np.sum(weights * all_data, axis=0)  / total_benchmarks
        avg_slowdown_weighted1 = np.sum(weights * all_data1, axis=0) / total_benchmarks
        s3 = np.sum(weights * (all_data < 3.0)  , axis=0) * 100.0  / total_benchmarks
        s4 = np.sum(weights * (all_data1 < 3.0) , axis=0) * 100.0  / total_benchmarks
        s5 = np.sum(weights * (all_data < 1.1)  , axis=0) * 100.0  / total_benchmarks
        s6 = np.sum(weights * (all_data1 < 1.1) , axis=0) * 100.0  / total_benchmarks
        s7 = np.sum(weights * (all_data2 < 1.1) , axis=0) * 100.0  / total_benchmarks

        def rnd(x):
            return round(x, 0)

        if number != 0:
            print "\multicolumn{8}{|c|}{%s} \\\\" % SUFFIXES[number]
            print "\\hline"
        for i in range(len(avg_slowdown_weighted)):
            s1 = round(avg_slowdown_weighted[i], 2)
            s2 = round(avg_slowdown_weighted1[i], 2)
            print "%s & $%0.2f\\times$ & $%0.2f\\times$ & $%0.0f$ & $%0.0f$ & $%0.0f$ & $%0.0f$ & $%0.0f$ \\\\" % ((LABELS[i].capitalize(), s1, s2) + tuple(map(rnd, (s3[i], s4[i], s5[i], s6[i], s7[i]))))
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
        weights   = [np.ones(d.means.shape[0]) / d.means.shape[0] for d in data]
        slowdowns = [d.means / d.means[0,0] for d in data]

        weights, slowdowns = pad_weights(weights, slowdowns)
        weights  = np.vstack(weights)
        all_data = np.vstack(slowdowns)


        for i in range(0, 2):
            ax.scatter(all_data[:,0], all_data[:,i], label=LABELS[i], color=COLORS[i], marker='.')
            if i == 0:
                continue
            m, b = np.polyfit(all_data[:,0], all_data[:,i], 1)
            x = np.vstack([np.arange(0, 100, 0.01), np.ones(10000)]).T
            print "y = %f * x + %f" % (m, b)
            y = m * x + b
            ax.plot(x, y, color=COLORS[i+2])
            textX = np.max(all_data[:,0]) / 2.0
            textY = np.max(all_data[:,i]) / 1.8
            plt.text(textX, textY, '$y = %0.3f x + %0.3f$' % (m, b), fontsize=20,
                     color='k',
                     horizontalalignment='center',
                     verticalalignment='bottom')


        plt.legend(loc='upper left')
        plt.ylim((0, 70))
        plt.xlim((0, 70))
        ax.set_xlabel("Racket gradual typing overhead")
        ax.set_ylabel("overhead relative to Racket")
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
