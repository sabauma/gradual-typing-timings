
from collections import namedtuple
from itertools   import izip

import argparse
import glob
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

COLORS = [(255.0 / 255.0, 90.0 / 255.0, 20.0 / 255.0), (255.0 / 255.0, 69.0 / 255.0, 0.0 / 255.0), (36.0 / 255.0, 36.0 / 255.0, 140.0 / 255.0), (218.0 / 255.0, 165.0 / 255.0, 32.0 / 255.0)]
LABELS = ['racket', 'baseline', 'pycket']
LINESTYLES = ['-', '--', ':']

parser = argparse.ArgumentParser(description="Plot some things")
parser.add_argument('action', help="what plot to generate")
parser.add_argument('data', nargs='+', help="data files to process", type=str)
parser.add_argument('--output', default=None, nargs=1, type=str)
parser.add_argument('--args', nargs=argparse.REMAINDER)

def print_help():
    pass

def read_data_files(pattern):
    files = glob.glob(pattern)
    # print "processing {} file(s)".format(len(files))

    if not files:
        raise ValueError("cannot find any matching files: " % pattern)

    keys, times = zip(*[stats.read_raw_data(fname) for fname in files])
    for i in keys:
        if keys[0] != i:
            raise ValueError("inconsistent data files")

    means     = np.mean(times, axis=0)
    variances = np.var(times, axis=0)
    return Data(keys[0], np.array(times), means, variances)

def slowdown_stats(slowdowns):
    pass

def stats_table(args, datas):
    data = datas[0]

    slowdowns = data.means / data.means[0,:]

    N = slowdowns.shape[0]
    max = np.max(slowdowns, axis=0)
    mean = np.mean(slowdowns, axis=0)
    ratio = slowdowns[-1,:]
    acceptable = np.sum(slowdowns < 3.0, axis=0) / float(N) * 100.0

    print "%d &" % N,
    print " & ".join(["$ %0.1f $ & $ %0.1f $ & $ %0.1f $ & $ %0.0f $" % (ratio[i], max[i], mean[i], acceptable[i]) for i in (0, 2)]),
    print "\\\\"

def slowdown_cdf(args, datas):
    L = int(args[0]) if args else 0

    fig, ax = plt.subplots(nrows=1, ncols=1)
    for number, data in enumerate(datas):
        means = data.means
        slowdowns = means / means[0,:]
        graph = lnm.fromkeyvals(data.names, slowdowns)
        graph = lnm.compute_lnm_times(graph, L)

        results = graph.ungraph()[1]
        results = zip(*results)
        entries = means.shape[0]

        for i, result in enumerate(results):
            if i == 1:
                continue
            counts, bin_edges = np.histogram(result, bins=max(entries, 1024))
            counts = counts * (100.0 / float(entries))
            cdf = np.cumsum(counts)
            ax.plot(bin_edges[:-1], cdf, LINESTYLES[number], label=LABELS[i], color=COLORS[i])

        step = float(len(means)) / 5.0
        upper = 10

        plt.axvline(3, color=COLORS[-1])
        plt.xlim((1,upper))
        ax.set_xticks(range(1, upper + 1))
        ax.set_xticklabels(["%dx" % (i + 1) for i in range(upper)])
        plt.ylim((0, 100))

def slowdown_cdf_small(args, datas):
    L = int(args[0]) if args else 0

    fig, ax = plt.subplots(nrows=1, ncols=1)
    for number, data in enumerate(datas):
        means = data.means
        slowdowns = means / means[0,:]
        graph = lnm.fromkeyvals(data.names, slowdowns)
        graph = lnm.compute_lnm_times(graph, L)

        results = graph.ungraph()[1]
        results = zip(*results)
        entries = means.shape[0]

        for i, result in enumerate(results):
            if i == 1:
                continue
            counts, bin_edges = np.histogram(result, bins=max(entries, 1024))
            counts = counts * (100.0 / float(entries))
            cdf = np.cumsum(counts)
            ax.plot(bin_edges[:-1], cdf, LINESTYLES[number], label=LABELS[i], color=COLORS[i])

        upper = 3

        plt.axvline(3, color=COLORS[-1])
        plt.xlim((1,upper))
        ax.set_xticks(range(1, upper + 1))
        ax.set_xticklabels(["%dx" % (i + 1) for i in range(upper)])
        plt.ylim((0, 100))

def slowdown_cdf_hidden(args, datas):

    if args:
        upper = int(args[0])
    else:
        upper = 5
    L = 0

    fig, ax = plt.subplots(nrows=1, ncols=1)
    for number, data in enumerate(datas):
        means = data.means
        slowdowns = means / means[0,:]
        graph = lnm.fromkeyvals(data.names, slowdowns)
        graph = lnm.compute_lnm_times(graph, L)

        print np.sum(slowdowns < 3.0, axis=0)

        results = graph.ungraph()[1]
        results = zip(*results)
        entries = means.shape[0]

        for i, result in enumerate(results):
            if i == 0:
                continue
            median = np.median(result)
            counts, bin_edges = np.histogram(result, bins=max(entries, 1024))
            counts = counts * (100.0 / float(entries))
            cdf = np.cumsum(counts)
            ax.plot(bin_edges[:-1], cdf, LINESTYLES[number], label=LABELS[i], color=COLORS[i])

        plt.axvline(3, color=COLORS[-1])
        plt.xlim((1,upper))
        ax.set_xticks(range(1, upper + 1))
        ax.set_xticklabels(["%dx" % (i + 1) for i in range(upper)])
        plt.ylim((0, 100))

def violin(args, data):
    means = data.means
    vars  = data.variances

    fake_handles = []

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 5))
    N = data.times.shape[-1]
    for i, color in izip(range(N), COLORS):
        parts = ax.violinplot(data.times[:,:,i], showmedians=True)
        for part in parts['bodies']:
            part.set_facecolor(color)
        parts['cmedians'].set_color(color)
        parts['cmins'].set_color(color)
        parts['cmaxes'].set_color(color)
        parts['cbars'].set_color(color)

        patch = mpatches.Patch(color=color)
        fake_handles.append(patch)

    ax.legend(fake_handles, LABELS[:N], bbox_to_anchor=(1.0, 0.5))

    ax.set_xticks(range(1, len(data.names) + 1))
    ax.set_xticklabels(data.names, rotation='vertical')
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(5)

def violin_order_runtime(args, data):
    times = data.times
    names = data.names
    means = data.means
    vars  = data.variances

    fake_handles = []

    mapping = list(enumerate(means[:,0]))
    mapping.sort(key=op.itemgetter(1))
    indices, means = zip(*mapping)
    times = times[:,indices,:]
    names = [names[i] for i in indices]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 5))
    N = data.times.shape[-1]
    for i, color in izip(range(N), COLORS):
        parts = ax.violinplot(times[:,:,i], showmedians=True)
        for part in parts['bodies']:
            part.set_facecolor(color)
        parts['cmedians'].set_color(color)
        parts['cmins'].set_color(color)
        parts['cmaxes'].set_color(color)
        parts['cbars'].set_color(color)

        patch = mpatches.Patch(color=color)
        fake_handles.append(patch)

    ax.legend(fake_handles, LABELS[:N], loc='best')

    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels(names, rotation='vertical')
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(5)

def popcnt(arg):
    return sum(c == '1' for c in arg[1])

def violin_order_lattice(args, data):
    names = data.names
    times = data.times
    vars  = data.variances

    # Compute the desired ordering then perform a scatter on the array
    mapping = list(enumerate(names))
    mapping.sort(key=popcnt)
    indices, names = zip(*mapping)
    times = times[:,indices,:]

    fake_handles = []

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 5))
    N = data.times.shape[-1]
    for i, color in izip(range(N), COLORS):
        parts = ax.violinplot(times[:,:,i], showmedians=True)
        for part in parts['bodies']:
            part.set_facecolor(color)
        parts['cmedians'].set_color(color)
        parts['cmins'].set_color(color)
        parts['cmaxes'].set_color(color)
        parts['cbars'].set_color(color)

        patch = mpatches.Patch(color=color)
        fake_handles.append(patch)

    ax.legend(fake_handles, LABELS[:N], bbox_to_anchor=(1.0, 0.5))

    ax.set_xticks(range(1, len(data.names) + 1))
    ax.set_xticklabels(names, rotation='vertical')
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(8)

def slowdown_to_racket(args, data):
    names = data.names
    times = data.times
    vars  = data.variances

    s = data.means.shape[-1] - 1

    base = data.means[:,0]
    data = data.means[:,1:] / np.tile(base, (s, 1)).T

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 5))
    for i, color in izip(range(s), COLORS[1:]):
        sys = data[:,i]
        ax.scatter(base, sys, color=color)

    ax.set_xlabel("racket runtime")
    ax.set_ylabel("pycket relative runtime")

PLOT = { 'violin': violin,
         'violin_order_runtime': violin_order_runtime,
         'violin_order_lattice': violin_order_lattice,
         'slowdown_cdf': slowdown_cdf,
         'slowdown_to_racket': slowdown_to_racket,
         'stats_table': stats_table,
         'slowdown_cdf_small': slowdown_cdf_small,
         'slowdown_cdf_hidden': slowdown_cdf_hidden }

def main(args):

    plot_type   = args.action
    input_files = args.data

    output = args.output

    try:
        plot = PLOT[plot_type]
    except KeyError:
        raise ValueError('invalid plot type "{}"'.format(plot_type))

    data = map(read_data_files, input_files)
    plot(args.args, data)

    if output is not None and output[0] == "show":
        plt.show()
    elif output is not None:
        plt.savefig(output[0], dpi=500)

    # graph = lnm.fromkeyvals(data.names, data.means)
    return data

if __name__ == '__main__':
    data = main(parser.parse_args())

