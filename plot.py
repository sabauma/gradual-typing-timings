
from collections import namedtuple
from itertools   import izip

import argparse
import lnm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import stats
import sys

Data = namedtuple('Data', 'names times means variances')

COLORS = ['red', 'green', 'blue', 'yellow', 'orange']
LABELS = ['racket', 'pycket', 'hidden']

parser = argparse.ArgumentParser(description="Plot some things")
parser.add_argument('action', help="what plot to generate")
parser.add_argument('data', nargs='+', help="data files to process", type=str)
parser.add_argument('--output', default=None, nargs=1, type=str)
parser.add_argument('--args', nargs=argparse.REMAINDER)

def print_help():
    pass

def read_data_files(files):
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

def slowdown_cdf(args, data):
    L = int(args[0]) if args else 0

    means = data.means
    slowdowns = means / means[0,:]
    graph = lnm.fromkeyvals(data.names, slowdowns)
    graph = lnm.compute_lnm_times(graph, L)

    results = graph.ungraph()[1]
    results = zip(*results)
    entries = means.shape[0]
    fig, ax = plt.subplots(nrows=1, ncols=1)

    for i, result in enumerate(results):
        counts, bin_edges = np.histogram(result)
        cdf = np.cumsum(counts)
        ax.plot(bin_edges[:-1], cdf, label=LABELS[i], color=COLORS[i])

    plt.axvline(3, color='y')
    plt.axvline(10, color='k')
    plt.axhline(0.7 * entries, color='c', ls='--')
    plt.xlabel('slowdown factor')
    plt.xlim((0,10))
    plt.ylim((0, entries))
    plt.ylabel('number below')
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

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

PLOT = { 'violin': violin, 'slowdown_cdf': slowdown_cdf }

def main(args):

    plot_type   = args.action
    input_files = args.data

    output = args.output

    try:
        plot = PLOT[plot_type]
    except KeyError:
        raise ValueError('invalid plot type "{}"'.format(plot_type))

    data = read_data_files(input_files)
    plot(args.args, data)

    if output is None:
        plt.show()
    else:
        plt.savefig(output[0], dpi=500)

    # graph = lnm.fromkeyvals(data.names, data.means)
    return data

if __name__ == '__main__':
    data = main(parser.parse_args())

