
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
import contextlib

import matplotlib as mpl
mpl.rc('lines', linewidth=3, color='r')
mpl.rc('font', family='Arial', size=22)

Data = namedtuple('Data', 'names times means variances')

GREEN  = (34.0 / 255.0, 139.0 / 255.0, 24.0 / 255.0)
COLORS = [(255.0 / 255.0, 90.0 / 255.0, 20.0 / 255.0),
          (36.0 / 255.0, 36.0 / 255.0, 140.0 / 255.0),
          GREEN,
          GREEN,
          GREEN,
          ]

LABELS = ['racket', 'pycket', 'baseline', 'no-callgraph', 'no-force-virtual-state', 'no-unroll', 'no-impersonator-loop']
LINESTYLES = ['-', '--', '-.']
VLINE = (218.0 / 255.0, 165.0 / 255.0, 32.0 / 255.0)

parser = argparse.ArgumentParser(description="Plot some things")
parser.add_argument('action', help="what plot to generate")
parser.add_argument('data', nargs='+', help="data files to process", type=str)
parser.add_argument('--output', default=None, nargs=1, type=str)
parser.add_argument('--args', nargs='+', default=None, type=str)
parser.add_argument('--systems', nargs='+', default=None, type=int)
parser.add_argument('--norm', nargs=1, default=None, type=int)
parser.add_argument('--abscolor', action="store_true")
parser.add_argument('--xmin', default=None, type=int)
parser.add_argument('--xmax', default=None, type=int)

PLOTS = {}

@contextlib.contextmanager
def smart_open(filename=None):
    if filename and filename != '-':
        fh = open(filename, 'w')
    else:
        fh = sys.stdout

    try:
        yield fh
    finally:
        if fh is not sys.stdout:
            fh.close()

def plot(f):
    assert f.__name__ not in PLOTS
    def wrapper(*args, **kwargs):
        result = f(*args, **kwargs)
        if result is None:
            return True
        assert isinstance(result, bool)
        return result

    PLOTS[f.__name__] = wrapper
    wrapper.func = f
    return wrapper

def print_help():
    pass

def validate_keys(keys):
    init = keys[0]
    for i in keys:
        for l, r in zip(init, i):
            assert all(l == r)
    return keys

def read_data_files(pattern):
    files = glob.glob(pattern)

    if not files:
        raise ValueError("cannot find any matching files: " % pattern)

    keys, times = zip(*[stats.read_raw_data(fname) for fname in files])
    validate_keys(keys)

    means     = np.mean(times, axis=0)
    variances = np.var(times, axis=0)
    return Data(keys[0], np.array(times), means, variances)

@plot
def aggregate(args, datas):
    all_data = np.hstack([d.means for d in datas])
    output = args.output

    if not output or output[0] == "show":
        output = None
    else:
        output = output[0]

    if args.systems is None:
        systems = np.array(range(all_data.shape[-1]))
    else:
        systems = np.array(args.systems)

    with smart_open(output) as outfile:
        for name, row in zip(datas[0].names, all_data):
            bit_string = "".join(map(str, name))
            entry_name = "configuration{}".format(bit_string)
            result = " ".join([entry_name] + map(str, row[systems]))
            outfile.write(result)
            outfile.write("\n")

    return False

@plot
def stats_table(args, datas):
    data = datas[0]

    slowdowns = data.means / data.means[0,:]
    untyped_untyped = data.means[0,1] / data.means[0,0]

    N = slowdowns.shape[0]
    max = np.max(slowdowns, axis=0)
    mean = np.mean(slowdowns, axis=0)
    ratio = slowdowns[-1,:]
    acceptable = np.sum(slowdowns < 2.0, axis=0) / float(N) * 100.0

    stats = np.array([max, mean, acceptable])

    rows = ["$ %0.1f $ & $ %0.1f $ & $ %0.0f $" % tuple(stats[:,i]) for i in [0, 1]]

    # print "%d &" % N,
    print " & ".join(rows),
    print " & $ %0.2f $" % untyped_untyped,
    print "\\\\"

@plot
def mean_slowdown(args, datas):
    all_data = np.hstack([d.means for d in datas])

    norm = args.norm and args.norm[0]
    if norm is None or norm == -1:
        norm = range(all_data.shape[-1])
    else:
        assert norm >= 0
    slowdowns = all_data / all_data[0,norm]

    systems = args.systems
    if systems is not None:
        slowdowns = slowdowns[:,systems]

    slowdowns = np.mean(slowdowns, axis=0)
    output = args.output
    if not output or output[0] == "show":
        output = None
    else:
        output = output[0]

    with smart_open(output) as outfile:
        data = " ".join(map(str, slowdowns))
        outfile.write(data)
        if output is None:
            outfile.write("\n")

    return False

def pad_weights(weights, arrs):
    needed = max([s.shape[-1] for s in arrs])
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

@plot
def aggregate_slowdown(args, datas):
    means   = [d.means for d in datas]
    weights = [np.ones(mean.shape[0]) / float(mean.shape[0]) for mean in means]

    norm = args.norm and args.norm[0]
    if norm is None or norm == -1:
        norm = Ellipsis
    else:
        assert norm >= 0
    slowdown = [m / m[0,norm] for m in means]

    weights, slowdown = pad_weights(weights, slowdown)

    slowdown = np.vstack(slowdown)
    weights  = np.vstack(weights)

    slowdown = np.sum(slowdown * weights, axis=0) / np.sum(weights, axis=0)
    output = args.output
    if not output or output[0] == "show":
        output = None
    else:
        output = output[0]

    with smart_open(output) as outfile:
        data = " ".join(map(str, slowdown))
        outfile.write(data)
        if output is None:
            outfile.write("\n")
    return False

@plot
def aggregate_slowdown_cdf(args, datas):
    means   = [d.means for d in datas]
    weights = [np.ones(mean.shape[0]) / float(mean.shape[0]) for mean in means]

    norm = args.norm and args.norm[0]
    if norm is None or norm == -1:
        norm = Ellipsis
    else:
        assert norm >= 0
    slowdowns = [m / m[0,norm] for m in means]

    weights, slowdowns = pad_weights(weights, slowdowns)
    weights = np.vstack(weights)
    slowdowns = np.vstack(slowdowns)

    systems = args.systems
    if systems is not None:
        slowdowns = slowdowns[:,systems]

    colors = colors_array(args)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    for i in range(slowdowns.shape[-1]):
        entries  = np.sum(weights[:,i])
        slowdown = slowdowns[:,i]
        counts, bin_edges = np.histogram(slowdown, bins=len(slowdown), weights=weights[:,i])
        cdf = np.cumsum(counts) / entries * 100.0
        ax.plot(bin_edges[:-1], cdf, color=colors[i])

    upper = 10
    # plt.axvline(3, color=VLINE)
    ax.set_xticks(range(1, upper + 1))
    ax.set_xticklabels(["%dx" % (i + 1) for i in range(upper)])
    plt.xlim((1,upper))
    plt.ylim((0, 100))

def colors_array(args):
    if not args.abscolor or args.systems is None:
        return COLORS
    colors = [COLORS[i] for i in args.systems]
    return colors

@plot
def slowdown_cdf(args, datas):
    assert datas
    if not args.args:
        LS = [0]
    else:
        LS = map(int, args.args)

    if args.xmin is None:
        xmin = 1
    else:
        xmin = args.xmin

    if args.xmax is None:
        xmax = 10
    else:
        xmax = args.xmax

    names = datas[0].names
    data = np.hstack([d.means for d in datas])

    norm = args.norm and args.norm[0]
    if norm is None or norm == -1:
        norm = range(data.shape[-1])
    else:
        assert norm >= 0
    slowdowns = data / data[0,norm]

    systems = args.systems
    if systems is not None:
        slowdowns = slowdowns[:,systems]

    colors = colors_array(args)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    for number in LS:

        graph = lnm.fromkeyvals(names, slowdowns)
        graph = lnm.compute_lnm_times(graph, number)

        results = graph.ungraph()[1]
        results = zip(*results)
        entries = data.shape[0]

        for i, result in enumerate(results):
            counts, bin_edges = np.histogram(result, bins=max(entries, 1024))
            counts = counts * (100.0 / float(entries))
            cdf = np.cumsum(counts)
            ax.plot(bin_edges[:-1], cdf, LINESTYLES[number], label=LABELS[i], color=colors[i])

    # plt.axvline(3, color=VLINE)
    plt.xlim((xmin, xmax))
    ax.set_xticks(range(1, xmax + 1))
    ax.set_xticklabels(["%dx" % (i + 1) for i in range(xmax)])
    plt.ylim((0, 100))

@plot
def slowdown_cdf_old(args, datas):
    args = args.args
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
            counts, bin_edges = np.histogram(result, bins=max(entries, 1024))
            counts = counts * (100.0 / float(entries))
            cdf = np.cumsum(counts)
            ax.plot(bin_edges[:-1], cdf, LINESTYLES[number], label=LABELS[i], color=COLORS[i])

    step = float(len(means)) / 5.0
    upper = 10
    # plt.axvline(3, color=VLINE)
    plt.xlim((1,upper))
    ax.set_xticks(range(1, upper + 1))
    ax.set_xticklabels(["%dx" % (i + 1) for i in range(upper)])
    plt.ylim((0, 100))

@plot
def slowdown_cdf_small(args, datas):
    args = args.args
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

        # plt.axvline(3, color=VLINE)
        plt.xlim((1,upper))
        ax.set_xticks(range(1, upper + 1))
        ax.set_xticklabels(["%dx" % (i + 1) for i in range(upper)])
        plt.ylim((0, 100))

@plot
def slowdown_cdf_hidden(args, datas):

    if args.args:
        upper = int(args.args[0])
    else:
        upper = 5
    L = 0

    LINESTYLES = ['-', ':']

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
            if args.systems is not None and i not in args.systems:
                continue
            median = np.median(result)
            counts, bin_edges = np.histogram(result, bins=max(entries, 1024))
            counts = counts * (100.0 / float(entries))
            cdf = np.cumsum(counts)
            ax.plot(bin_edges[:-1], cdf, LINESTYLES[number], label=LABELS[i], color=COLORS[i])

        # plt.axvline(3, color=VLINE)
        plt.xlim((1,upper))
        ax.set_xticks(range(1, upper + 1))
        ax.set_xticklabels(["%dx" % (i + 1) for i in range(upper)])
        plt.ylim((0, 100))

@plot
def slowdown_cdf_big(args, datas):
    args = args.args
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

        upper = 50

        plt.xlim((1,upper))
        ax.set_xticks([1] + range(5, upper + 1, 5))
        ax.set_xticklabels(["1x"] + ["%dx" % i for i in range(5, upper + 1, 5)])
        plt.ylim((0, 100))

@plot
def violin(args, data):
    args = args.args
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

@plot
def violin_order_runtime(args, data):
    args = args.args
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

@plot
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

@plot
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

def main(args):
    plot_type   = args.action
    input_files = args.data

    output = args.output

    try:
        plot = PLOTS[plot_type]
    except KeyError:
        raise ValueError('invalid plot type "{}"'.format(plot_type))

    data = map(read_data_files, input_files)
    needs_plot = plot(args, data)

    if needs_plot:
        if output is not None and output[0] == "show":
            plt.show()
        elif output is not None:
            plt.savefig(output[0], dpi=500)

    return data

if __name__ == '__main__':
    if len(sys.argv) > 1:
        data = main(parser.parse_args())

