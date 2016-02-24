
from collections import namedtuple

import glob
import lnm
import numpy as np
import os
import stats
import sys

Data = namedtuple('Data', 'names means variances')

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

    return Data(keys[0], means, variances)

def main(args):
    if len(args) < 2:
        return print_help()

    pattern = args[1]
    data = read_data_files(pattern)
    graph = lnm.fromkeyvals(data.names, data.means)
    return data

if __name__ == '__main__':
    data = main(sys.argv)

