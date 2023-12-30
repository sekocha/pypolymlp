#!/usr/bin/env python 
import numpy as np
import argparse
import signal

from pypolymlp.mlp_opt.optimal import find_optimal_mlps
from pypolymlp.utils.vasprun_compress import convert
from pypolymlp.utils.dataset.auto_divide import auto_divide


def run():

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    parser = argparse.ArgumentParser()
    parser.add_argument('--vasprun_compress', 
                        nargs='*',
                        type=str, 
                        default=None,
                        help='Compression of vasprun.xml files')
    parser.add_argument('--n_jobs',
                        type=int,
                        default=1,
                        help='Number of parallel jobs')

    parser.add_argument('--auto_dataset', 
                        nargs='*',
                        type=str, 
                        default=None,
                        help='Automatic dataset division using ' + 
                             'vasprun.xml files')

    parser.add_argument('--find_optimal', 
                        nargs='*',
                        type=str, 
                        default=None,
                        help='Find optimal MLPs using a set of MLPs. ' + 
                             'Directories for the set of MLPs.')
    parser.add_argument('--key',
                        type=str,
                        default=None,
                        help='Identification key for the dataset ' +
                             'in finding optimal MLPs')

    args = parser.parse_args()

    if args.vasprun_compress is not None:
        if args.n_jobs == 1:
            for vasp in args.vasprun_compress:
                convert(vasp)
        else:
            from joblib import Parallel, delayed
            res = Parallel(n_jobs=args.n_jobs)(delayed(convert)(vasp)
                                        for vasp in args.vasprun_compress)

    elif args.auto_dataset is not None:
        auto_divide(args.auto_dataset)

    elif args.find_optimal is not None:
        find_optimal_mlps(args.find_optimal, args.key)

    '''
    spglib_utils
    str_gen/run_strgen
    todo: args.strgen should be implemented.
    '''

