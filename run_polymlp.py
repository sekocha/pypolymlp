#!/usr/bin/env python 
import numpy as np
import argparse
import signal

from pypolymlp.mlp_gen.multi_datasets.generator import (
        run_generator_multiple_datasets
)
from pypolymlp.mlp_gen.multi_datasets.generator_sequential import (
        run_sequential_generator_multiple_datasets
)
from pypolymlp.mlp_gen.multi_datasets.additive.generator import (
        run_generator_additive
)
from pypolymlp.mlp_gen.multi_datasets.additive.generator_sequential import (
        run_sequential_generator_additive
)


if __name__ == '__main__':

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--infile', 
                        nargs='*',
                        type=str, 
                        default=['polymlp.in'],
                        help='Input file name')
    parser.add_argument('--sequential', 
                        action='store_true',
                        help='Use sequential evaluation of X.T @ X')
    args = parser.parse_args()

    if len(args.infile) == 1:
        infile = args.infile[0]
        if args.sequential:
            print('Mode: Sequential regression')
            run_sequential_generator_multiple_datasets(infile)
        else:
            print('Mode: Regression')
            run_generator_multiple_datasets(infile)
    else:
        if args.sequential:
            print('Mode: Sequential regression (additive model)')
            run_sequential_generator_additive(args.infile)
        else:
            print('Mode: Regression (additive model)')
            run_generator_additive(args.infile)
