#!/usr/bin/env python
import numpy as np
import os, sys
import itertools
import argparse

from mlptools.model_selection.common import print_train_infile
from mlptools.model_selection.common import enumerate_lcomb

if __name__ == '__main__':

    ps = argparse.ArgumentParser()
    ps.add_argument('-n', '--n_type', type=int, required=True)
    ps.add_argument('-d','--des_type',choices=['pair','gtinv'],default='gtinv')
    ps.add_argument('--cutoffs', 
                    type=float, 
                    nargs='*', 
                    default=[6.0,8.0,10.0])
    ps.add_argument('--min_alpha', type=float, default=-5)
    ps.add_argument('--max_alpha', type=float, default=-1)
    ps.add_argument('--n_alpha', type=int, default=6)
    ps.add_argument('--n_gauss',type=int, nargs='*', default=[7,10])

    ps.add_argument('--max_p', type=int, nargs='*', default=[1,2])
    ps.add_argument('--start_index',type=int, default=1)

    ps.add_argument('--model_type',type=int, nargs='*',default=[1])
    ps.add_argument('--max_gtinv_order',
                    type=int, 
                    default=4)
    ps.add_argument('--max_gtinv_maxl',
                    type=int, 
                    nargs='*',
                    default=[8,4,2,1,1])
    ps.add_argument('--interval_gtinv_maxl',
                    type=int, 
                    nargs='*',
                    default=[4,2,1,1,1])
    args = ps.parse_args()

    cutoff_a = np.linspace(args.min_cutoff, args.max_cutoff, args.n_cutoff)
    index = args.start_index
    if args.des_type == 'pair':
        print('#  model, max_p, n_gauss, cutoff')
        for model, p, ng, cutoff in itertools.product\
            (args.model_type, args.max_p, args.gauss, cutoff_a):
            gauss1, gauss2 = [1.0,1.0,1], [0.0, cutoff-1.0, ng]
            dirname='grid/pair-'+str(index)
            os.makedirs(dirname, exist_ok=True)
            print(dirname, model, p, ng, cutoff)
            print_train_infile\
                (filename=dirname+'/train.in', 
                n_type=args.n_type, wforce=args.with_force, \
                min_alpha=args.min_alpha, max_alpha=args.max_alpha,\
                n_alpha=args.n_alpha, \
                des_type='pair', gauss1=gauss1, gauss2=gauss2, \
                cutoff=cutoff, model_type=model, max_p=p)
            index += 1
    elif args.des_type == 'gtinv':
        gtinv_order_a = list(range(2,args.max_gtinv_order+1))
        lcomb_a = [enumerate_lcomb(gtinv_order=order,\
            maxl_a=args.max_gtinv_maxl, interval=args.interval_gtinv_maxl) \
            for order in gtinv_order_a]
        print('#  model, max_p, n_gauss, cutoff, lcomb')
        for order, model, p, ng, cutoff in itertools.product\
            (gtinv_order_a,args.model_type, args.max_p, args.gauss, cutoff_a):
            gauss1, gauss2 = [1.0,1.0,1], [0.0, cutoff-1.0, ng]
            for comb in lcomb_a[order-2]:
                dirname='grid/gtinv-'+str(index)
                os.makedirs(dirname, exist_ok=True)
                print(dirname, model, p, ng, cutoff, comb)
                print_train_infile\
                    (filename=dirname+'/train.in', 
                    n_type=args.n_type, wforce=args.with_force, \
                    min_alpha=args.min_alpha, max_alpha=args.max_alpha,\
                    n_alpha=args.n_alpha, \
                    des_type='gtinv', gauss1=gauss1, gauss2=gauss2, \
                    cutoff=cutoff, model_type=model, max_p=p,\
                    gtinv_order=order,gtinv_maxl=comb)
                index += 1
