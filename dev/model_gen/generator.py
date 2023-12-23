#!/usr/bin/env python
import numpy as np
import os, sys
import itertools
import argparse

def print_train_infile\
    (filename='./train.in', n_type=1, wforce='True', reg_method='ridge', \
    min_alpha=-5, max_alpha=-1, n_alpha=5, des_type='gtinv', \
    pair_type='gaussian', gauss1=[0.25,0.5,2], gauss2=[0,10,11], \
    cutoff=10.0,model_type=1,max_p=1,max_l=5, gtinv_order=6,\
    gtinv_maxl=[5,5,3,2,2],\
    gtinv_sym=['False','False','False','False','False'], weight='False'):

    f = open(filename,'w')

    print('n_type', n_type, file=f) 
    print('with_force', wforce, file=f) 
    print('reg_method', reg_method, file=f) 
    print('alpha_min', min_alpha, file=f)
    print('alpha_max', max_alpha, file=f)
    print('n_alpha', n_alpha, file=f)

    print('des_type', des_type, file=f) 
    print('pair_type', pair_type, file=f) 
    print('gaussian_params1', gauss1[0], gauss1[1], gauss1[2], file=f)
    print('gaussian_params2', gauss2[0], gauss2[1], gauss2[2], file=f)

    print('cutoff', cutoff, file=f) 
    print('model_type', model_type, file=f)
    print('max_p', max_p, file=f)

    if (des_type == 'afs'):
        print('max_l', max_l, file=f)

    if (des_type == 'gtinv'):
        print('gtinv_order', gtinv_order, file=f)
        print('gtinv_maxl', end=' ', file=f)
        for i in range(gtinv_order-1):
            print(gtinv_maxl[i], end=' ', file=f)
        print('',file=f)
        print('gtinv_sym', end=' ', file=f)
        for i in range(gtinv_order-1):
            print(gtinv_sym[i], end=' ', file=f)
        print('',file=f)
#    print('weight', weight, file=f)

    f.close()

def enumerate_lcomb(gtinv_order=6,maxl_a=[8,4,2,1,1],interval=[2,2,1,1,1]):
    cand = [list(range(0,maxl_a[order-2]+1,interval[order-2])) \
        for order in range(2,gtinv_order+1)]

    lcomb = []
    for comb in list(itertools.product(*cand)):
        diff = np.array(comb) - np.array(sorted(comb, reverse=True))
        if np.all(diff == 0) == True:
            lcomb.append(comb)

    return lcomb

if __name__ == '__main__':

    ps = argparse.ArgumentParser()
    ps.add_argument('-n', '--n_type', type=int, required=True)
    ps.add_argument('-d','--des_type',choices=['pair','gtinv'],default='gtinv')
    ps.add_argument('--with_force', choices=['True','False'], default='True')
    ps.add_argument('--min_cutoff',type=float,default=6.0)
    ps.add_argument('--max_cutoff',type=float,default=10.0)
    ps.add_argument('--n_cutoff',type=int,default=3)
    ps.add_argument('--min_alpha',type=float,default=-5)
    ps.add_argument('--max_alpha',type=float,default=-1)
    ps.add_argument('--n_alpha',type=int,default=5)
    ps.add_argument('--gauss',type=int, nargs='*', default=[5,10])
    ps.add_argument('--max_p',type=int, nargs='*', default=[1,2,3])
    ps.add_argument('--start_index',type=int, default=1)

    ps.add_argument\
        ('--model_type',type=int, nargs='*',default=[1])
    ps.add_argument('--max_gtinv_order',type=int, default=4)
    ps.add_argument\
        ('--max_gtinv_maxl',type=int, nargs='*',default=[8,4,2,1,1])
    ps.add_argument\
        ('--interval_gtinv_maxl',type=int, nargs='*',default=[4,2,1,1,1])
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
