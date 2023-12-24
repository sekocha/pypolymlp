#!/usr/bin/env python
import numpy as np
import os, sys
import itertools
import argparse


def print_train_infile(filename='train.in', 
                       n_type=1, 
                       wforce='True', 
                       reg_method='ridge',
                       min_alpha=-5, 
                       max_alpha=-1, 
                       n_alpha=5, 
                       des_type='gtinv', 
                       pair_type='gaussian', 
                       gauss1=[1.0,1.0,1], 
                       gauss2=[0,7,10],
                       cutoff=8.0,
                       model_type=1,
                       max_p=1,
                       max_l=5, 
                       gtinv_order=6,
                       gtinv_maxl=[5,5,3,2,2],
                       gtinv_sym=['False','False','False','False','False'], 
                       weight='False'):

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

    f.close()

def enumerate_lcomb(gtinv_order=6,
                    maxl_a=[8,4,2,1,1],
                    interval=[2,2,1,1,1]):

    cand = [list(range(0,maxl_a[order-2]+1,interval[order-2])) \
        for order in range(2,gtinv_order+1)]

    lcomb = []
    for comb in list(itertools.product(*cand)):
        diff = np.array(comb) - np.array(sorted(comb, reverse=True))
        if np.all(diff == 0) == True:
            lcomb.append(comb)

    return lcomb

