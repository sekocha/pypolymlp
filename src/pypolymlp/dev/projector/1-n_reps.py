#!/usr/bin/env python
import numpy as np
import itertools
import argparse
from math import *
from scipy.integrate import quad

def chi(x, l, n):
    return sin((2*l+1)*n*x/2)/sin(x*n/2)

def chi1(x, lcomb):
    return np.prod([chi(x,l,1) for l in lcomb])

def n_all_lcomb(lcomb, lproj):
    num, _ = quad(lambda x:chi(x,lproj,1)*chi1(x,lcomb)*sin(x/2)**2, 0, 2*pi)
    return round(num/pi)
 
def n_all(maxl, order, lproj):
    lcomb_all = \
        list(itertools.combinations_with_replacement(range(0,maxl+1),order))
    n_list = [n_all_lcomb(lcomb, lproj) for lcomb in lcomb_all]
    return n_list, lcomb_all

def print_n_comb(n_list, lcomb_all, fname='l_list'):
    f = open(fname, 'w')
    for n, comb in zip(n_list, lcomb_all):
        if (n > 0):
            print(n, comb, file=f)
    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--maxl', type=int, required=True, help='Max l value.')
    parser.add_argument('--order', type=int, required=True, \
        help='n-th order product.')
    parser.add_argument('--lproj', type=int, default=0, \
        help='l value for projection')
    args = parser.parse_args()

    n_list, lcomb_all = n_all(args.maxl, args.order, args.lproj)

    n_total = []
    for l in range(0,args.maxl+1):
        n1 = sum([n for n, lcomb in zip(n_list, lcomb_all) \
            if max(lcomb) <= l])
        n_total.append(n1)

    print('order (accumurate) =', args.order, n_total)
    print_n_comb(n_list, lcomb_all, \
        fname='./lists/lcomb/lcomb-o'+str(args.order)+'-l'+str(args.lproj))

#print(n_all_lcomb([5,4,2], 3, 0))
#n_list, lcomb_all = n_all(5, 2, 2)
#for a, b in zip(n_list, lcomb_all):
#    print(a, b)


