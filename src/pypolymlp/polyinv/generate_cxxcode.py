#!/usr/bin/env python
import numpy as np
import re
import yaml

def print2d(array, varname):

    print('    '+varname+' =')
    print('        {{'+', '.join(array[0])+'},')
    for a1 in array[1:-1]:
        print('        {'+', '.join(a1)+'},')
    print('        {'+', '.join(array[-1])+'}};')
    print('')


def print2dnewline(array, varname):

    print('    '+varname+' =')
    print('        {{'+',\n        '.join(array[0])+'},')
    for a1 in array[1:-1]:
        print('        {'+',\n         '.join(a1)+'},')
    print('        {'+',\n         '.join(array[-1])+'}};')
    print('')


def print3d(array, varname):

    print('    '+varname+' = {')
    for i, a2 in enumerate(array):
        if len(a2) == 1:
            print('        {{'+', '.join(a2[0])+'}},')
        else:
            print('        {{'+', '.join(a2[0])+'},')
            for a1 in a2[1:-1]:
                print('        {'+', '.join(a1)+'},')
            if i == len(array)-1:
                print('        {'+', '.join(a2[-1])+'}}};')
            else:
                print('        {'+', '.join(a2[-1])+'}},')
    print('')


l_all, m_all, c_all = [], [], []
#maxl = [[1],[30],[10,12,20],[2,4,8],[2],[2]]
maxl = [[1],[30],[20],[8],[2],[2]]
for order in range(1,7):
    f = open('lists/basis-order'+str(order)+'-l0.yaml')
    yamldata = yaml.safe_load(f)
    f.close()

    for j in range(len(maxl[order-1])):
        if j == 0:
            minl_t, maxl_t = 0, maxl[order-1][j]
        else:
            minl_t, maxl_t = maxl[order-1][j-1] + 1, maxl[order-1][j]

        for d in yamldata['basis_set']:
            lcomb = d['lcomb']
            if (sum(lcomb) % 2 == 0 
                and max(lcomb) <= maxl_t and max(lcomb) >= minl_t):
                l_all.append([str(l) for l in lcomb])
                mcombs = [mc_attr[0] for mc_attr in d['mcombs_coeffs']]
                coeffs = [str(mc_attr[1]) for mc_attr in d['mcombs_coeffs']]
                m_all.append([[str(m2) for m2 in m1] for m1 in mcombs])
                c_all.append(coeffs)


print('void GtinvData::set_gtinv_info(){')
print('')

print2d(l_all, 'l_array_all')
print2dnewline(c_all, 'coeffs_all')
print3d(m_all, 'm_array_all')

print('}')


