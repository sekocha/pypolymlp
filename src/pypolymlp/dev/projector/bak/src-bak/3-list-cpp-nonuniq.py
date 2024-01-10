#!/usr/bin/env python
import numpy as np
import re

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
maxl = [1,20,10,2,1,1]
for order in range(1,7):
    f = open('lists/coeffs/coeffs-o'+str(order)+'-l0')
    lines = f.readlines()
    f.close()
    for i, line1 in enumerate(lines):
        if 'independent invariant:' in line1:
            split1 = re.split('[\[\]]', line1)
            lcomb = [x for x in split1[1].replace(' ','').split(',')]
            lcombint = [int(x) for x in lcomb]
            if sum(lcombint) % 2 == 0 and max(lcombint) <= maxl[order-1]:
                split2 = split1[2].split()
                iinv, nlines = int(split2[0]), int(split2[1])
                mcomb, coeffs = [], []
                for line2 in lines[i+1:i+nlines+1]:
                    split3 = line2.split()
                    coeffs.append(split3[-1])
                    mcomb.append([x for x in split3[1::2]])
                l_all.append(lcomb)
                m_all.append(mcomb)
                c_all.append(coeffs)

print('void GtinvData::set_gtinv_info(){')
print('')

print2d(l_all, 'l_array_all')
print2dnewline(c_all, 'coeffs_all')
print3d(m_all, 'm_array_all')

print('}')


