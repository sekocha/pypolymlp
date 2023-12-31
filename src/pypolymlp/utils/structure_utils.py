#!/usr/bin/env python
import numpy as np
import argparse
import itertools
from math import pi

import sys
from pypolymlp.core.interface_vasp import Poscar
from pypolymlp.utils.vasp_utils import print_poscar

def refine_positions(st_dict, tol=1e-13):

    positions = st_dict['positions']
    positions -= np.floor(positions)
    positions[np.where(positions > 1-tol)] -= 1.0
    st_dict['positions'] = positions
    return st_dict

def reset_types(st_dict):
    st_dict['types'] = [i for i, n1 in enumerate(st_dict['n_atoms'])
                          for n2 in range(n1)]
    return st_dict

def calc_positions_cartesian(st_dict):
    return st_dict['axis'] @ st_dict['positions']

def get_reciprocal_axis(st_dict=None,axis=None):
    if axis is None:
        axis = st_dict['axis']
    return 2 * pi * np.linalg.inv(axis).T

def disp(st_dict, eps=0.001):
    shape = st_dict['positions'].shape
    disp = (2 * eps) * (np.random.rand(shape) - 0.5)
    st_dict['positions'] += disp
    return st_dict

def isotropic_volume_change(st_dict, eps=1.0):
    eps1 = pow(eps, 0.3333333333333)
    st_dict['axis'] *= eps1
    return st_dict

def multiple_isotropic_volume_changes(st_dict, 
                                      eps_min=0.8, eps_max=2.0, n_eps=10):
    volmin = pow(eps_min, 0.3333333333333)
    volmax = pow(eps_max, 0.3333333333333)
    eps_array = np.linspace(volmin, volmax, n_eps)
    st_dicts = [isotropic_volume_change(st_dict, eps=eps) for eps in eps_array]
    return st_dicts

def supercell_diagonal(st_dict, size=[2,2,2]):

    supercell_matrix = np.diag(size)
    n_expand = np.prod(size)

    supercell = dict()
    supercell['axis'] = st_dict['axis'] @ supercell_matrix
    supercell['n_atoms'] =  np.array(st_dict['n_atoms']) * n_expand
    supercell['types'] = np.repeat(st_dict['types'], n_expand)
    supercell['elements'] = np.repeat(st_dict['elements'], n_expand)
    supercell['volume'] = st_dict['volume'] * n_expand

    trans_all = np.array(list(itertools.product(range(size[0]),
                                                range(size[1]),
                                                range(size[2]))), 
                                                dtype=float)
    size = np.array(size, dtype=float)
    positions_new = []
    for pos in st_dict['positions'].T:
        pos_new = (pos + trans_all) / size
        positions_new.extend(pos_new)
    supercell['positions'] = np.array(positions_new).T

    supercell['supercell_matrix'] = supercell_matrix
    return supercell

def remove(st_dict, idx):
    ''' idx-th element is removed from st_dict. '''
    begin = int(np.sum(st_dict['n_atoms'][:idx]))
    end = begin + st_dict['n_atoms'][idx]
    st_dict['positions'] = np.delete(st_dict['positions'], 
                                     range(begin, end), axis=1)
    st_dict['n_atoms'] = np.delete(st_dict['n_atoms'], idx)
    st_dict['elements'] = np.delete(st_dict['elements'], range(begin, end))
    st_dict['types'] = np.delete(st_dict['types'], range(begin, end))
    return st_dict

def reorder(st_dict, order=None, index1=None, index2=None):

    if order is None:
        order = list(range(len(st_dict['n_atoms'])))
        order[index1], order[index2] = order[index2], order[index1]

    positions, n_atoms, types, elements = [], [], [], []
    for i in order:
        n_atoms.append(st_dict['n_atoms'][i])
        begin = int(np.sum(st_dict['n_atoms'][:i]))
        end = int(begin + st_dict['n_atoms'][i])
        positions.extend(st_dict['positions'].T[begin:end])
        elements.extend(st_dict['elements'][begin:end])
        types.extend(st_dict['types'][begin:end])

    st_dict['n_atoms'] = np.array(n_atoms)
    st_dict['elements'] = np.array(elements)
    st_dict['types'] = np.array(types)
    st_dict['positions'] = np.array(positions).T
    return st_dict

def swap_elements(st_dict, order=None, index1=None, index2=None):
    ''' Orders of element and type are fixed. 
        Orders of positions, n_atoms are swapped.'''
    if order is None:
        order = list(range(len(st_dict['n_atoms'])))
        order[index1], order[index2] = order[index2], order[index1]

    uniq_ele = []
    for i, n in enumerate(st_dict['n_atoms']):
        begin = int(np.sum(st_dict['n_atoms'][:i]))
        uniq_ele.append(st_dict['elements'][begin])

    positions, n_atoms, types, elements = [], [], [], []
    for t, i in enumerate(order):
        n_atoms.append(st_dict['n_atoms'][i])
        begin = int(np.sum(st_dict['n_atoms'][:i]))
        end = int(begin + st_dict['n_atoms'][i])
        positions.extend(st_dict['positions'].T[begin:end])

        elements.extend([uniq_ele[t] for n in range(st_dict['n_atoms'][i])])
        types.extend([t for n in range(st_dict['n_atoms'][i])])

    st_dict['n_atoms'] = np.array(n_atoms)
    st_dict['elements'] = np.array(elements)
    st_dict['types'] = np.array(types)
    st_dict['positions'] = np.array(positions).T
    return st_dict


if __name__ == '__main__':

    st_dict = Poscar(sys.argv[1]).get_structure()
    st_dict = supercell_diagonal(st_dict)
    #st_dict = remove(st_dict, 1)
    #st_dict = reorder(st_dict, order=[1,0])
    st_dict = swap_elements(st_dict, order=[1,0])
    print_poscar(st_dict)


    
