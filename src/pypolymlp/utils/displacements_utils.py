#!/usr/bin/env python
import numpy as np
from pypolymlp.utils.structure_utils import refine_positions

def convert_disps_to_positions(disps, axis, positions):

    ''' disps: (n_str, 3, n_atoms) # Angstrom'''
    axis_inv = np.linalg.inv(axis)
    np.set_printoptions(suppress=True)
    positions_all = np.array([positions + (axis_inv @ d) for d in disps])
    return positions_all

def get_structures_from_multiple_positions(st_dict, positions_all):
    ''' positions_all: (n_str, 3, n_atom)'''
    st_dicts = []
    for positions_iter in positions_all:
        st = dict()
        st['axis'] = st_dict['axis']
        st['positions'] = positions_iter
        st['n_atoms'] = st_dict['n_atoms']
        st['elements'] = st_dict['elements']
        st['types'] = st_dict['types']
        st['volume'] = st_dict['volume']
        st_dicts.append(st)
    return st_dicts

def get_structures_from_displacements(disps, st_dict):
    ''' disps: (n_str, 3, n_atoms)'''
    positions_all = convert_disps_to_positions(disps,
                                               st_dict['axis'],
                                               st_dict['positions'])
    st_dicts = get_structures_from_multiple_positions(st_dict, positions_all)
    return st_dicts

def generate_random_const_displacements(st_dict, 
                                        n_samples=100, 
                                        displacements=0.03):

    positions = st_dict['positions']
    disps = []
    for i in range(n_samples):
        rand = np.random.rand(3, positions.shape[1]) - 0.5
        rand = rand / np.linalg.norm(rand, axis=0)
        disps.append(rand * displacements)
    disps = np.array(disps)

    st_dicts = get_structures_from_displacements(disps, st_dict)
    return disps, st_dicts

def generate_random_displacements(st_dict, 
                                  n_samples=100, 
                                  displacements=0.03):

    positions = st_dict['positions']
    disps = (np.random.rand(n_samples, 3, positions.shape[1]) - 0.5)
    disps = (2.0 * displacements) * disps

    st_dicts = get_structures_from_displacements(disps, st_dict)
    return disps, st_dicts


