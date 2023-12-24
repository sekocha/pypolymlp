#!/usr/bin/env python
import numpy as np

def generate_random_displacements(st_dict, 
                                  n_samples=100, 
                                  displacements=0.03):

    axis = st_dict['axis']
    positions = st_dict['positions']
    disps = (np.random.rand(n_samples, 3, positions.shape[1]) - 0.5)
    disps = (2.0 * displacements) * disps
    axis_inv = np.linalg.inv(axis)
    positions_all = np.array([positions + (axis_inv @ d) for d in disps])

    st_dicts = []
    for positions_iter in positions_all:
        st = dict()
        st['axis'] = axis
        st['positions'] = positions_iter
        st['n_atoms'] = st_dict['n_atoms']
        st['elements'] = st_dict['elements']
        st['types'] = st_dict['types']
        st['volume'] = st_dict['volume']
        st_dicts.append(st)

    return disps, st_dicts

   
