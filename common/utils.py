#!/usr/bin/env python
import numpy as np

def permute_atoms(st, force, element_order):

    positions, n_atoms, elements, types = [], [], [], []
    force_permute = []
    for atomtype, ele in enumerate(element_order):
        ids = np.where(np.array(st['elements']) == ele)[0]
        n_match = len(ids)
        positions.extend(st['positions'][:,ids].T)
        n_atoms.append(n_match)
        elements.extend([ele for _ in range(n_match)])
        types.extend([atomtype for _ in range(n_match)])
        force_permute.extend(force[:,ids].T)
    positions = np.array(positions).T
    force_permute = np.array(force_permute).T

    st['positions'] = positions
    st['n_atoms'] = n_atoms
    st['elements'] = elements
    st['types'] = types
    return st, force_permute

