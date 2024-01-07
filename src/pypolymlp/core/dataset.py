#!/usr/bin/env python
import numpy as np
from collections import defaultdict

from pypolymlp.core.utils import permute_atoms

def set_dft_dict_from_displacement_dataset(
        forces, 
        energies, 
        positions_all, 
        st_dict, 
        element_order=None
):
    '''
    Parameters
    ----------
    forces: (n_str, 3, n_atom)
    energies: (n_str)
    positions_all: (n_str, 3, n_atom)
    st_dict: structure without displacements

    Return
    ------
    dft_dict: DFT training or test dataset in pypolymlp format
    '''
    dft_dict = defaultdict(list)
    dft_dict['energy'] = energies
    dft_dict['stress'] = np.zeros(forces.shape[0] * 6)
    for positions_iter, forces_iter in zip(positions_all, forces):
        st = dict()
        st['axis'] = st_dict['axis']
        st['positions'] = positions_iter
        st['n_atoms'] = st_dict['n_atoms']
        st['elements'] = st_dict['elements']
        st['types'] = st_dict['types']
        st['volume'] = st_dict['volume']

        if element_order is not None:
            st, forces_iter = permute_atoms(st, forces_iter, element_order)

        dft_dict['force'].extend(forces_iter.T.reshape(-1))
        dft_dict['structures'].append(st)
    dft_dict['force'] = np.array(dft_dict['force'])

    if element_order is not None:
        dft_dict['elements'] = element_order
    else:
        elements_rep = dft_dict['structures'][0]['elements']
        dft_dict['elements'] = sorted(set(elements_rep), 
                                      key=elements_rep.index)

    dft_dict['total_n_atoms'] = np.array([sum(st['n_atoms'])
                                         for st in dft_dict['structures']])
    n_data = len(dft_dict['structures'])
    dft_dict['filenames'] = ['disp-' + str(i+1).zfill(5) for i in range(n_data)]
    return dft_dict


