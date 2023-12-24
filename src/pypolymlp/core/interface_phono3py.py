#!/usr/bin/env python
import numpy as np
import sys
from collections import defaultdict

import phono3py
from pypolymlp.core.utils import permute_atoms
from pypolymlp.utils.phonopy_utils import phonopy_cell_to_st_dict
from pypolymlp.utils.displacements_utils import convert_disps_to_positions

def parse_phono3py_yaml(yaml_filename, 
                        energies_filename=None,
                        element_order=None,
                        select_ids=None,
                        return_displacements=False):
    ''' yaml_filename: phono3py yaml file
        energies_filename: first line gives the energy of perfect supercell

        phono3py.yaml must be composed of datasets for a fixed composition.
    '''
    ph3 = Phono3pyYaml(yaml_filename)
    disps, forces = ph3.get_phonon_dataset()
    st_dict, positions_all = ph3.get_structure_dataset()

    if energies_filename is not None:
        energies = np.loadtxt(energies_filename)[1:,1]
    else:
        energies = None

    if select_ids is not None:
        select_ids = np.array(select_ids)
        forces = forces[select_ids]
        positions_all = positions_all[select_ids]
        energies = energies[select_ids]
        disps = disps[select_ids]

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

    elements_rep = dft_dict['structures'][0]['elements']
    dft_dict['elements'] = sorted(set(elements_rep), key=elements_rep.index)
    dft_dict['total_n_atoms'] = np.array([sum(st['n_atoms'])
                                         for st in dft_dict['structures']])
    n_data = len(dft_dict['structures'])
    dft_dict['filenames'] = ['phono3py-' + str(i+1).zfill(5) 
                               for i in range(n_data)]
    if return_displacements:
        return dft_dict, disps
    return dft_dict

''' not restricted to phono3py'''
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

''' not restricted to phono3py'''
def get_structures_from_displacements(disps, st_dict):
    ''' disps: (n_str, 3, n_atoms)'''
    positions_all = convert_disps_to_positions(disps, 
                                               st_dict['axis'], 
                                               st_dict['positions'])
    st_dicts = get_structures_from_multiple_positions(st_dict, positions_all)
    return st_dicts

def parse_structures_from_phono3py_yaml(phono3py_yaml):

    ph3 = Phono3pyYaml(phono3py_yaml)
    st_dict, positions_all = ph3.get_structure_dataset()
    st_dicts = get_structures_from_multiple_positions(st_dict, positions_all)
    return st_dicts

def parse_phono3py_yaml_fcs(phono3py_yaml):

    ''' disps: (n_str, 3, n_atoms)
        forces: (n_str, 3, n_atoms)
    '''
    ph3 = Phono3pyYaml(phono3py_yaml)
    disps, _ = ph3.get_phonon_dataset()
    st_dict, positions_all = ph3.get_structure_dataset()
    st_dicts = get_structures_from_multiple_positions(st_dict, positions_all)
    return ph3.supercell, disps, st_dicts


class Phono3pyYaml:

    def __init__(self, yaml_filename):
        ''' displacements: (n_samples, 3, n_atom)
            forces: (n_samples, 3, n_atom)
            positions_all: (n_samples, 3, n_atom)
        '''
        ph3 = phono3py.load(yaml_filename, produce_fc=False, log_level=1)
        self.supercell = ph3.supercell
        self.st_dict = phonopy_cell_to_st_dict(ph3.supercell)
        self.displacements = ph3.displacements.transpose((0,2,1))  # Angstrom
        self.forces = ph3.forces.transpose((0,2,1))  # eV/Angstrom

        self.positions_all = convert_disps_to_positions(
                self.displacements, 
                self.st_dict['axis'], 
                self.st_dict['positions']
        )

    def get_phonon_dataset(self):
        ''' displacements: (n_samples, 3, n_atom)
            forces: (n_samples, 3, n_atom)
        '''
        return (self.displacements, self.forces)

    def get_structure_dataset(self):
        ''' positions_all: (n_samples, 3, n_atom) '''
        return (self.st_dict, self.positions_all)


if __name__ == '__main__':

    dft_dict = parse_phono3py_yaml(sys.argv[1], sys.argv[2], 
                                    select_ids=range(200))
    print(dft_dict['filenames'])

