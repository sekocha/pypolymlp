#!/usr/bin/env python
import numpy as np
import sys

import phono3py
from pypolymlp.utils.phonopy_utils import phonopy_cell_to_st_dict
from pypolymlp.core.displacements import (
    set_dft_dict,
    convert_disps_to_positions,
    get_structures_from_multiple_positions,
)


def parse_phono3py_yaml(
    yamlfile, 
    energies_filename=None,
    element_order=None,
    select_ids=None,
    return_displacements=False,
    use_phonon_dataset=False
):
    '''
    Parameters
    ----------
    yamlfile: phono3py yaml file
    energies_filename: first line gives the energy of perfect supercell

    phono3py.yaml must be composed of datasets for a fixed composition.
    '''
    ph3 = Phono3pyYaml(yamlfile, use_phonon_dataset=use_phonon_dataset)
    disps, forces = ph3.phonon_dataset
    supercell, positions_all = ph3.structure_dataset

    if energies_filename is None:
        energies = ph3.energies
    else:
        energies = np.loadtxt(energies_filename)[1:,1]

    if select_ids is not None:
        select_ids = np.array(select_ids)
        forces = forces[select_ids]
        positions_all = positions_all[select_ids]
        energies = energies[select_ids]
        disps = disps[select_ids]

    dft_dict = set_dft_dict(
        forces, energies, positions_all, supercell, 
        element_order=element_order,
    )
    dft_dict['include_force'] = True
    dft_dict['weight'] = 1.0

    if return_displacements:
        return dft_dict, disps
    return dft_dict


def parse_structures_from_phono3py_yaml(phono3py_yaml, select_ids=None):

    ph3 = Phono3pyYaml(phono3py_yaml)
    if select_ids is not None:
        return [ph3.supercells[i] for i in select_ids]
    return ph3.supercells


def parse_phono3py_yaml_fcs(phono3py_yaml, use_phonon_dataset=False):
    '''displacements: shape=(n_str, 3, n_atoms)'''
    ph3 = Phono3pyYaml(phono3py_yaml, use_phonon_dataset=use_phonon_dataset)
    return (ph3.supercell_phono3py, ph3.displacements, ph3.supercells)


class Phono3pyYaml:

    def __init__(self, yamlfile, use_phonon_dataset=False):
        self.__ph3 = phono3py.load(yamlfile, produce_fc=False, log_level=1)

        if use_phonon_dataset == False:
            self.__supercell = self.__ph3.supercell
            self.__displacements = self.__ph3.displacements.transpose((0,2,1))
            self.__forces = self.__ph3.forces.transpose((0,2,1)) 
            self.__energies = self.__ph3.supercell_energies
        else:
            print('Using phono3py.phonon_*** dataset')
            self.__supercell = self.__ph3.phonon_supercell
            self.__displacements = self.__ph3.phonon_dataset['displacements']
            self.__displacements = self.__displacements.transpose((0,2,1))
            self.__forces = self.__ph3.phonon_dataset['forces']
            self.__forces = self.__forces.transpose((0,2,1))
            """TODO: Must be revised"""
            self.__energies = None

        self.__supercell_dict = phonopy_cell_to_st_dict(self.__supercell)
        self.__positions_all = convert_disps_to_positions(
            self.__displacements, 
            self.__supercell_dict['axis'], 
            self.__supercell_dict['positions']
        )

    @property
    def phono3py(self):
        return self.__ph3

    @property
    def supercell_phono3py(self):
        return self.__supercell

    @property
    def supercell_dict(self):
        return self.__supercell_dict

    @property
    def displacements(self):
        '''displacements: shape=(n_samples, 3, n_atom)'''
        return self.__displacements

    @property
    def forces(self):
        '''forces: shape=(n_samples, 3, n_atom)'''
        return self.__forces

    @property
    def energies(self):
        '''forces: shape=(n_samples)'''
        return self.__energies

    @property
    def phonon_dataset(self):
        return (self.displacements, self.forces)

    @property
    def structure_dataset(self):
        ''' positions_all: shape=(n_samples, 3, n_atom) '''
        return (self.supercell_dict, self.__positions_all)

    def supercells(self):
        return get_structures_from_multiple_positions(
            self.supercell_dict, self.__positions_all
        )


if __name__ == '__main__':

    dft_dict = parse_phono3py_yaml(sys.argv[1], sys.argv[2], 
                                    select_ids=range(200))
    print(dft_dict['filenames'])
