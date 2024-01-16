#!/usr/bin/env python 
import numpy as np

from phonopy import Phonopy
from pypolymlp.calculator.compute_properties import compute_properties
from pypolymlp.utils.phonopy_utils import phonopy_cell_to_st_dict


class HarmonicReciprocal:

    def __init__(self, phonopy_obj, params_dict, coeffs, fc2=None):

        self.ph = phonopy_obj
        self.__params_dict = params_dict
        self.__coeffs = coeffs
        self.fc2 = fc2

        self.__tp_dict = dict()
        '''
        self.__mesh_dict = dict()
        '''

    def compute_polymlp_properties(self, st_dicts):
        ''' energies: (n_str)
            forces: (n_str, 3, n_atom)
        '''
        energies, forces, _ = compute_properties(
                st_dicts,
                params_dict=self.__params_dict,
                coeffs=self.__coeffs
        )
        return energies, forces

    def produce_harmonic_force_constants(self, displacements=0.01):

        self.ph.generate_displacements(distance=displacements)
        supercells = self.ph.supercells_with_displacements
        st_dicts = [phonopy_cell_to_st_dict(cell) for cell in supercells]
        _, forces = self.compute_polymlp_properties(st_dicts)
        forces = np.array(forces).transpose((0,2,1))

        self.ph.set_forces(forces)
        self.ph.produce_force_constants()
        self.fc2 = self.ph.force_constants
        return self.fc2

    def compute_thermal_properties(self, t=1000, qmesh=[10,10,10]):

        self.ph.run_mesh(qmesh)
        self.ph.run_thermal_properties(t_step=10, t_max=t, t_min=t)
        self.__tp_dict = self.ph.get_thermal_properties_dict()
        return self

    @property
    def force_constants(self):
        ''' (n_atom, n_atom, 3, 3)'''
        return self.fc2

    @force_constants.setter
    def force_constants(self, fc2):
        ''' (n_atom, n_atom, 3, 3)'''
        self.fc2 = fc2
        self.ph.force_constants = fc2

    @property
    def free_energy(self):
        return self.__tp_dict['free_energy'][0]
 
