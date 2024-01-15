#!/usr/bin/env python 
import numpy as np
import argparse
import signal

from phonopy import Phonopy

from pypolymlp.core.interface_vasp import Poscar
from pypolymlp.core.io_polymlp import load_mlp_lammps
from pypolymlp.core.displacements import get_structures_from_displacements

from pypolymlp.utils.phonopy_utils import (
        phonopy_cell_to_st_dict,
        st_dict_to_phonopy_cell,
)
from pypolymlp.calculator.compute_properties import compute_properties


class PolymlpSSCHA:

    def __init__(self, 
                 unitcell_dict, 
                 supercell_matrix, 
                 pot=None, 
                 params_dict=None,
                 coeffs=None,
                 unitcell_dict=None, supercell_matrix=None):

        if pot is not None:
            self.params_dict, mlp_dict = load_mlp_lammps(filename=pot)
            self.coeffs = mlp_dict['coeffs'] / mlp_dict['scales']
        else:
            self.params_dict = params_dict
            self.coeffs = coeffs

        unitcell = st_dict_to_phonopy_cell(unitcell_dict)
        self.ph = Phonopy(unitcell, supercell_matrix)
        self.supercell = self.ph.supercell
        self.n_unitcells = int(round(np.linalg.det(supercell_matrix)))

        self.fc2_basis = FCBasisSetO2(self.ph.supercell, use_mkl=False).run()
        compress_mat_fc2 = fc2_basis.compression_matrix
        compress_eigvecs_fc2 = fc2_basis.basis_set

        ''' for bubble diagram
        fc3_basis = FCBasisSetO3(supercell, use_mkl=False).run()
        compress_mat_fc3 = fc3_basis.compression_matrix
        compress_eigvecs_fc3 = fc3_basis.basis_set
        '''

        self.fc2 = None

    def __compute_properties(self, st_dicts):
        ''' forces: (n_str, 3, n_atom) --> (n_str, n_atom, 3)'''
        _, forces, _ = compute_properties(st_dicts, 
                                          params_dict=self.params_dict,
                                          coeffs=self.coeffs)
        forces = np.array(forces).transpose((0,2,1))
        return forces
        

    def produce_harmonic_force_constants(self, displacements=0.01):
        self.ph.generate_displacements(distance=displacements)
        supercells = self.ph.supercells_with_displacements
        st_dicts = [phonopy_cell_to_st_dict(cell) for cell in supercells]
        forces = self.__compute_properties(st_dicts) # (n_str, n_atom, 3)
        self.ph.set_forces(forces)
        self.ph.produce_force_constants()
        self.fc2 = self.ph.force_constants
        return self.fc2

    def produce_harmonic_realspace_distribution(self, t=1000, n_samples=100):
        supercell_dict = phonopy_cell_to_st_dict(self.supercell)
        harm = HarmonicReal(supercell_dict, self.n_unitcells, self.fc2)
        disps = harm.get_distribution(t=t, n_samples=n_samples)
        supercells = get_structures_from_displacement(disps, supercell_dict)
        return disps, supercells

        

def run_sscha(unitcell_dict, 
              supercell_matrix,
              pot=None,
              params_dict=None,
              coeffs=None):

    sscha = PolymlpSSCHA(unitcell_dict, 
                         supercell_matrix, 
                         pot=pot, 
                         params_dict=params_dict, 
                         coeffs=coeffs)

    fc2_init = sscha.produce_force_constants()

if __name__ == '__main__':

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--poscar', 
                        type=str, 
                        default='POSCAR',
                        help='poscar file (unit cell)')
    parser.add_argument('--pot', 
                        type=str, 
                        default='polymlp.lammps',
                        help='polymlp.lammps file')
    parser.add_argument('--supercell',
                        nargs=3,
                        type=int,
                        default=None,
                        help='Supercell size (diagonal components)')
    args = parser.parse_args()

    unitcell_dict = Poscar(args.poscar).get_structure()
    supercell_matrix = np.diag(args.supercell)

    run_sscha(unitcell_dict, supercell_matrix, pot=args.pot)


