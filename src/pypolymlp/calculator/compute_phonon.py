#!/usr/bin/env python 
import numpy as np
import argparse
import os
import signal

from phonopy import Phonopy

from pypolymlp.core.interface_vasp import Poscar
from pypolymlp.utils.yaml_utils import load_cells
from pypolymlp.utils.phonopy_utils import (
        phonopy_cell_to_st_dict,
        st_dict_to_phonopy_cell,
)
from pypolymlp.calculator.compute_properties import compute_properties

class PolymlpPhonon:

    def __init__(self, pot, unitcell_dict=None, supercell_matrix=None):

        self.pot = pot
        unitcell = st_dict_to_phonopy_cell(unitcell_dict)
        self.ph = Phonopy(unitcell, supercell_matrix)

    def produce_force_constants(self, displacements=0.01):

        self.ph.generate_displacements(distance=displacements)
        supercells = self.ph.supercells_with_displacements
        st_dicts = [phonopy_cell_to_st_dict(cell) for cell in supercells]

        ''' forces: (n_str, 3, n_atom) --> (n_str, n_atom, 3)'''
        _, forces, _ = compute_properties(st_dicts, pot=self.pot)
        forces = np.array(forces).transpose((0,2,1)) 
        self.ph.set_forces(forces)
        self.ph.produce_force_constants()

    def compute_properties(self,
                           mesh=[10,10,10],
                           t_min=0,
                           t_max=1000,
                           t_step=10,
                           with_eigenvectors=False,
                           is_mesh_symmetry=True,
                           pdos=False):

        self.ph.run_mesh(mesh,
                         with_eigenvectors=with_eigenvectors,
                         is_mesh_symmetry=is_mesh_symmetry)
        self.ph.run_total_dos()
        self.ph.run_thermal_properties(t_step=t_step, t_max=t_max, t_min=t_min)
        mesh_dict = self.ph.get_mesh_dict()

        os.makedirs('polymlp_phonon', exist_ok=True)
        np.savetxt('polymlp_phonon/mesh-qpoints.txt',
                    mesh_dict['qpoints'], fmt='%f')
        self.ph.write_total_dos(filename="polymlp_phonon/total_dos.dat")
        self.ph.write_yaml_thermal_properties(
            filename='polymlp_phonon/thermal_properties.yaml'
        )

        if pdos:
            self.ph.run_mesh(mesh,
                             with_eigenvectors=True,
                             is_mesh_symmetry=False)
            self.ph.run_projected_dos()
            self.ph.write_projected_dos(filename="polymlp_phonon/proj_dos.dat")


if __name__ == '__main__':

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    parser = argparse.ArgumentParser()
    parser.add_argument('--pot',
                        type=str,
                        default='polymlp.lammps',
                        help='polymlp file')
    parser.add_argument('--yaml',
                        type=str,
                        default=None,
                        help='polymlp_str.yaml file')
    parser.add_argument('--poscar',
                        type=str,
                        default=None,
                        help='poscar')
    parser.add_argument('--supercell',
                        nargs=3,
                        type=int,
                        default=None,
                        help='Supercell size (diagonal components)')
    parser.add_argument('--disp',
                        type=float,
                        default=0.01,
                        help='random displacement (in Angstrom)')

    parser.add_argument('--ph_mesh',
                        type=int,
                        nargs=3,
                        default=[10,10,10],
                        help='k-mesh used for phonon calculation')
    parser.add_argument('--ph_tmin',
                        type=float,
                        default=100,
                        help='Temperature (min)')
    parser.add_argument('--ph_tmax',
                        type=float,
                        default=1000,
                        help='Temperature (max)')
    parser.add_argument('--ph_tstep',
                        type=float,
                        default=100,
                        help='Temperature (step)')
    parser.add_argument('--ph_pdos',
                        action='store_true',
                        help='Compute phonon PDOS')
    args = parser.parse_args()

    if args.yaml is not None:
        unitcell_dict, supercell_dict = load_cells(filename=args.yaml)
        supercell_matrix = supercell_dict['supercell_matrix']
    elif args.poscar is not None:
        unitcell_dict = Poscar(args.poscar).get_structure()
        supercell_matrix = np.diag(args.supercell)

    ph = PolymlpPhonon(args.pot, unitcell_dict, supercell_matrix)
    ph.produce_force_constants(displacements=args.disp)
    ph.compute_properties(mesh=args.ph_mesh,
                          t_min=args.ph_tmin,
                          t_max=args.ph_tmax,
                          t_step=args.ph_tstep,
                          pdos=args.ph_pdos)

