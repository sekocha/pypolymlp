#!/usr/bin/env python
import numpy as np
import argparse
import os
import random

from pypolymlp.common.vasp import Poscar
from pypolymlp.stgen.structure import make_supercell
from pypolymlp.stgen.structure import print_poscar_tofile

class StructureGenerator:

    def __init__(self, poscar='POSCAR', max_natoms=128):

        self.st_orig, self.axis_inv_array, self.name_array = [], [], []

        self.label = poscar
        self.supercell = self.__poscar_to_supercell(poscar, 
                                                    max_natoms=max_natoms)

    def __poscar_to_supercell(self, poscar, max_natoms=128):

        unitcell = Poscar(poscar).get_structure()
        size = self.__find_supercell_size_nearly_isotropic(unitcell['axis'],
                                                           unitcell['n_atoms'], 
                                                           max_natoms)
        supercell = make_supercell(unitcell, size)
        supercell['axis_inv'] = np.linalg.inv(supercell['axis'])
        return supercell

    def __find_supercell_size_nearly_isotropic(self, axis, n_atoms, max_natoms):

        len_axis = [np.linalg.norm(axis[:,i]) for i in range(3)]
        ratio1 = len_axis[0] / len_axis[1]
        ratio2 = len_axis[0] / len_axis[2]
        ratio = np.array([1, ratio1, ratio2])
        cand = np.array([1,2,3,4,5])

        size = [1,1,1]
        for c in cand:
            ex = np.maximum(np.round(ratio * c), [1,1,1])
            n_total = np.sum(n_atoms) * np.prod(ex)
            if n_total <= max_natoms:
                size = [int(e) for e in ex]
        return size

    def random_structure(self, n_st=10, max_disp=0.1, vol_ratio=1.0):

        st_array = []
        cell = self.supercell
        total_n_atoms = cell['positions'].shape[1]
        axis_ratio = pow(vol_ratio, 1.0/3.0)
        disp_array = [(i+1) * max_disp / float(n_st) for i in range(n_st)]

        for disp in disp_array:
            axis_add = (np.random.rand(3,3) * 2.0 - 1) * disp
            positions_add = (np.random.rand(3,total_n_atoms) * 2.0 - 1) * disp
            positions_add = np.dot(cell['axis_inv'], positions_add)

            structure = dict()
            structure['axis'] = cell['axis'] * axis_ratio + axis_add
            structure['positions'] = cell['positions'] + positions_add
            structure['n_atoms'] = cell['n_atoms']
            structure['elements'] = cell['elements']
            structure['types'] = cell['types']
            st_array.append(structure)

        return st_array

    """
    # obsolete
    def __find_supercell_size(self, n_atoms, max_natoms):
        cand = np.array([1, 8, 27, 64, 125]) * np.sum(n_atoms)
        if cand[0] > max_natoms:
            index = 1
        else:
            index = np.argmax(cand[np.where(cand <= max_natoms)]) + 1
        return [index, index, index]
    """


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--poscar',
                        type=str,
                        nargs='*',
                        required=True,
                        help='Initial structures in POSCAR format')

    parser.add_argument('--max_natom',
                        type=int,
                        default=100, 
                        help='Maximum number of atoms in structures')

    parser.add_argument('-n','--n_structures',
                        type=int,
                        nargs='*',
                        default=[100,10],
                        help='List of numbers of structures generated from '
                             'each poscar. n_structures[i] structures are '
                             'generated using max_disp[i].')
    parser.add_argument('-d','--max_disp',
                        type=float,
                        nargs='*',
                        default=[0.5,1.5], 
                        help='Maximum std. dev. for generating '
                             'atomic displacements')
 
    parser.add_argument('--density_mode_n_structures',
                        type=int,
                        default=10,
                        help='Number of structures generated in '
                             'low and high density modes')

    parser.add_argument('--density_mode_disp',
                        type=float,
                        default=0.2, 
                        help='Std. dev. for generating atomic '
                             'displacements in low and high density modes')

    parser.add_argument('--first_index',
                        type=int,
                        default=1,
                        help='First index for labeling structures')
    parser.add_argument('--output_dir',
                        type=str,
                        default='poscars-disp-init',
                        help='Output directory')

    parser.add_argument('--no_low_density',
                        action='store_true', 
                        help='Low density mode is forbidden.')
    parser.add_argument('--no_high_density',
                        action='store_true', 
                        help='High density mode is forbidden.')
    parser.add_argument('--no_standard',
                        action='store_true', 
                        help='Standard procedure is forbidden '
                             'when using only low and high density modes.')
    args = parser.parse_args()

    vol_array, n_st_array, disp_array = [], [], []
    if args.no_low_density == False:
        vol1 = np.linspace(1.1, 4.0, args.density_mode_n_structures)
        vol_array.extend(vol1)
        for i in range(len(vol1)):
            n_st_array.append(1)
            disp_array.append(args.density_mode_disp)

    if args.no_high_density == False:
        vol1 = np.linspace(0.6, 0.8, args.density_mode_n_structures)
        vol_array.extend(vol1)
        for i in range(len(vol1)):
            n_st_array.append(1)
            disp_array.append(args.density_mode_disp)

    if args.no_standard == False:
        for n, disp in zip(args.n_structures, args.max_disp):
            vol_array.append(1.0)
            n_st_array.append(n)
            disp_array.append(disp)

    os.makedirs(args.output_dir, exist_ok=True)
    index = args.first_index

    for vol, n_st, disp in zip(vol_array, n_st_array, disp_array):
        for name in args.poscar:
            gen = StructureGenerator(poscar=name, max_natoms=args.max_natom)
            structures = gen.random_structure(n_st=n_st,
                                              max_disp=disp,
                                              vol_ratio=vol)
            for st in structures:
                filename = args.output_dir+'/poscar-'+str(index).zfill(5) 
                header = name + ': vol = ' + str(vol) + ': disp = ' + str(disp)
                print_poscar_tofile(st, filename, header=header)
                index += 1


