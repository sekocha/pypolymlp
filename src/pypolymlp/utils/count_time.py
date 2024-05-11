#!/usr/bin/env python
import numpy as np
import glob

from pypolymlp.core.interface_vasp import Poscar
from pypolymlp.utils.phonopy_utils import (
   phonopy_cell_to_st_dict,
   st_dict_to_phonopy_cell,
)
from pypolymlp.calculator.properties import Properties

from phonopy import Phonopy

import argparse
import time

def run_single(prop, supercell_dict, args, filename='polymlp_cost.yaml'):

    print('Calculations have been started.')
    t1 = time.time()
    for i in range(args.n_calc):
        e, _, _ = prop.eval(supercell_dict)
    t2 = time.time()

    n_atoms_sum = sum(supercell_dict['n_atoms'])
    cost1 = (t2 - t1) / n_atoms_sum / args.n_calc
    cost1 *= 1000
    print('Total time (sec):', t2 - t1)
    print('Number of atoms:', n_atoms_sum)
    print('Number of steps:', args.n_calc)
    print('Computational cost (msec/atom/step):', cost1)

    print('Calculations have been started (openmp).')
    n_calc = args.n_calc * 10
    st_dicts = [supercell_dict for i in range(n_calc)]

    t3 = time.time()
    _, _, _ = prop.eval_multiple(st_dicts)
    t4 = time.time()

    cost2 = (t4 - t3) / n_atoms_sum / n_calc
    cost2 *= 1000
    print('Total time (sec):', t4 - t3)
    print('Number of atoms:', n_atoms_sum)
    print('Number of steps:', n_calc)
    print('Computational cost (msec/atom/step):', cost2)

    f = open(filename, 'w')
    print('units:', file=f)
    print('  time: msec/atom/step', file=f)
    print('', file=f)
    print('costs:', file=f)
    print('  single_core:', cost1, file=f)
    print('  openmp:     ', cost2, file=f)
    f.close()


if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--poscar',
                        type=str,
                        default=None,
                        help='poscar file')
    parser.add_argument('--pot',
                        nargs='*',
                        type=str,
                        default='polymlp.lammps',
                        help='polymlp file')

    parser.add_argument('-d', '--dirs',
                        type=str,
                        default=None,
                        help='directory path')

    parser.add_argument('--supercell',
                        nargs=3,
                        type=int,
                        default=[4,4,4],
                        help='supercell size')
    parser.add_argument('--n_calc',
                        type=int,
                        default=20,
                        help='number of calculations')

    args = parser.parse_args()

    if args.dirs is None:
        prop = Properties(pot=args.pot)
    else:
        pot_dirs = sorted(glob.glob(args.dirs + '/*'))
        pot = sorted(glob.glob(pot_dirs[0] + '/polymlp.lammps*'))
        prop = Properties(pot=pot)

    if args.poscar is not None:
        unitcell_dict = Poscar(args.poscar).get_structure()
    else:
        unitcell_dict = dict()
        if isinstance(prop.params_dict, list):
            elements = prop.params_dict[0]['elements']
        else:
            elements = prop.params_dict['elements']
        if len(elements) == 1:
            unitcell_dict['axis'] = np.array([[4,0,0],[0,4,0],[0,0,4]])
            unitcell_dict['positions'] = np.array([[0.0,0.0,0.0],
                                                   [0.0,0.5,0.5],
                                                   [0.5,0.0,0.5],
                                                   [0.5,0.5,0.0]]).T
            unitcell_dict['n_atoms'] = np.array([4])
            unitcell_dict['types'] = np.array([0,0,0,0])
            unitcell_dict['elements'] = [elements[t] 
                                         for t in unitcell_dict['types']]
            unitcell_dict['volume'] = np.linalg.det(unitcell_dict['axis'])
        elif len(elements) == 2:
            unitcell_dict['axis'] = np.array([[4,0,0],[0,4,0],[0,0,4]])
            unitcell_dict['positions'] = np.array([[0.0,0.0,0.0],
                                                   [0.0,0.5,0.5],
                                                   [0.5,0.0,0.5],
                                                   [0.5,0.5,0.0]]).T
            unitcell_dict['n_atoms'] = np.array([2,2])
            unitcell_dict['types'] = np.array([0,0,1,1])
            unitcell_dict['elements'] = [elements[t] 
                                         for t in unitcell_dict['types']]
            unitcell_dict['volume'] = np.linalg.det(unitcell_dict['axis'])
        else:
            raise ValueError('No structure setting for more than binary system')

    supercell_matrix = np.diag(args.supercell)

    unitcell = st_dict_to_phonopy_cell(unitcell_dict)
    phonopy = Phonopy(unitcell, supercell_matrix)
    supercell_dict = phonopy_cell_to_st_dict(phonopy.supercell)

    if args.dirs is None:
        run_single(prop, supercell_dict, args)

    else:
        for dir1 in pot_dirs:
            print('------- Target MLP:', dir1, '-------')
            pot = sorted(glob.glob(dir1 + '/polymlp.lammps*'))
            prop = Properties(pot=pot)
            run_single(prop, supercell_dict, args, 
                       filename=dir1+'/polymlp_cost.yaml')


