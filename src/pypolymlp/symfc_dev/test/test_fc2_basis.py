#!/usr/bin/env python
import numpy as np
import argparse
import time

from pypolymlp.symfc_dev.cell import poscar_to_supercell
from pypolymlp.symfc_dev.cell import st_dict_to_phonony

from pypolymlp.symfc_dev.basis_set_O2 import run_fc2, run_fc2_symfc

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--poscar',
                        type=str,
                        default='POSCAR',
                        help='poscar file name')
    parser.add_argument('--supercell',
                        type=int,
                        nargs=9,
                        default=[2,0,0,0,2,0,0,0,2],
                        help='Supercell size')
    parser.add_argument('--mkl',
                        action='store_true',
                        help='use mkl')
    args = parser.parse_args()

    supercell_mat = np.array(args.supercell).reshape([3,3])
    unitcell, supercell = poscar_to_supercell(args.poscar, supercell_mat)
    supercell_phonopy = st_dict_to_phonony(supercell)

    t1 = time.time()
    compress_mat, compress_eigvecs = run_fc2(supercell_phonopy, mkl=args.mkl)
    t2 = time.time()
    print(' elapsed time (rot + trans + perm + sum) :', t2-t1, '(s)')

#    run_fc2_symfc(supercell_phonopy)


