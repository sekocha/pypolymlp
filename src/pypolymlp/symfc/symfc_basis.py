#!/usr/bin/env python 
import numpy as np
import argparse
import signal
import time

from pypolymlp.core.interface_vasp import Poscar
from pypolymlp.utils.phonopy_utils import phonopy_supercell

from symfc.basis_sets.basis_sets_O2 import FCBasisSetO2
from symfc.basis_sets.basis_sets_O3 import FCBasisSetO3


def compute_fc_basis_stable(supercell):

    ''' Constructing fc2 basis and fc3 basis '''
    t1 = time.time()
    fc2_basis = FCBasisSetO2(supercell, use_mkl=False).run()
    compress_mat_fc2 = fc2_basis.compression_matrix
    compress_eigvecs_fc2 = fc2_basis.basis_set
    print('n_basis (FC2) =', compress_eigvecs_fc2.shape[1])

    fc3_basis = FCBasisSetO3(supercell, use_mkl=False).run()
    #fc3_basis = FCBasisSetO3(supercell, use_mkl=True).run()
    compress_mat_fc3 = fc3_basis.compression_matrix
    compress_eigvecs_fc3 = fc3_basis.basis_set
    t2 = time.time()
    print(' elapsed time (basis sets for fc2 and fc3) =', t2-t1)
    print('n_basis (FC3) =', compress_eigvecs_fc3.shape[1])



if __name__ == '__main__':

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    parser = argparse.ArgumentParser()
    parser.add_argument('--poscar',
                        type=str,
                        default=None,
                        help='poscar')
    parser.add_argument('--supercell',
                        nargs=3,
                        type=int,
                        default=None,
                        help='Supercell size (diagonal components)')
    args = parser.parse_args()

    unitcell_dict = Poscar(args.poscar).get_structure()
    supercell_matrix = np.diag(args.supercell)

    supercell = phonopy_supercell(unitcell_dict, supercell_matrix)
    compute_fc_basis_stable(supercell)


