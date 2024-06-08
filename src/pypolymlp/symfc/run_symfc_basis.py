#!/usr/bin/env python
import argparse
import signal
import time

import numpy as np
from symfc.basis_sets.basis_sets_O2 import FCBasisSetO2
from symfc.basis_sets.basis_sets_O3 import FCBasisSetO3

from pypolymlp.core.interface_vasp import Poscar
from pypolymlp.utils.phonopy_utils import phonopy_supercell


def run_fc2(supercell):
    """Constructing fc2 basis and fc3 basis"""
    fc2_basis = FCBasisSetO2(supercell, use_mkl=False).run()
    compress_mat_fc2 = fc2_basis.compression_matrix
    compress_eigvecs_fc2 = fc2_basis.basis_set
    return compress_mat_fc2, compress_eigvecs_fc2


def run_fc3(supercell):

    fc3_basis = FCBasisSetO3(supercell, use_mkl=False).run()
    # fc3_basis = FCBasisSetO3(supercell, use_mkl=True).run()
    compress_mat_fc3 = fc3_basis.compression_matrix
    compress_eigvecs_fc3 = fc3_basis.basis_set
    return compress_mat_fc3, compress_eigvecs_fc3


if __name__ == "__main__":

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    parser = argparse.ArgumentParser()
    parser.add_argument("--poscar", type=str, default=None, help="poscar")
    parser.add_argument(
        "--supercell",
        nargs=3,
        type=int,
        default=None,
        help="Supercell size (diagonal components)",
    )
    parser.add_argument(
        "--fc2", action="store_true", help="Calculate only fc2 basis set"
    )
    args = parser.parse_args()

    unitcell_dict = Poscar(args.poscar).get_structure()
    supercell_matrix = np.diag(args.supercell)

    supercell = phonopy_supercell(unitcell_dict, supercell_matrix)

    t1 = time.time()
    compress_mat_fc2, compress_eigvecs_fc2 = run_fc2(supercell)
    print("n_basis (FC2) =", compress_eigvecs_fc2.shape[1])

    if args.fc2 is False:
        compress_mat_fc3, compress_eigvecs_fc3 = run_fc3(supercell)
        print("n_basis (FC3) =", compress_eigvecs_fc3.shape[1])

    t2 = time.time()
    print(" elapsed time (basis sets for fc2 and fc3) =", t2 - t1)
