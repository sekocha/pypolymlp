#!/usr/bin/env python
import argparse
import signal
import time

import numpy as np
from symfc.basis_sets.basis_sets_O4 import FCBasisSetO4

from pypolymlp.core.interface_vasp import Poscar
from pypolymlp.utils.phonopy_utils import phonopy_supercell

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
        "--cutoff",
        type=float,
        default=None,
        help="Cutoff radius for setting zero elements.",
    )
    args = parser.parse_args()

    unitcell_dict = Poscar(args.poscar).get_structure()
    supercell_matrix = np.diag(args.supercell)

    supercell = phonopy_supercell(unitcell_dict, supercell_matrix)

    t1 = time.time()
    fc4_basis = FCBasisSetO4(supercell, cutoff=args.cutoff, use_mkl=True, log_level=1)
    fc4_basis.run()
    t2 = time.time()
    print("Elapsed time (basis sets for fc4) =", t2 - t1)
