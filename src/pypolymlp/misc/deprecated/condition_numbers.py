#!/usr/bin/env python
import argparse
import signal
import time
from math import sqrt

import numpy as np
from symfc.basis_sets.basis_sets_O2 import FCBasisSetO2
from symfc.basis_sets.basis_sets_O3 import FCBasisSetO3
from symfc.solvers.solver_O2O3 import get_training_exact

from pypolymlp.calculator.properties import Properties
from pypolymlp.core.displacements import (  # get_structures_from_displacements,
    generate_random_const_displacements,
)
from pypolymlp.core.interface_phono3py import parse_phono3py_yaml_fcs
from pypolymlp.core.interface_vasp import Poscar
from pypolymlp.utils.phonopy_utils import (
    phonopy_cell_to_st_dict,
    phonopy_supercell,
    st_dict_to_phonopy_cell,
)
from pypolymlp.utils.yaml_utils import load_cells


def calc_symmetric_matrix_norm(A):
    eigvals = np.linalg.eigvalsh(A)
    for i, e in enumerate(eigvals):
        print(" v", i, " =", e)
    print(np.where(eigvals < 1e-4)[0])
    print(" n_near_zero =", len(np.where(eigvals < 1e-4)[0]))
    return max(eigvals), min(eigvals)


def compute_fcs_from_dataset(
    st_dicts, disps, supercell, pot=None, params_dict=None, coeffs=None
):
    t1 = time.time()
    prop = Properties(pot=pot, params_dict=params_dict, coeffs=coeffs)

    print("Computing forces using polymlp")
    _, forces, _ = prop.eval_multiple(st_dicts)

    """residual forces"""
    print("Eliminating residual forces")
    supercell_dict = phonopy_cell_to_st_dict(supercell)
    _, residual_forces, _ = prop.eval(supercell_dict)
    for f in forces:
        f -= residual_forces

    """
    disps: (n_str, 3, n_atom) --> (n_str, n_atom, 3)
    forces: (n_str, 3, n_atom) --> (n_str, n_atom, 3)
    """
    disps = disps.transpose((0, 2, 1))
    forces = np.array(forces).transpose((0, 2, 1))
    t2 = time.time()
    print(" elapsed time (computing forces) =", t2 - t1)

    n_data, N, _ = forces.shape
    disps = disps.reshape((n_data, -1))
    forces = forces.reshape((n_data, -1))

    """ Constructing fc2 basis and fc3 basis """
    t1 = time.time()
    fc2_basis = FCBasisSetO2(supercell, use_mkl=False).run()
    compress_mat_fc2 = fc2_basis.compression_matrix
    compress_eigvecs_fc2 = fc2_basis.basis_set

    fc3_basis = FCBasisSetO3(supercell, use_mkl=False).run()
    compress_mat_fc3 = fc3_basis.compression_matrix
    compress_eigvecs_fc3 = fc3_basis.basis_set
    t2 = time.time()
    print(" elapsed time (basis sets for fc2 and fc3) =", t2 - t1)

    XTX, XTy = get_training_exact(
        disps,
        forces,
        compress_mat_fc2,
        compress_mat_fc3,
        compress_eigvecs_fc2,
        compress_eigvecs_fc3,
        batch_size=200,
        use_mkl=True,
    )
    norm0 = calc_symmetric_matrix_norm(XTX)
    print(norm0)

    XTX_inv = np.linalg.inv(XTX)

    mat1 = XTX.T @ XTX
    mat2 = XTX_inv.T @ XTX_inv

    max1, min1 = calc_symmetric_matrix_norm(mat1)
    print(max1 / min1)

    norm1, _ = calc_symmetric_matrix_norm(mat1)
    print(sqrt(norm1))
    norm2, _ = calc_symmetric_matrix_norm(mat2)
    print(sqrt(norm2))
    print(sqrt(norm1 * norm2))


def compute_fcs_from_structure(
    pot=None,
    params_dict=None,
    coeffs=None,
    unitcell_dict=None,
    supercell_matrix=None,
    supercell_dict=None,
    n_samples=100,
    displacements=0.03,
    is_plusminus=False,
):

    if supercell_dict is not None:
        supercell = st_dict_to_phonopy_cell(supercell_dict)
    elif unitcell_dict is not None:
        supercell = phonopy_supercell(unitcell_dict, supercell_matrix)
        supercell_dict = phonopy_cell_to_st_dict(supercell)
    else:
        raise ValueError(
            "(unitcell_dict, supercell_matrix) " "or supercell_dict is requried."
        )

    disps, st_dicts = generate_random_const_displacements(
        supercell_dict,
        n_samples=n_samples,
        displacements=displacements,
        is_plusminus=is_plusminus,
    )
    compute_fcs_from_dataset(
        st_dicts,
        disps,
        supercell,
        pot=pot,
        params_dict=params_dict,
        coeffs=coeffs,
    )


def compute_fcs_phono3py_dataset(
    pot=None,
    params_dict=None,
    coeffs=None,
    phono3py_yaml=None,
    use_phonon_dataset=False,
    n_samples=None,
    displacements=0.03,
    is_plusminus=False,
):

    supercell, disps, st_dicts = parse_phono3py_yaml_fcs(
        phono3py_yaml, use_phonon_dataset=use_phonon_dataset
    )

    if n_samples is not None:
        supercell_dict = phonopy_cell_to_st_dict(supercell)
        disps, st_dicts = generate_random_const_displacements(
            supercell_dict,
            n_samples=n_samples,
            displacements=displacements,
            is_plusminus=is_plusminus,
        )

    compute_fcs_from_dataset(
        st_dicts,
        disps,
        supercell,
        pot=pot,
        params_dict=params_dict,
        coeffs=coeffs,
    )


if __name__ == "__main__":

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    parser = argparse.ArgumentParser()
    parser.add_argument("--pot", type=str, default=None, help="polymlp file")

    parser.add_argument(
        "--poscar", nargs="*", type=str, default=None, help="poscar files"
    )
    parser.add_argument(
        "--supercell",
        nargs=3,
        type=int,
        default=None,
        help="Supercell size (diagonal components)",
    )
    parser.add_argument(
        "--str_yaml", type=str, default=None, help="polymlp_str.yaml file"
    )

    parser.add_argument(
        "--fc_n_samples",
        type=int,
        default=None,
        help="Number of random displacement samples",
    )
    parser.add_argument(
        "--disp",
        type=float,
        default=0.03,
        help="Displacement (in Angstrom)",
    )
    parser.add_argument(
        "--is_plusminus",
        action="store_true",
        help="Plus-minus displacements will be generated.",
    )

    parser.add_argument(
        "--phono3py_yaml", type=str, default=None, help="phono3py.yaml file"
    )
    args = parser.parse_args()

    print("Mode: Condition number calculations")

    if args.phono3py_yaml is not None:
        compute_fcs_phono3py_dataset(
            pot=args.pot,
            phono3py_yaml=args.phono3py_yaml,
            use_phonon_dataset=False,
            n_samples=args.fc_n_samples,
            displacements=args.disp,
            is_plusminus=args.is_plusminus,
        )

    else:
        if args.str_yaml is not None:
            _, supercell_dict = load_cells(filename=args.str_yaml)
            unitcell_dict = None
            supercell_matrix = None
        elif args.poscar is not None:
            unitcell_dict = Poscar(args.poscar).get_structure()
            supercell_matrix = np.diag(args.supercell)
            supercell_dict = None

        compute_fcs_from_structure(
            pot=args.pot,
            unitcell_dict=unitcell_dict,
            supercell_dict=supercell_dict,
            supercell_matrix=supercell_matrix,
            n_samples=args.fc_n_samples,
            displacements=args.disp,
            is_plusminus=args.is_plusminus,
        )
