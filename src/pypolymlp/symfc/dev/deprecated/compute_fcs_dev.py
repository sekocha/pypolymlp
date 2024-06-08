#!/usr/bin/env python
import argparse
import gc
import signal
import time

import numpy as np
from phono3py.file_IO import write_fc2_to_hdf5, write_fc3_to_hdf5
from symfc.basis_sets.basis_sets_O2 import FCBasisSetO2
from symfc.solvers.solver_O2O3 import run_solver_O2O3, run_solver_O2O3_no_sum_rule_basis
from symfc.spg_reps import SpgRepsO1
from symfc.utils.matrix_tools_O3 import set_complement_sum_rules
from symfc.utils.utils_O3 import get_lat_trans_compr_matrix_O3

from pypolymlp.calculator.properties import Properties
from pypolymlp.calculator.str_opt.optimization_simple import Minimize
from pypolymlp.core.displacements import (
    generate_random_const_displacements,
    get_structures_from_displacements,
)
from pypolymlp.core.interface_phono3py import parse_phono3py_yaml_fcs
from pypolymlp.core.interface_vasp import Poscar
from pypolymlp.symfc.dev.symfc_basis_dev import run_basis
from pypolymlp.utils.phonopy_utils import (
    phonopy_cell_to_st_dict,
    phonopy_supercell,
    st_dict_to_phonopy_cell,
)
from pypolymlp.utils.yaml_utils import load_cells


def recover_fc2(coefs, compress_mat, compress_eigvecs, N):
    n_a = compress_mat.shape[0] // (9 * N)
    n_lp = N // n_a
    fc2 = compress_eigvecs @ coefs
    fc2 = (compress_mat @ fc2).reshape((N, N, 3, 3))
    fc2 /= np.sqrt(n_lp)
    return fc2


def recover_fc3(coefs, compress_mat, compress_eigvecs, N):
    n_a = compress_mat.shape[0] // (27 * (N**2))
    n_lp = N // n_a
    fc3 = compress_eigvecs @ coefs
    fc3 = (compress_mat @ fc3).reshape((n_a, N, N, 3, 3, 3))
    fc3 /= np.sqrt(n_lp)
    return fc3


def recover_fc3_variant(
    coefs,
    compress_mat,
    proj_pt,
    trans_perms,
    n_iter=10,
):
    """if using full compression_matrix
    fc3 = compress_eigvecs @ coefs
    fc3 = (compress_mat @ fc3).reshape((N,N,N,3,3,3))
    """
    n_lp, N = trans_perms.shape
    n_a = compress_mat.shape[0] // (27 * (N**2))

    fc3 = compress_mat @ coefs
    c_sum_cplmt = set_complement_sum_rules(trans_perms)

    for i in range(n_iter):
        fc3 -= c_sum_cplmt.T @ (c_sum_cplmt @ fc3)
        fc3 = proj_pt @ fc3

    fc3 = fc3.reshape((n_a, N, N, 3, 3, 3))
    fc3 /= np.sqrt(n_lp)
    return fc3


def compute_fcs_from_disps_forces(
    disps, forces, supercell, batch_size=100, sum_rule_basis=True
):
    """
    disps: (n_str, 3, n_atom) --> (n_str, n_atom, 3)
    forces: (n_str, 3, n_atom) --> (n_str, n_atom, 3)
    """
    disps = disps.transpose((0, 2, 1))
    forces = np.array(forces).transpose((0, 2, 1))

    n_data, N, _ = forces.shape
    disps = disps.reshape((n_data, -1))
    forces = forces.reshape((n_data, -1))

    """ Constructing fc2 basis and fc3 basis """
    t1 = time.time()
    fc2_basis = FCBasisSetO2(supercell, use_mkl=False).run()
    compress_mat_fc2_full = fc2_basis.compression_matrix
    compress_eigvecs_fc2 = fc2_basis.basis_set

    if sum_rule_basis:
        compress_mat_fc3, compress_eigvecs_fc3 = run_basis(
            supercell,
            apply_sum_rule=True,
        )
    else:
        compress_mat_fc3, proj_pt = run_basis(supercell, apply_sum_rule=False)

    trans_perms = SpgRepsO1(supercell).translation_permutations
    c_trans = get_lat_trans_compr_matrix_O3(trans_perms)
    compress_mat_fc3_full = c_trans @ compress_mat_fc3
    del c_trans
    gc.collect()

    t2 = time.time()
    print(" elapsed time (basis sets for fc2 and fc3) =", t2 - t1)

    print("----- Solving fc2 and fc3 using run_solver -----")
    t1 = time.time()
    use_mkl = False if N > 400 else True
    if sum_rule_basis:
        coefs_fc2, coefs_fc3 = run_solver_O2O3(
            disps,
            forces,
            compress_mat_fc2_full,
            compress_mat_fc3_full,
            compress_eigvecs_fc2,
            compress_eigvecs_fc3,
            use_mkl=use_mkl,
            batch_size=batch_size,
        )
    else:
        coefs_fc2, coefs_fc3 = run_solver_O2O3_no_sum_rule_basis(
            disps,
            forces,
            compress_mat_fc2_full,
            compress_mat_fc3_full,
            compress_eigvecs_fc2,
            use_mkl=use_mkl,
            batch_size=batch_size,
        )
    t2 = time.time()
    print(" elapsed time (solve fc2 + fc3) =", t2 - t1)

    t1 = time.time()
    fc2 = recover_fc2(coefs_fc2, compress_mat_fc2_full, compress_eigvecs_fc2, N)

    if sum_rule_basis:
        fc3 = recover_fc3(coefs_fc3, compress_mat_fc3, compress_eigvecs_fc3, N)
    else:
        print("Applying sum rules to fc3")
        fc3 = recover_fc3_variant(coefs_fc3, compress_mat_fc3, proj_pt, trans_perms)

    t2 = time.time()
    print(" elapsed time (recover fc2 and fc3) =", t2 - t1)

    print("writing fc2.hdf5")
    write_fc2_to_hdf5(fc2)
    print("writing fc3.hdf5")
    write_fc3_to_hdf5(fc3)


def compute_fcs_from_dataset(
    st_dicts,
    disps,
    supercell,
    pot=None,
    params_dict=None,
    coeffs=None,
    geometry_optimization=False,
    batch_size=100,
    sum_rule_basis=True,
):
    """
    Parameters
    ----------
    disps: Displacements (n_str, 3, n_atom)
    supercell: Supercell in phonopy format
    pot or (params_dict and coeffs): polynomal MLP
    """
    if geometry_optimization:
        print("Running geometry optimization")
        supercell_dict = phonopy_cell_to_st_dict(supercell)
        minobj = Minimize(
            supercell_dict, pot=pot, params_dict=params_dict, coeffs=coeffs
        )
        minobj.run(gtol=1e-6)
        print("Residual forces:")
        print(minobj.residual_forces.T)
        print("E0:", minobj.energy)
        print("n_iter:", minobj.n_iter)
        print("Fractional coordinate changes:")
        diff_positions = supercell_dict["positions"] - minobj.structure["positions"]
        print(diff_positions.T)
        print("Success:", minobj.success)

        if minobj.success:
            supercell_dict = minobj.structure
            supercell = st_dict_to_phonopy_cell(supercell_dict)
            st_dicts = get_structures_from_displacements(disps, supercell_dict)

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

    t2 = time.time()
    print(" elapsed time (computing forces) =", t2 - t1)

    compute_fcs_from_disps_forces(
        disps,
        forces,
        supercell,
        batch_size=batch_size,
        sum_rule_basis=sum_rule_basis,
    )


def compute_fcs_from_structure(
    pot=None,
    params_dict=None,
    coeffs=None,
    unitcell_dict=None,
    supercell_matrix=None,
    supercell_dict=None,
    n_samples=100,
    displacements=0.001,
    is_plusminus=False,
    geometry_optimization=False,
    batch_size=100,
    sum_rule_basis=True,
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
        geometry_optimization=geometry_optimization,
        batch_size=batch_size,
        sum_rule_basis=sum_rule_basis,
    )


def compute_fcs_phono3py_dataset(
    pot=None,
    params_dict=None,
    coeffs=None,
    phono3py_yaml=None,
    use_phonon_dataset=False,
    n_samples=None,
    displacements=0.001,
    is_plusminus=False,
    geometry_optimization=False,
    batch_size=100,
    sum_rule_basis=True,
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
        geometry_optimization=geometry_optimization,
        batch_size=batch_size,
        sum_rule_basis=sum_rule_basis,
    )


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

    parser.add_argument("--pot", type=str, default=None, help="polymlp file")
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
        "--geometry_optimization",
        action="store_true",
        help="Geometry optimization is performed " "for initial structure.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Batch size for FC solver.",
    )

    args = parser.parse_args()

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
        geometry_optimization=args.geometry_optimization,
        batch_size=args.batch_size,
        sum_rule_basis=True,
        # sum_rule_basis=False,
    )
