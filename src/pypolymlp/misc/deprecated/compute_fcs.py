#!/usr/bin/env python
import time

import numpy as np
from phono3py.file_IO import write_fc2_to_hdf5, write_fc3_to_hdf5
from symfc.basis_sets.basis_sets_O2 import FCBasisSetO2
from symfc.basis_sets.basis_sets_O3 import FCBasisSetO3
from symfc.solvers.solver_O2O3 import FCSolverO2O3

from pypolymlp.calculator.properties import Properties
from pypolymlp.calculator.str_opt.optimization_simple import Minimize
from pypolymlp.calculator.str_opt.optimization_sym import MinimizeSym
from pypolymlp.core.displacements import (
    generate_random_const_displacements,
    get_structures_from_displacements,
)
from pypolymlp.core.interface_phono3py import parse_phono3py_yaml_fcs
from pypolymlp.utils.phonopy_utils import (
    phonopy_cell_to_st_dict,
    phonopy_supercell,
    st_dict_to_phonopy_cell,
)


def compute_fcs_from_dataset(
    st_dicts,
    disps,
    supercell,
    pot=None,
    params_dict=None,
    coeffs=None,
    geometry_optimization=False,
    geometry_optimization_full=False,
    gtol=1e-4,
    batch_size=200,
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
        try:
            minobj = MinimizeSym(
                supercell_dict,
                pot=pot,
                params_dict=params_dict,
                coeffs=coeffs,
            )
            run_go = True
        except ValueError:
            print("No geomerty optimization is performed.")
            run_go = False

        if run_go:
            minobj.run(gtol=gtol)
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

    elif geometry_optimization_full:
        print("Running geometry optimization")
        supercell_dict = phonopy_cell_to_st_dict(supercell)
        minobj = Minimize(
            supercell_dict, pot=pot, params_dict=params_dict, coeffs=coeffs
        )
        minobj.run(gtol=gtol)
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

    supercell_dict = phonopy_cell_to_st_dict(supercell)
    st_dicts = get_structures_from_displacements(disps, supercell_dict)

    t1 = time.time()
    prop = Properties(pot=pot, params_dict=params_dict, coeffs=coeffs)

    print("Computing forces using polymlp")
    _, forces, _ = prop.eval_multiple(st_dicts)

    """residual forces"""
    print("Eliminating residual forces")
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

    """Constructing fc2 basis and fc3 basis """
    t1 = time.time()
    fc2_basis = FCBasisSetO2(supercell, use_mkl=False).run()
    fc3_basis = FCBasisSetO3(supercell, use_mkl=False).run()
    t2 = time.time()
    print(" elapsed time (basis sets for fc2 and fc3) =", t2 - t1)

    """ Solving fc3 using run_solver_sparse """
    print("-----")
    t1 = time.time()
    solver = FCSolverO2O3([fc2_basis, fc3_basis], use_mkl=False)
    solver.solve(disps, forces, batch_size=batch_size)
    fc2, fc3 = solver.compact_fc
    t2 = time.time()
    print(" elapsed time (solve fc2 + fc3) =", t2 - t1)

    print("writing fc2.hdf5")
    write_fc2_to_hdf5(fc2)
    print("writing fc3.hdf5")
    write_fc3_to_hdf5(fc3)


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
    geometry_optimization=False,
    batch_size=200,
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
    geometry_optimization=False,
    batch_size=200,
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
    )
