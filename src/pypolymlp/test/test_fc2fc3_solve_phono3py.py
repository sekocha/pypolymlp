import sys
import time

import numpy as np
import phono3py
import phonopy
from phono3py.file_IO import write_fc2_to_hdf5, write_fc3_to_hdf5
from symfc.basis_sets import FCBasisSetO2, FCBasisSetO3
from symfc.solvers.solver_O2 import run_solver_sparse_O2
from symfc.solvers.solver_O2O3 import run_solver_sparse_O2O3


def parse_dataset_phono3py_xz(filename):
    ph3 = phono3py.load(filename, produce_fc=False, log_level=1)
    supercell = ph3.supercell

    n_data = ph3.dataset["displacements"].shape[0]
    disps = ph3.dataset["displacements"].reshape((n_data, -1))
    forces = ph3.dataset["forces"].reshape((n_data, -1))

    n_data = ph3.phonon_dataset["displacements"].shape[0]
    ph_disps = ph3.phonon_dataset["displacements"].reshape((n_data, -1))
    ph_forces = ph3.phonon_dataset["forces"].reshape((n_data, -1))
    ph_supercell = ph3.phonon_supercell

    ph3_init = phono3py.Phono3py(
        ph3.unitcell,
        supercell_matrix=ph3.supercell_matrix,
        primitive_matrix=ph3.primitive_matrix,
        phonon_supercell_matrix=ph3.phonon_supercell_matrix,
    )
    ph3_init.nac_params = ph3.nac_params
    ph3_init.save("phono3py_init.yaml")
    ph_init = phonopy.Phonopy(
        ph3.unitcell,
        supercell_matrix=ph3.phonon_supercell_matrix,
        primitive_matrix=ph3.primitive_matrix,
    )
    ph_init.nac_params = ph3.nac_params
    ph_init.save("phonopy_init.yaml")

    return disps, forces, supercell, ph_disps, ph_forces, ph_supercell


def recover_fc2(coefs, compress_mat, compress_eigvecs, N):
    fc2 = compress_eigvecs @ coefs
    fc2 = (compress_mat @ fc2).reshape((N, N, 3, 3))
    return fc2


def recover_fc3(coefs, compress_mat, compress_eigvecs, N):
    fc3 = compress_eigvecs @ coefs
    fc3 = (compress_mat @ fc3).reshape((N, N, N, 3, 3, 3))
    return fc3


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    filename = sys.argv[1]
    (
        disps,
        forces,
        supercell,
        ph_disps,
        ph_forces,
        ph_supercell,
    ) = parse_dataset_phono3py_xz(filename)

    """ Constructing fc2 basis """
    t1 = time.time()
    basis_set_o2 = FCBasisSetO2(supercell, use_mkl=False).run()
    compress_mat_fc2 = basis_set_o2.compression_matrix
    compress_eigvecs_fc2 = basis_set_o2.basis_set
    t2 = time.time()
    print(" elapsed time (basis fc2) =", t2 - t1)

    """ Constructing fc3 basis """
    t1 = time.time()
    basis_set_o3 = FCBasisSetO3(supercell, use_mkl=False).run()
    compress_mat_fc3 = basis_set_o3.compression_matrix
    compress_eigvecs_fc3 = basis_set_o3.basis_set
    t2 = time.time()
    print(" elapsed time (basis fc3) =", t2 - t1)

    """ Solving fc3 using run_solver_sparse """
    print("-----")
    t1 = time.time()
    coefs_fc2, coefs_fc3 = run_solver_sparse_O2O3(
        disps,
        forces,
        compress_mat_fc2,
        compress_mat_fc3,
        compress_eigvecs_fc2,
        compress_eigvecs_fc3,
        batch_size=200,
        use_mkl=False,
    )
    t2 = time.time()
    print(" elapsed time (solve fc2 + fc3) =", t2 - t1)

    # fc2 = recover_fc2(coefs_fc2, compress_mat_fc2, compress_eigvecs_fc2, len(supercell))  # noqa
    # write_fc2_to_hdf5(fc2)
    fc3 = recover_fc3(coefs_fc3, compress_mat_fc3, compress_eigvecs_fc3, len(supercell))
    write_fc3_to_hdf5(fc3)

    """ Constructing phonon-fc2 basis """
    t1 = time.time()
    ph_basis_set_o2 = FCBasisSetO2(ph_supercell, use_mkl=False).run()
    ph_compress_mat_fc2 = ph_basis_set_o2.compression_matrix
    ph_compress_eigvecs_fc2 = ph_basis_set_o2.basis_set
    t2 = time.time()
    print(" elapsed time (basis phonon-fc2) =", t2 - t1)

    """ Solvin phonon-fc2 using run_solver_sparse_O2 """
    ph_coefs = run_solver_sparse_O2(
        ph_disps, ph_forces, ph_compress_mat_fc2, ph_compress_eigvecs_fc2
    )
    fc2 = recover_fc2(
        ph_coefs,
        ph_compress_mat_fc2,
        ph_compress_eigvecs_fc2,
        len(ph_supercell),
    )
    write_fc2_to_hdf5(fc2)
