"""Utilities for harmonic calculation."""

import numpy as np
import scipy

from pypolymlp.calculator.utils.fc_utils import eval_properties_fc2
from pypolymlp.core.units import Kb, Planck

"""
Constants
---------
const_amplitude: J*s/THz -> atomic_mass * angstrom^2
    6.62607015e-34 * 6.02214076e23 * 1e11 / (4 * pi * pi)

const_planck is set to 6.62607015e-34 * 1e11 = 6.62607015e-22.
"""
const_planck = Planck * 1e12  # = 6.62607015e-22
const_amplitude = 1.010758017933576


def mask_imaginary_modes(freq: np.ndarray, freq_threshold: float = 0.1):
    """Mask branches with imaginary frequencies."""
    freq_rev = np.array(freq)
    freq_rev[np.where(freq_rev < freq_threshold)] = 0.0
    return freq_rev


def sample_real_space_distribution(
    mesh_dict: dict,
    masses: np.ndarray,
    temp: float = 1000,
    n_samples: int = 100,
):
    """Calculate atomic real-space distribution from density matrix."""
    if "frequencies" not in mesh_dict:
        raise RuntimeError("frequencies not found in mesh_dict.")
    if "eigenvectors" not in mesh_dict:
        raise RuntimeError("eigenvectors not found in mesh_dict.")

    freq = mask_imaginary_modes(mesh_dict["frequencies"])
    nonzero = np.isclose(freq, 0.0) == False

    beta = np.inf if np.isclose(temp, 0.0) else 1.0 / (Kb * temp)
    const_exp = 0.5 * beta * const_planck
    beta_h_freq = const_exp * freq[nonzero]

    # Calculate occupancies.
    occ = np.zeros(freq.shape)
    occ[nonzero] = 0.5 * np.reciprocal(np.tanh(beta_h_freq))

    # Calculate amplitudes.
    # Arbitrary setting for branches with low and imaginary frequencies.
    amplitudes = np.ones(freq.shape) * 0.01
    rec_freq = np.array([1 / f for f in freq[nonzero]])
    amplitudes[nonzero] = const_amplitude * occ[nonzero] * rec_freq

    # Generate atomic displacements in normal coordinates.
    cov = np.diag(amplitudes)
    mean = np.zeros(cov.shape[0])
    disp_normal_coords = np.random.multivariate_normal(mean, cov, n_samples)

    # Generate atomic displacements.
    eigvecs = mesh_dict["eigenvectors"]
    masses_sqrt = np.sqrt(np.repeat(masses, 3))
    dot1 = eigvecs @ disp_normal_coords.T
    disps = (np.diag(np.reciprocal(masses_sqrt)) @ dot1).T
    disps = disps.reshape((n_samples, -1, 3)).transpose((0, 2, 1))
    return disps


def eliminate_outliers(
    disps: np.ndarray,
    supercells: list,
    energies_full: np.ndarray,
    forces: np.ndarray,
    stress_tensors: np.ndarray,
    tol_negative: float = -10,
    tol_n_samples: int = 30,
    ialgo: int = 2,
    verbose: bool = False,
):
    """Eliminate outliers."""
    if len(disps) != len(supercells):
        raise RuntimeError("Size mismatch in disps and supercells.")
    if len(disps) != len(energies_full):
        raise RuntimeError("Size mismatch in disps and energies.")
    if len(disps) != len(forces):
        raise RuntimeError("Size mismatch in disps and forces.")
    if len(disps) != len(stress_tensors):
        raise RuntimeError("Size mismatch in disps and stress")

    energies = np.array(energies_full)
    ids1 = np.where(energies > tol_negative)[0]
    e_ave = np.mean(energies[ids1])
    e_std = np.std(energies[ids1])

    if len(ids1) > tol_n_samples:
        ialgo = 2
        if ialgo == 1:
            tol = 2 * abs(e_ave)
            ids2 = np.where(np.abs(energies - e_ave) < tol)[0]
        elif ialgo == 2:
            ub = 5 * e_std + e_ave
            lb = -5 * e_std + e_ave
            ids2 = np.where((lb < energies) & (energies < ub))[0]
        else:
            raise RuntimeError("No algorithm found.")
        ids = set(ids1) & set(ids2)
    else:
        ids = set(ids1)

    if verbose:
        outlier_ids = set(range(len(energies))) - ids
        if len(outlier_ids) > 0:
            print("Outliers are eliminated.")
            print("- Average potential energy: ", "{:f}".format(e_ave))
            print("- Std.Dev. potential energy:", "{:f}".format(e_std))
            for i in sorted(outlier_ids):
                prefix = "- Potential energy (outlier " + str(i) + "):"
                print(prefix, "{:f}".format(energies[i]), flush=True)

    ids = np.array(list(ids))
    disps = disps[ids]
    supercells = [supercells[i] for i in ids]
    energies_full = energies_full[ids]
    forces = forces[ids]
    stress_tensors = stress_tensors[ids]
    return (disps, supercells, energies_full, forces, stress_tensors)


def eval_harmonic_properties(disps: np.ndarray, fc2: np.ndarray):
    """Calculate harmonic potentials and properties from displacements.

    Parameters
    ----------
    disps: Displacements. shape=(n_str, 3, n_atom).
    fc2: Force constants. shape=(n_atom, n_atom, 3, 3).
    """
    if disps is None:
        raise RuntimeError("Displacements not found.")
    if fc2 is None:
        raise RuntimeError("Force constants not found.")

    N3 = fc2.shape[0] * 3
    fc2mat = fc2.transpose((0, 2, 1, 3)).reshape((N3, N3))
    harmonic_potentials, harmonic_forces, harmonic_stress_tensors = [], [], []
    for d in disps:
        e, f, s = eval_properties_fc2(fc2mat, d.T.reshape(-1))
        harmonic_potentials.append(e)
        harmonic_forces.append(f)
        harmonic_stress_tensors.append(s)

    return (
        np.array(harmonic_potentials),
        np.array(harmonic_forces),
        np.array(harmonic_stress_tensors),
    )


def reduce_dynamical_matrix(dyn: np.ndarray, null_space_basis: np.ndarray):
    """Reduce null space component from dynamical matrix.

    Compute B @ [M/(C.T @ D @ C)] @ B.T.

    B: Basis set for complement of null space.
    C: Basis set for null space.
    D: Dynamical matrix.
    M: [[C.T @ D @ C, C.T @ D @ B],
        [B.T @ D @ C, B.T @ D @ B]]
    M/A: Schur complement of A in M.

    M/(C.T @ D @ C) =
        (B.T @ D @ B) - (B.T @ D @ C) @ (C.T @ D @ C)^{-1} @ (C.T @ D @ B)
    B @ [M/(C.T @ D @ C)] @ B.T =
        P_B @ D @ P_B - P_B @ (D @ C) @ (C.T @ D @ C)^{-1} @ (C.T @ D) @ P_B

    inv = np.linalg.inv(null_space_basis.T @ dyn @ null_space_basis)
    prod = dyn @ null_space_basis @ inv @ null_space_basis.T @ dyn

    inv @ null_space_basis.T @ dyn
        = solve(null_space_basis.T @ dyn @ null_space_basis, null_space_basis.T @ dyn)
    """
    N3 = null_space_basis.shape[0]
    if dyn.shape[0] != N3:
        raise RuntimeError(
            "Shape mismatch error (Dynamical matrix and null space basis)."
        )
    if dyn.shape[1] != N3:
        raise RuntimeError(
            "Shape mismatch error (Dynamical matrix and null space basis)."
        )

    proj = np.eye(N3) - null_space_basis @ null_space_basis.T

    mat2 = null_space_basis.T @ dyn
    mat1 = mat2 @ null_space_basis
    prod = scipy.linalg.solve(mat1, mat2)
    prod = mat2.T @ prod

    reduced_dyn = proj @ (dyn - prod) @ proj
    return reduced_dyn
