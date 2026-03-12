"""Utility function for polymlp development."""

import os

import numpy as np

from pypolymlp.core.dataset import DatasetList


def get_min_energy(datasets: DatasetList) -> float:
    """Calculate minimum of DFT energies."""
    min_e = 1e10
    for data in datasets:
        if len(data.energies) == 0:
            raise RuntimeError("Empty energy data.")

        e_per_atom = data.energies / data.total_n_atoms
        min_e_trial = np.min(e_per_atom)
        if min_e_trial < min_e:
            min_e = min_e_trial
    return min_e


def check_memory_size_in_regression(
    n_features: int,
    use_gradient: bool = False,
    verbose: bool = False,
):
    """Estimate memory size in regression."""
    if use_gradient:
        mem_req = np.round(n_features**2 * 8e-9 * 1.1, 1)
    else:
        mem_req = np.round(n_features**2 * 8e-9 * 2, 1)

    if verbose:
        print("n_features:", n_features, flush=True)
        if use_gradient:
            text = "Minimum memory required for X.T @ X in gradient-based solver in GB:"
        else:
            text = "Minimum memory required for Cholesky solver in GB:"
        print(text, mem_req, flush=True)

    mem_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") * 1e-9
    if mem_req > mem_bytes:
        if verbose and use_gradient:
            print("Failed to allocate X.T @ X. Use X directly.", flush=True)
        raise RuntimeError("Larger amount of memory is required.")

    if verbose:
        print("Memory required for allocating X additionally.", flush=True)
    return mem_req
