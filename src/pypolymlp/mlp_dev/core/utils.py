"""Utility function for polymlp development."""

import os
from typing import Union

import numpy as np

from pypolymlp.core.data_format import PolymlpDataDFT, PolymlpParams
from pypolymlp.core.parser_polymlp_params import set_common_params


def set_params(params_in: Union[PolymlpParams, list[PolymlpParams]]):
    """Set parameters, hybrid parameters, and common parameters."""
    if isinstance(params_in, (list, tuple, np.ndarray)):
        if len(params_in) > 1:
            params = hybrid_params = params_in
            common_params = set_common_params(params_in)
        else:
            params = common_params = params[0]
            hybrid_params = None
    else:
        params = common_params = params_in
        hybrid_params = None
    return (params, common_params, hybrid_params)


def get_min_energy(dft_all: list[PolymlpDataDFT]) -> float:
    """Calculate minimum of DFT energies."""
    min_e = 1e10
    for dft in dft_all:
        e_per_atom = dft.energies / dft.total_n_atoms
        min_e_trial = np.min(e_per_atom)
        if min_e_trial < min_e:
            min_e = min_e_trial
    return min_e


def check_memory_size_in_regression(n_features: int, verbose: bool = False):
    """Estimate memory size in regression."""
    mem_req = np.round(n_features**2 * 8e-9 * 2, 1)
    mem_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") * 1e-9
    if mem_req > mem_bytes:
        print("Minimum memory required for solver in GB:", mem_req, flush=True)
        raise RuntimeError("Larger size of memory required.")

    if verbose:
        print("Minimum memory required for solver in GB:", mem_req, flush=True)
        print("Memory required for allocating X additionally.", flush=True)
    return mem_req
