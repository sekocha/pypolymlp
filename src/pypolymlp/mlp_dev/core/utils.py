"""Utility function for polymlp development."""

from typing import Union

import numpy as np

from pypolymlp.core.data_format import PolymlpDataDFT, PolymlpParams
from pypolymlp.core.parser_polymlp_params import set_common_params


def set_params(params_in: Union[PolymlpParams, list[PolymlpParams]]):
    """Set parameters, hybrid parameters, and common parameters."""
    if isinstance(params_in, list):
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


def round_scales(
    scales: np.ndarray, include_force: bool = True, threshold: float = 1e-10
):
    """Set scales so that they are not used for zero features."""
    if include_force:
        zero_ids = np.abs(scales) < threshold
    else:
        # Threshold value can be improved.
        zero_ids = np.abs(scales) < threshold * threshold
    scales[zero_ids] = 1.0
    return scales, zero_ids
