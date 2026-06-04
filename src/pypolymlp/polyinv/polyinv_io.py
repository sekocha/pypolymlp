"""Utility functions for enumerating polynomial invariants."""

import io

import numpy as np
from numpy.typing import NDArray


def save_polyinv_lcombs(
    lcomb_all: NDArray,
    n_list: NDArray,
    lproj: int = 0,
    filename: str = "polyinv_angular.yaml",
):
    """Save combinations of l values and number of basis vectors."""
    np.set_printoptions(legacy="1.21")
    with open(filename, "w") as f:
        print("combinations:", file=f)
        for n, comb in zip(n_list, lcomb_all):
            print("- l:  ", list(comb), file=f)
            print("  num:", n, file=f)
            print(file=f)


def save_polyinv_coeffs(
    eigvecs: NDArray,
    lm_indices: NDArray,
    filename: str | io.IOBase = "polyinv_coeffs.yaml",
    mode: str = "a",
):
    """Save coefficients of polynomial invariants."""
    np.set_printoptions(legacy="1.21")
    if isinstance(filename, io.IOBase):
        f = filename
    else:
        f = open(filename, mode)

    lcomb = lm_indices[0][:, 0]
    for i, eig in enumerate(eigvecs.T):
        print("- l: ", list(lcomb), file=f)
        print("  id:", i, file=f)
        print("  coeffs:", file=f)
        for c, lm in zip(eig, lm_indices):
            if abs(c) > 1e-15:
                print("  - c:", c, file=f)
                print("    m:", list(lm[:, 1]), file=f)
        print(file=f)

    if not isinstance(filename, io.IOBase):
        f.close()
