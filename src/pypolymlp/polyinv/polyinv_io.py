"""Utility functions for enumerating polynomial invariants."""

import io

import numpy as np
from numpy.typing import NDArray


def print_list_nospace(array: list, prefix: str, file: io.IOBase):
    """Print one-dimensional array."""
    print(prefix + " [", end="", file=file)
    for i, d in enumerate(array[:-1]):
        print(d, end=",", file=file)
    print(array[-1], end="", file=file)
    print("]", file=file)


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


def save_polyinv_coeffs_multiple_l(
    eigvecs: list[NDArray],
    lm_indices: list[NDArray],
    filename: str | io.IOBase = "polyinv_coeffs.yaml",
):
    """Save coefficients of polynomial invariants for multiple l combinations."""
    with open(filename, "w") as f:
        print("invariants:", file=f)
        for eig, lm in zip(eigvecs, lm_indices):
            save_polyinv_coeffs(eig, lm, filename=f, mode="a")


def save_polyinv_coeffs(
    eigvecs: NDArray,
    lm_indices: NDArray,
    filename: str | io.IOBase = "polyinv_coeffs.yaml",
    mode: str = "a",
    tag: str | None = None,
):
    """Save coefficients of polynomial invariants."""
    np.set_printoptions(legacy="1.21")
    if isinstance(filename, io.IOBase):
        f = filename
    else:
        f = open(filename, mode)

    if tag is not None:
        print(tag + ":", file=f)

    lcomb = lm_indices[0][:, 0]
    for i, eig in enumerate(eigvecs.T):
        print_list_nospace(lcomb, "- l:", file=f)
        print("  id:", i, file=f)

        print("  m:", file=f)
        for c, lm in zip(eig, lm_indices):
            if abs(c) > 1e-15:
                print_list_nospace(lm[:, 1], "  -", file=f)

        print("  coeffs:", file=f)
        for c in eig:
            if abs(c) > 1e-15:
                print("  -", c, file=f)
        print(file=f)

    if not isinstance(filename, io.IOBase):
        f.close()
