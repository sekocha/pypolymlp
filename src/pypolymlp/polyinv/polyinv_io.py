"""Utility functions for enumerating polynomial invariants."""

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
