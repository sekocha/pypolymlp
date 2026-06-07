"""API functions for enumerating polynomial invariants."""

import io

import numpy as np
from numpy.typing import NDArray

from pypolymlp.polyinv.eig_solver import eigh
from pypolymlp.polyinv.polyinv_io import (
    save_polyinv_coeffs,
    save_polyinv_coeffs_multiple_l,
    save_polyinv_lcombs,
)
from pypolymlp.polyinv.polyinv_utils import get_l_combs
from pypolymlp.polyinv.projector import build_projector


def run_enum(
    orders: list,
    maxl: int = 10,
    minl: int | None = None,
    eliminate_odd: bool = True,
    lproj: int = 0,
    verbose: bool = False,
):
    """Enumerate polynomial invariants."""
    if np.any(np.array(orders) > 6):
        raise RuntimeError("Orders must be lower than or equal to 6.")

    eigvecs_all = []
    lm_indices_all = []
    for order in orders:
        eigvecs, lm_indices = run_enum_single_order(
            order=order,
            maxl=maxl,
            minl=minl,
            eliminate_odd=eliminate_odd,
            lproj=lproj,
            verbose=verbose,
        )
        eigvecs_all.extend(eigvecs)
        lm_indices_all.extend(lm_indices)
    return eigvecs_all, lm_indices_all


def run_enum_single_order(
    order: int,
    maxl: int = 10,
    minl: int | None = None,
    eliminate_odd: bool = True,
    lproj: int = 0,
    verbose: bool = False,
    return_l: bool = False,
):
    """Enumerate polynomial invariants for single order."""
    lcomb_all, n_list = get_l_combs(maxl, order, lproj)

    match = n_list > 0
    if minl is not None:
        match2 = np.any(lcomb_all >= minl, axis=1)
        match = match & match2

    if eliminate_odd:
        match2 = np.sum(lcomb_all, axis=1) % 2 == 0
        match = match & match2

    lcomb_all = lcomb_all[match]
    n_list = n_list[match]

    eigvecs_all, lm_indices_all = [], []
    for lcomb in lcomb_all:
        eigvecs, lm_indices = solve(lcomb, lproj, verbose=verbose)
        eigvecs_all.append(eigvecs)
        lm_indices_all.append(lm_indices)

    if return_l:
        return (eigvecs_all, lm_indices_all, lcomb_all, n_list)
    return (eigvecs_all, lm_indices_all)


def solve(lcomb: list, lproj: int = 0, verbose: bool = False):
    """Solve projector."""
    if lproj != 0:
        # for mproj in range(-lproj, lproj + 1):
        raise RuntimeError("Function solve is available only for lproj = 0.")

    if verbose:
        print("lcomb:", lcomb, flush=True)
        print("Building projector.", flush=True)

    proj, lm_indices = build_projector(lcomb)
    if verbose:
        print("Solving projector.", flush=True)
        print("- Core projector shape:", proj.shape, flush=True)

    eigvecs = eigh(proj, log_level=verbose)
    if verbose:
        print("- Basis shape: ", eigvecs.shape, flush=True)
    return (eigvecs, lm_indices)


def save_l(
    lcomb_all: list,
    n_list: list,
    lproj: int = 0,
    filename: str | io.IOBase = "polyinv_lcombs.yaml",
):
    save_polyinv_lcombs(lcomb_all, n_list, lproj, filename=filename)


def save_coeffs_multiple_l(
    eigvecs: list[NDArray],
    lm_indices: list[NDArray],
    filename: str | io.IOBase = "polyinv_coeffs.yaml",
):
    """Save coefficients of polynomial invariants for multiple l combinations."""
    save_polyinv_coeffs_multiple_l(eigvecs, lm_indices, filename=filename)


def save_coeffs(
    eigvecs: NDArray,
    lm_indices: NDArray,
    filename: str | io.IOBase = "polyinv_coeffs.yaml",
    mode: str = "w",
    tag: str | None = None,
):
    """Save coefficients of polynomial invariants."""
    save_polyinv_coeffs(eigvecs, lm_indices, filename, mode, tag)
