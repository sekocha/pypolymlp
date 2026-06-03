"""API functions for enumerating polynomial invariants."""

import numpy as np
from symfc.api_symfc import eigsh

# from pypolymlp.polyinv.polyinv_io import save_polyinv_lcombs
from pypolymlp.polyinv.polyinv_utils import get_l_combs, get_m_combs
from pypolymlp.polyinv.projector import build_projector


def run_enum(
    orders: list,
    maxl: int = 10,
    minl: int | None = None,
    lproj: int = 0,
    filename: str = "polyinv_angular.yaml",
    verbose: bool = False,
):
    """Enumerate polynomial invariants."""
    if np.any(np.array(orders) > 6):
        raise RuntimeError("Orders must be lower than or equal to 6.")

    for order in orders:
        run_enum_single_order(order, maxl, minl, lproj, verbose=verbose)


def run_enum_single_order(
    order: int,
    maxl: int = 10,
    minl: int | None = None,
    lproj: int = 0,
    verbose: bool = False,
):
    """Enumerate polynomial invariants for single order."""

    lcomb_all, n_list = get_l_combs(maxl, order, lproj)
    match = n_list > 0
    if minl is not None:
        match2 = np.any(lcomb_all >= minl, axis=1)
        match = match & match2

    lcomb_all = lcomb_all[match]
    n_list = n_list[match]
    # save_polyinv_lcombs(lcomb_all, n_list, lproj, filename=filename)
    # print(lcomb_all)
    for lcomb in lcomb_all:
        solve(lcomb, lproj, verbose=verbose)


def solve(lcomb: list, lproj: int = 0, verbose: bool = False):
    """Solve projector."""
    if lproj != 0:
        raise RuntimeError("Function solve is available only for lproj = 0.")
    mcomb_all = get_m_combs(lcomb, lproj=lproj)
    # for mproj in range(-lproj, lproj + 1):
    mproj = 0
    if verbose:
        print("lcomb:", lcomb, flush=True)
        print("- lp_mp:", [lproj, mproj], flush=True)
        print("Building projector.", flush=True)

    proj, lm_indices = build_projector(lcomb, mcomb_all)
    if verbose:
        print(lm_indices.shape)
        print("Solving projector.")
    eigvecs = eigsh(proj, log_level=verbose)
    print(eigvecs.shape)
