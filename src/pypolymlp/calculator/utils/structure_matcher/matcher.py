"""Utilities for structure matchers"""

import itertools
import time

import numpy as np


def _argsort_2d(array: np.ndarray):
    """Returns the indices that sort rows in numpy 2D array."""
    return np.lexsort(array.T[::-1])


def _sort_positions(positions: np.array, n_atoms: np.ndarray):
    """Sort positions with respect to fractional coordinate numbers."""
    ibegin = 0
    for n in n_atoms:
        iend = ibegin + n
        ids = _argsort_2d(positions[:, ibegin:iend].T) + ibegin
        positions[:, ibegin:iend] = positions[:, ids]
        ibegin = iend
    return positions


def _round_positions(
    positions: np.ndarray,
    tol: float = 1e-13,
    decimals: int = 5,
    use_center: bool = False,
):
    """Round fractional coordinates of positions (-0.5 <= p < 0.5)."""
    positions_rint = np.round(positions - np.rint(positions), decimals)
    positions_rint[np.abs(positions_rint - 0.5) < tol] = -0.5
    if use_center:
        center = np.average(positions_rint, axis=1)
        for rpos in positions_rint.T:
            rpos -= center
    return positions_rint


def _apply_origin_shifts_1d(pos1d: np.ndarray, tol: float = 1e-13, decimals: int = 5):
    """Return 1D position candidates where origin shifts are applied."""
    pos = pos1d - np.average(pos1d)
    natom = pos.shape[0]
    shifts = np.arange(0.0, 1.0, 1.0 / natom)
    candidates = pos[None, :] + shifts[:, None]
    candidates = _round_positions(candidates, tol=tol, decimals=decimals)

    candidates_sorted = np.sort(candidates, axis=1)
    rep_id = _argsort_2d(candidates_sorted)[0]
    diff = candidates_sorted - candidates_sorted[rep_id]
    rep_ids = np.where(np.linalg.norm(diff, axis=1) < tol)[0]
    return candidates[rep_ids]


def normalize_positions(
    positions: np.ndarray, n_atoms: np.ndarray, tol: float = 1e-13, decimals: int = 5
):
    """Get an irreducible representation of fractional coordinates.

    Return
    ------
    positions_irrep: Irreducible representation of fractional coordinates.
                     shape=(3, natom)

    Algorithm
    ---------
    1. Transform fractional coordinates to -0.5 <= x_a, x_b, x_c < 0.5.
    2. Calculate irreducible representation for the set of fractional coordinates
        along each axis.
    3. Sort the irreducible representation of fractional coordinates.
    """
    rpositions = _round_positions(
        positions,
        tol=tol,
        decimals=decimals,
        use_center=True,
    )
    candidates_each_axis = [
        _apply_origin_shifts_1d(pos1d, tol=tol, decimals=decimals)
        for pos1d in rpositions
    ]

    cands = np.array(
        [
            _sort_positions(np.array(rpos), n_atoms).reshape(-1)
            for rpos in itertools.product(*candidates_each_axis)
        ]
    )
    positions_irrep = cands[_argsort_2d(cands)[0]].reshape((3, -1))
    return positions_irrep


if __name__ == "__main__":

    n_atoms = [16]
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.25, 0.5, 0.0],
            [0.75, 0.5, 0.0],
            [0.46, 0.0, 0.25],
            [0.96, 0.0, 0.25],
            [0.21, 0.5, 0.25],
            [0.71, 0.5, 0.25],
            [0.42, 0.0, 0.5],
            [0.92, 0.0, 0.5],
            [0.17, 0.5, 0.5],
            [0.67, 0.5, 0.5],
            [0.21, 0.0, 0.75],
            [0.71, 0.0, 0.75],
            [0.46, 0.5, 0.75],
            [0.96, 0.5, 0.75],
        ]
    ).T

    positions[0] += 0.21686568
    positions[2] += 0.55659138

    t1 = time.time()
    rep = normalize_positions(positions, n_atoms)
    t2 = time.time()
    print(rep)
    print("Time:", t2 - t1)

    positions = np.random.rand(3, 16)
    t1 = time.time()
    rep = normalize_positions(positions, n_atoms)
    t2 = time.time()
    print(rep)
    print("Time:", t2 - t1)

    n_atoms = [8, 8]
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.46, 0.0, 0.25],
            [0.96, 0.0, 0.25],
            [0.42, 0.0, 0.5],
            [0.92, 0.0, 0.5],
            [0.21, 0.0, 0.75],
            [0.71, 0.0, 0.75],
            [0.25, 0.5, 0.0],
            [0.75, 0.5, 0.0],
            [0.21, 0.5, 0.25],
            [0.71, 0.5, 0.25],
            [0.17, 0.5, 0.5],
            [0.67, 0.5, 0.5],
            [0.46, 0.5, 0.75],
            [0.96, 0.5, 0.75],
        ]
    ).T

    t1 = time.time()
    rep = normalize_positions(positions, n_atoms)
    t2 = time.time()
    print(rep)
    print("Time:", t2 - t1)
