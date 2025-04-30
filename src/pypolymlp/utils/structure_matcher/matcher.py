"""Utilities for structure matchers"""

import numpy as np


def round_positions(positions: np.ndarray, tol: float = 1e-13, decimals: int = 5):
    """Round fractional coordinates of positions (-0.5 <= p < 0.5)."""
    positions_rint = positions - np.rint(positions)
    positions_rint[np.abs(positions_rint - 0.5) < tol] = -0.5
    positions_rint = np.round(positions_rint, decimals)
    return positions_rint


def _normalize_positions_1d(pos1d: np.ndarray, tol: float = 1e-13, decimals: int = 5):
    """Get an irreducible representation of one-dimensional fractional coordinates."""
    pos = pos1d - np.average(pos1d)
    natom = pos.shape[0]
    center_shifts = np.arange(0.0, 1.0, 1.0 / natom)
    candidates = [
        np.sort(round_positions(pos + shift, tol=tol, decimals=decimals))
        for shift in center_shifts
    ]
    rep_id = np.lexsort(np.array(candidates).T[::-1])[0]
    return round_positions(pos + center_shifts[rep_id], tol=tol, decimals=decimals)


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
    positions_irrep = np.zeros(positions.shape)

    rpositions = round_positions(positions, tol=tol, decimals=decimals)
    for i, pos1d in enumerate(rpositions):
        positions_irrep[i] = _normalize_positions_1d(pos1d, tol=tol, decimals=decimals)

    ibegin = 0
    for n in n_atoms:
        iend = ibegin + n
        ids = np.lexsort(positions_irrep[:, ibegin:iend][::-1]) + ibegin
        positions_irrep[:, ibegin:iend] = positions_irrep[:, ids]
        ibegin = iend

    return positions_irrep


if __name__ == "__main__":

    n_atoms = [3, 1]
    positions1 = np.array(
        [
            [0.5, 0.5, 0.5, 0.6],
            [0.25, 0.25, 0.3, 0.7],
            [0.1, 0.2, 0.8, 0.9],
        ]
    )

    #    positions2 = np.array([
    #        [0.0, 0.2, 0.6],
    #        [0.25, 0.25, 0.7],
    #        [0.1, 0.2, 0.9],
    #        #[0.5, 0.6, 0.3],
    #    ])
    positions_irrep = normalize_positions(positions1, n_atoms, decimals=5)
    print(positions_irrep)
