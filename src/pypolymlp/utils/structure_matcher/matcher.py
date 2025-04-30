"""Utilities for structure matchers"""

import itertools

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
    candidates = np.array(
        [
            np.sort(round_positions(pos + shift, tol=tol, decimals=decimals))
            for shift in center_shifts
        ]
    )
    rep_id = np.lexsort(candidates.T[::-1])[0]
    diff = candidates - candidates[rep_id]
    rep_ids = np.where(np.linalg.norm(diff, axis=1) < tol)[0]
    return center_shifts[rep_ids]


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
    rpositions = round_positions(positions, tol=tol, decimals=decimals)
    centers = []
    for i, pos1d in enumerate(rpositions):
        center_trans = _normalize_positions_1d(pos1d, tol=tol, decimals=decimals)
        centers.append(center_trans)

    rpositions -= np.tile(np.average(rpositions, axis=1), (rpositions.shape[1], 1)).T

    cands = []
    for t in itertools.product(*centers):
        t = np.array(t)
        pos = rpositions + np.tile(t, (rpositions.shape[1], 1)).T
        rpos = round_positions(pos, tol=tol, decimals=decimals)
        rpos = _sort_positions(rpos, n_atoms)
        cands.append(tuple(rpos.reshape(-1)))

    positions_irrep = np.array(min(cands)).reshape(3, -1)

    return positions_irrep


def _sort_positions(positions: np.array, n_atoms: np.ndarray):
    """Sort positions with respect to fractional coordinate numbers."""
    ibegin = 0
    for n in n_atoms:
        iend = ibegin + n
        ids = np.lexsort(positions[:, ibegin:iend][::-1]) + ibegin
        positions[:, ibegin:iend] = positions[:, ids]
        ibegin = iend
    return positions


#
# if __name__ == '__main__':
# #    pos1d_a = np.array([0, 0.5, 0.46, 0.96, 0.42, 0.92, 0.21, 0.71])
# #    pos1d_c = np.array([0, 0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75])
# #    _normalize_positions_1d(pos1d_a)
# #    _normalize_positions_1d(pos1d_c)
# #    pos1d_a += 0.11686568
# #    pos1d_c += 0.55659138
# #    _normalize_positions_1d(pos1d_a)
# #    _normalize_positions_1d(pos1d_c)
#
# #    pos1d_a += 0.3
# #    pos1d_c += 0.3
# #    _normalize_positions_1d(pos1d_a)
# #    _normalize_positions_1d(pos1d_c)
# #    pos1d_a += 0.6
# #    pos1d_c += 0.6
# #    _normalize_positions_1d(pos1d_a)
# #    _normalize_positions_1d(pos1d_c)
#
#     n_atoms = [8]
#     positions = np.array([
#         [0.  , 0. , 0. ],
#         [0.5 , 0. , 0. ],
# #        [0.25, 0.5, 0. ],
# #        [0.75, 0.5, 0. ],
#         [0.46, 0. , 0.25],
#         [0.96, 0. , 0.25],
# #        [0.21, 0.5, 0.25],
# #        [0.71, 0.5, 0.25],
#         [0.42, 0. , 0.5 ],
#         [0.92, 0. , 0.5 ],
# #        [0.17, 0.5, 0.5 ],
# #        [0.67, 0.5, 0.5 ],
#         [0.21, 0. , 0.75],
#         [0.71, 0. , 0.75],
# #        [0.46, 0.5, 0.75],
# #        [0.96, 0.5, 0.75],
#     ]).T
#
#     positions[0] += 0.11686568
#     positions[2] += 0.55659138
#     normalize_positions(positions, n_atoms)
