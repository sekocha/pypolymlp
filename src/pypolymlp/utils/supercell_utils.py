"""Utility functions for generating supercell."""

import copy
from typing import Optional, Union

import numpy as np

from pypolymlp.core.data_format import PolymlpStructure


def _is_diagonal(a: np.array):
    return (
        a.ndim == 2
        and a.shape[0] == a.shape[1]
        and np.all(a == np.diag(np.diagonal(a)))
    )


def _refine_positions(positions: np.ndarray, tol: float = 1e-13):
    """Refine fractional coordinates."""
    positions -= np.floor(positions)
    positions[np.where(positions > 1 - tol)] -= 1.0
    return positions


def _sort_wrt_types(st: PolymlpStructure, return_ids: bool = False):
    """Sort atoms with respect to types."""
    # TODO: Unify sort_wrt_types with the same function in structure_utils.
    map_elements = dict()
    for t, e in zip(st.types, st.elements):
        map_elements[t] = e

    n_atoms, positions, types = [], [], []
    ids_all = []
    for i in sorted(set(st.types)):
        ids = np.array(st.types) == i
        n_atoms.append(np.count_nonzero(ids))
        positions.extend(st.positions.T[ids])
        types.extend(np.array(st.types)[ids])
        ids_all.extend(np.where(ids == True)[0])

    st.positions = np.array(positions).T
    st.n_atoms = n_atoms
    st.types = types
    st.elements = [map_elements[t] for t in types]
    if return_ids:
        return st, np.array(ids_all)
    return st


def get_supercell(
    st: PolymlpStructure,
    supercell_matrix: np.ndarray,
    use_phonopy: bool = False,
) -> PolymlpStructure:
    """Construct supercell for a given supercell matrix.

    Parameters
    ----------
    st: Unitcell.
    supercell_matrix: Supercell matrix of shape (3, 3)
                      or supercell vector composed of three diagonal elements.
    use_phonopy: Use phonopy supercell algorithm.
                 This is activated if the supercell matrix is diagonal.
                 If the supercell matrix is non-diagonal,
                 phonopy algorithm is automatically used.
    """
    size = np.array(supercell_matrix).astype(int)
    if size.shape == (3, 3):
        if not _is_diagonal(size):
            from pypolymlp.utils.phonopy_utils import phonopy_supercell

            sup = phonopy_supercell(st, supercell_matrix=size, return_phonopy=False)
            sup.positions = _refine_positions(sup.positions)
            return sup

        return _get_supercell_diagonal(st, np.diag(size), use_phonopy=use_phonopy)
    elif size.shape == (3,):
        return _get_supercell_diagonal(st, size, use_phonopy=use_phonopy)

    raise RuntimeError("Supercell size not appropriate.")


def _get_supercell_diagonal(
    st: PolymlpStructure,
    size: tuple = (2, 2, 2),
    use_phonopy: bool = False,
) -> PolymlpStructure:
    """Construct supercell for a diagonal supercell matrix."""
    if use_phonopy:
        from pypolymlp.utils.phonopy_utils import phonopy_supercell

        return phonopy_supercell(st, supercell_diag=size, return_phonopy=False)

    supercell_matrix = np.diag(size)
    n_expand = np.prod(size)

    sup = copy.deepcopy(st)
    sup.axis = st.axis @ supercell_matrix
    sup.n_atoms = np.array(st.n_atoms) * n_expand
    sup.types = np.repeat(st.types, n_expand)
    sup.elements = np.repeat(st.elements, n_expand)
    sup.volume = st.volume * n_expand
    sup.supercell_matrix = supercell_matrix

    # Loop sequence: z -> y -> x, which is compatible with phonopy
    nx, ny, nz = size
    trans_all = np.indices((nz, ny, nx)).reshape(3, -1).T
    trans_all = trans_all[:, [2, 1, 0]]
    positions_new = (st.positions.T[:, None] + trans_all[None, :]).reshape((-1, 3))
    sup.positions = (positions_new / size).T
    sup.positions = _refine_positions(sup.positions)
    return sup


def get_supercell_size(supercell_matrix: Union[np.array, list, tuple]):
    """Return number of unitcells from supercell matrix or its diagonal elements."""
    mat = np.array(supercell_matrix)
    if mat.size == 3:
        return np.prod(supercell_matrix)
    elif mat.shape == (3, 3):
        return int(round(np.linalg.det(supercell_matrix)))
    raise RuntimeError("Inappropriate supercell matrix.")


def _triangularize(supercell: PolymlpStructure):
    """Triangularize axis."""
    Q, R = np.linalg.qr(supercell.axis)
    s = np.sign(np.diag(R))
    s[s == 0] = 1
    D = np.diag(s)
    Q = Q @ D
    R = D @ R
    supercell.axis = R
    return supercell


def get_supercell_three_directions(
    st: PolymlpStructure,
    direction1: tuple = (1, 0, 0),
    direction2: tuple = (0, 1, 0),
    direction3: tuple = (0, 0, 1),
    n_layers: int = 2,
    supercell_matrix: Optional[np.ndarray] = None,
):
    """Set supercell using three directions."""
    if supercell_matrix is not None:
        if np.array(supercell_matrix).shape != (3, 3):
            raise RuntimeError("Supercell matrix shape is not (3, 3).")
        matrix = copy.deepcopy(supercell_matrix)
    else:
        if len(direction1) != 3:
            raise RuntimeError("Three elements required for disp1.")
        if len(direction2) != 3:
            raise RuntimeError("Three elements required for disp2.")
        if len(direction3) != 3:
            raise RuntimeError("Three elements required for slip plane.")
        matrix = np.zeros((3, 3), dtype=int)
        matrix[:, 0] = np.array(direction1)
        matrix[:, 1] = np.array(direction2)
        matrix[:, 2] = np.array(direction3) * n_layers

    for i in range(3):
        if matrix[i, i] < 0:
            matrix[:, i] *= -1

    supercell = get_supercell(st, matrix)
    supercell = _triangularize(supercell)
    return supercell


def get_slab(
    st: PolymlpStructure,
    direction1: tuple = (1, 0, 0),
    direction2: tuple = (0, 1, 0),
    direction3: tuple = (0, 0, 1),
    n_layers: int = 2,
    supercell_matrix: Optional[np.ndarray] = None,
    vacuum_width: float = 10.0,
    end_frac: Optional[float] = None,
    tol: float = 1e-13,
):
    """Set slab supercell model using three directions."""
    supercell = get_supercell_three_directions(
        st,
        direction1=direction1,
        direction2=direction2,
        direction3=direction3,
        n_layers=n_layers,
        supercell_matrix=supercell_matrix,
    )
    len3 = np.linalg.norm(supercell.axis[:, 2])
    ratio = (len3 + vacuum_width) / len3
    if end_frac is None:
        supercell.axis[:, 2] *= ratio
        supercell.positions[2] /= ratio
        return supercell

    supercell.positions[2] += end_frac
    supercell.positions = _refine_positions(supercell.positions)
    match_end = np.abs(supercell.positions[2]) < tol

    add_positions = supercell.positions[:, match_end]
    add_positions[2] += 1.0
    add_elements = np.array(supercell.elements)[match_end]
    add_types = np.array(supercell.types)[match_end]

    supercell.positions = np.hstack([supercell.positions, add_positions])
    supercell.elements = np.concatenate([supercell.elements, add_elements])
    supercell.types = np.concatenate([supercell.types, add_types])
    supercell = _sort_wrt_types(supercell)

    supercell.axis[:, 2] *= ratio
    supercell.positions[2] /= ratio
    return supercell
