"""Utility functions for generating supercell."""

import copy

import numpy as np

from pypolymlp.core.data_format import PolymlpStructure


def _is_diagonal(a: np.array):
    return (
        a.ndim == 2
        and a.shape[0] == a.shape[1]
        and np.all(a == np.diag(np.diagonal(a)))
    )


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
    size = np.array(supercell_matrix)
    if size.shape == (3, 3):
        if not _is_diagonal(size):
            from pypolymlp.utils.phonopy_utils import phonopy_supercell

            return phonopy_supercell(st, supercell_matrix=size, return_phonopy=False)

        return _get_supercell_diagonal(st, size, use_phonopy=use_phonopy)
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

    trans_all = np.indices(size).reshape(3, -1).T
    positions_new = (st.positions.T[:, None] + trans_all[None, :]).reshape((-1, 3))
    sup.positions = (positions_new / size).T
    return sup
