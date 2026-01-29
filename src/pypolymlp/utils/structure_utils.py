"""Utility functions for modifying structure."""

import copy
from typing import Optional

import numpy as np

from pypolymlp.core.data_format import PolymlpStructure


def _refine_positions(positions: np.ndarray, tol: float = 1e-13):
    """Refine fractional coordinates."""
    positions -= np.floor(positions)
    positions[np.where(positions > 1 - tol)] -= 1.0
    return positions


def refine_positions(
    st: Optional[PolymlpStructure] = None,
    positions: Optional[np.ndarray] = None,
    tol: float = 1e-13,
):
    """Refine fractional coordinates."""
    if positions is not None:
        return _refine_positions(positions, tol=tol)

    st.positions = _refine_positions(st.positions, tol=tol)
    return st


def reset_types(st: PolymlpStructure):
    """Reset types using n_atoms."""
    st.types = [i for i, n1 in enumerate(st.n_atoms) for n2 in range(n1)]
    return st


def calc_positions_cartesian(st: PolymlpStructure):
    """Calculate Cartesian coordinates."""
    return st.axis @ st.positions


def get_reciprocal_axis(
    st: Optional[PolymlpStructure] = None,
    axis: Optional[np.ndarray] = None,
):
    """Calculate reciprocal axis."""
    if axis is None:
        axis = st.axis
    return 2 * np.pi * np.linalg.inv(axis).T


def introduce_disp(st: PolymlpStructure, eps: float = 0.001):
    """Introduce small random displacements."""
    shape = st.positions.shape
    disp = (2 * eps) * (np.random.random(shape) - 0.5)
    st.positions += disp
    return st


def isotropic_volume_change(st: PolymlpStructure, eps: float = 1.0):
    """Return structure with isotropic volume change."""
    eps1 = pow(eps, 0.3333333333333)
    st_vol = copy.deepcopy(st)
    st_vol.axis *= eps1
    st_vol.volume *= eps
    return st_vol


def multiple_isotropic_volume_changes(
    st: PolymlpStructure,
    eps_min: float = 0.7,
    eps_max: float = 2.0,
    n_eps: float = 10,
):
    """Return structures with sequential isotropic volume changes."""
    eps_array = np.linspace(eps_min, eps_max, n_eps)
    return [isotropic_volume_change(st, eps=eps) for eps in eps_array]


def supercell(
    st: PolymlpStructure,
    supercell_matrix: np.ndarray,
) -> PolymlpStructure:
    """Construct supercell for a given supercell matrix."""
    from pypolymlp.utils.phonopy_utils import phonopy_supercell

    return phonopy_supercell(
        st,
        supercell_matrix=supercell_matrix,
        return_phonopy=False,
    )


def supercell_diagonal(
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


def remove(st: PolymlpStructure, idx: int):
    """idx-th element is removed from st_dict."""
    begin = int(np.sum(st.n_atoms[:idx]))
    end = begin + st.n_atoms[idx]
    st.positions = np.delete(st.positions, range(begin, end), axis=1)
    st.n_atoms = np.delete(st.n_atoms, idx)
    st.elements = np.delete(st.elements, range(begin, end))
    st.types = np.delete(st.types, range(begin, end))
    if st.positions_cartesian is not None:
        st.positions_cartesian = np.delete(
            st.positions_cartesian, range(begin, end), axis=1
        )
    return st


def remove_atom(st: PolymlpStructure, idx: int):
    """idx-th atom is removed from st_dict."""
    st.positions = np.delete(st.positions, idx, axis=1)
    st.elements = np.delete(st.elements, idx)
    st.types = np.delete(st.types, idx)

    sum1 = np.cumsum(st.n_atoms)
    match = np.where(idx < sum1)[0][0]
    st.n_atoms[match] -= 1
    if st.positions_cartesian is not None:
        st.positions_cartesian = np.delete(st.positions_cartesian, idx, axis=1)
    return st


def swap_elements(
    st: PolymlpStructure,
    order: Optional[list] = None,
    index1: Optional[int] = None,
    index2: Optional[int] = None,
):
    """Swap positions and number of atoms.

    Orders of element and type are fixed.
    Orders of positions and n_atoms are swapped.
    """
    if order is None:
        order = list(range(len(st.n_atoms)))
        order[index1], order[index2] = order[index2], order[index1]

    uniq_ele = []
    for i, n in enumerate(st.n_atoms):
        begin = int(np.sum(st.n_atoms[:i]))
        uniq_ele.append(st.elements[begin])

    positions, n_atoms, types, elements = [], [], [], []
    for t, i in enumerate(order):
        n_atoms.append(st.n_atoms[i])
        begin = int(np.sum(st.n_atoms[:i]))
        end = int(begin + st.n_atoms[i])
        positions.extend(st.positions.T[begin:end])

        elements.extend([uniq_ele[t] for n in range(st.n_atoms[i])])
        types.extend([t for n in range(st.n_atoms[i])])

    st.n_atoms = np.array(n_atoms)
    st.elements = np.array(elements)
    st.types = np.array(types)
    st.positions = np.array(positions).T
    return st


def get_lattice_constants(st: PolymlpStructure):
    """Return lattice constants from axis."""
    a = np.linalg.norm(st.axis[:, 0])
    b = np.linalg.norm(st.axis[:, 1])
    c = np.linalg.norm(st.axis[:, 2])
    calpha = st.axis[:, 1] @ st.axis[:, 2] / (b * c)
    cbeta = st.axis[:, 2] @ st.axis[:, 0] / (c * a)
    cgamma = st.axis[:, 0] @ st.axis[:, 1] / (a * b)
    return a, b, c, calpha, cbeta, cgamma


def triangulation_axis(st: PolymlpStructure):
    """Return structure with axis in triangle form."""
    a, b, c, calpha, cbeta, cgamma = get_lattice_constants(st)

    lx = a
    xy = b * cgamma
    xz = c * cbeta
    ly = np.sqrt(b * b - xy * xy)
    yz = (b * c * calpha - xy * xz) / ly
    lz = np.sqrt(c * c - xz * xz - yz * yz)

    tri_axis = np.array([[lx, xy, xz], [0, ly, yz], [0, 0, lz]])
    st.axis = tri_axis
    return st


def random_deformation(st: PolymlpStructure, max_deform: float = 0.1):
    """Deform structure randomly."""
    rand = (np.random.rand(3, 3) - 0.5) * 2 * max_deform
    st.axis += rand
    return st


def multiple_random_deformation(
    structures: list[PolymlpStructure],
    max_deform: float = 0.1,
):
    """Deform structure randomly."""
    n_st = len(structures)
    random_deform = (np.random.rand(n_st, 3, 3) - 0.5) * 2 * max_deform
    for st, rand in zip(structures, random_deform):
        st.axis += rand
    return structures


def sort_wrt_types(st: PolymlpStructure, return_ids: bool = False):
    """Sort atoms with respect to types."""
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
