"""Utility functions for refining structure.

This file contains functions that do not change structure essentially.
"""

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


def replace_types(
    st: PolymlpStructure,
    elements_string: list[str],
) -> PolymlpStructure:
    """Replace atom types used for property and feature calculations.

    Atom types are replaced based on element strings.
    """

    st.elements = np.array(st.elements)
    st.types = np.array(st.types)
    count = 0
    if len(elements_string) == len(np.unique(elements_string)):
        for atomtype, ele in enumerate(elements_string):
            match = st.elements == ele
            st.types[match] = atomtype
            count += np.count_nonzero(match)
    else:
        atomtype = 0
        types = np.zeros(st.types.shape, dtype=int)
        unique_elements = list(dict.fromkeys(elements_string))
        for ele in unique_elements:
            match = st.elements == ele
            count += np.count_nonzero(match)

            match_ids = np.where(match)[0]
            idx = np.argsort(st.types[match])
            match_ids = match_ids[idx]
            types_local = st.types[match_ids]
            mapping = {v: i + atomtype for i, v in enumerate(np.unique(types_local))}
            types_local = np.array([mapping[t] for t in types_local])
            types[match_ids] = types_local

            atomtype += len(np.unique(types_local))
        st.types = types

    if count != st.elements.shape[0]:
        raise RuntimeError("Elements in structure not found in elements.")
    return st
