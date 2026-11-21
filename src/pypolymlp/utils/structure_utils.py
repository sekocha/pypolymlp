"""Utility functions for modifying structure."""

import copy
from typing import Optional

import numpy as np

from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.utils.structure_core_utils import reset_types


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
    eps_array: Optional[np.ndarray] = None,
    n_eps: float = 10,
):
    """Return structures with sequential isotropic volume changes."""
    if eps_array is None:
        eps_array = np.linspace(eps_min, eps_max, n_eps)
    return [isotropic_volume_change(st, eps=eps) for eps in eps_array]


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


def introduce_disp(st: PolymlpStructure, eps: float = 0.001):
    """Introduce small random displacements."""
    shape = st.positions.shape
    disp = (2 * eps) * (np.random.random(shape) - 0.5)
    st.positions += disp
    return st


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
    st = reset_types(st)
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
    st = reset_types(st)
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
