"""Utility functions for modifying structure."""

import copy
import itertools
import sys
from math import pi, sqrt
from typing import Optional

import numpy as np

from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.core.interface_vasp import Poscar


def refine_positions(st: PolymlpStructure, tol=1e-13):
    positions = st.positions
    positions -= np.floor(positions)
    positions[np.where(positions > 1 - tol)] -= 1.0
    st.positions = positions
    return st


def reset_types(st: PolymlpStructure):
    st.types = [i for i, n1 in enumerate(st.n_atoms) for n2 in range(n1)]
    return st


def calc_positions_cartesian(st: PolymlpStructure):
    return st.axis @ st.positions


def get_reciprocal_axis(
    st: Optional[PolymlpStructure] = None, axis: Optional[np.ndarray] = None
):
    if axis is None:
        axis = st.axis
    return 2 * pi * np.linalg.inv(axis).T


def disp(st: PolymlpStructure, eps: float = 0.001):
    shape = st.positions.shape
    disp = (2 * eps) * (np.random.rand(shape) - 0.5)
    st.positions += disp
    return st


def isotropic_volume_change(st: PolymlpStructure, eps: float = 1.0):
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

    axis = st.axis @ supercell_matrix
    n_atoms = np.array(st.n_atoms) * n_expand
    types = np.repeat(st.types, n_expand)
    elements = np.repeat(st.elements, n_expand)
    volume = st.volume * n_expand

    trans_all = np.array(
        list(itertools.product(range(size[0]), range(size[1]), range(size[2]))),
        dtype=float,
    )
    size = np.array(size, dtype=float)
    positions_new = []
    for pos in st.positions.T:
        pos_new = (pos + trans_all) / size
        positions_new.extend(pos_new)
    positions = np.array(positions_new).T

    supercell = PolymlpStructure(
        axis,
        positions,
        n_atoms,
        elements,
        types,
        volume,
        supercell_matrix=supercell_matrix,
    )
    return supercell


def remove(st: PolymlpStructure, idx: int):
    """idx-th element is removed from st_dict."""
    begin = int(np.sum(st.n_atoms[:idx]))
    end = begin + st.n_atoms[idx]
    st.positions = np.delete(st.positions, range(begin, end), axis=1)
    st.n_atoms = np.delete(st.n_atoms, idx)
    st.elements = np.delete(st.elements, range(begin, end))
    st.types = np.delete(st.types, range(begin, end))
    return st


def remove_atom(st: PolymlpStructure, idx: int):
    """idx-th atom is removed from st_dict."""
    st.positions = np.delete(st.positions, idx, axis=1)
    st.elements = np.delete(st.elements, idx)
    st.types = np.delete(st.types, idx)

    sum1 = np.cumsum(st.n_atoms)
    match = np.where(idx < sum1)[0][0]
    st.n_atoms[match] -= 1
    return st


def reorder(st: PolymlpStructure, order=None, index1=None, index2=None):

    if order is None:
        order = list(range(len(st.n_atoms)))
        order[index1], order[index2] = order[index2], order[index1]

    positions, n_atoms, types, elements = [], [], [], []
    for i in order:
        n_atoms.append(st.n_atoms[i])
        begin = int(np.sum(st.n_atoms[:i]))
        end = int(begin + st.n_atoms[i])
        positions.extend(st.positions.T[begin:end])
        elements.extend(st.elements[begin:end])
        types.extend(st.types[begin:end])

    st.n_atoms = np.array(n_atoms)
    st.elements = np.array(elements)
    st.types = np.array(types)
    st.positions = np.array(positions).T
    return st


def swap_elements(st: PolymlpStructure, order=None, index1=None, index2=None):
    """Orders of element and type are fixed.
    Orders of positions, n_atoms are swapped."""
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

    a = np.linalg.norm(st.axis[:, 0])
    b = np.linalg.norm(st.axis[:, 1])
    c = np.linalg.norm(st.axis[:, 2])
    calpha = np.dot(st.axis[:, 1], st.axis[:, 2]) / (b * c)
    cbeta = np.dot(st.axis[:, 2], st.axis[:, 0]) / (c * a)
    cgamma = np.dot(st.axis[:, 0], st.axis[:, 1]) / (a * b)

    return a, b, c, calpha, cbeta, cgamma


def triangulation_axis(st: PolymlpStructure):

    a, b, c, calpha, cbeta, cgamma = get_lattice_constants(st)

    lx = a
    xy = b * cgamma
    xz = c * cbeta
    ly = sqrt(b * b - xy * xy)
    yz = (b * c * calpha - xy * xz) / ly
    lz = sqrt(c * c - xz * xz - yz * yz)

    tri_axis = np.array([[lx, xy, xz], [0, ly, yz], [0, 0, lz]])
    st.axis = tri_axis
    return st


def random_deformation(st: PolymlpStructure, max_deform: float = 0.1):
    rand = (np.random.rand(3, 3) - 0.5) * 2 * max_deform
    st.axis += rand
    return st


def multiple_random_deformation(
    structures: list[PolymlpStructure],
    max_deform: float = 0.1,
):
    n_st = len(structures)
    random_deform = (np.random.rand(n_st, 3, 3) - 0.5) * 2 * max_deform
    for st, rand in zip(structures, random_deform):
        st.axis += rand
    return structures


if __name__ == "__main__":

    from pypolymlp.utils.vasp_utils import print_poscar, write_poscar_file

    st = Poscar(sys.argv[1]).structure

    st = supercell_diagonal(st)
    # st_dict = remove(st_dict, 1)
    st = reorder(st, order=[1, 0])
    # st = swap_elements(st, order=[1,0])
    print_poscar(st)

    st = Poscar(sys.argv[1]).structure
    structures = multiple_isotropic_volume_changes(
        st, eps_min=0.7, eps_max=2.0, n_eps=15
    )

    for i, st in enumerate(structures):
        write_poscar_file(st, filename="POSCAR-" + str(i + 1).zfill(3))
