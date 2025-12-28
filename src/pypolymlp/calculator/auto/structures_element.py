"""Functions for providing initial structures for systematic calculations."""

import numpy as np

from pypolymlp.calculator.auto.dataclass import Prototype
from pypolymlp.core.data_format import PolymlpStructure


def _get_structure_attrs(n_atom: int, element_strings: tuple):
    """Return structure attributes for elemental system."""
    n_atoms = [n_atom]
    types = np.zeros(n_atom, dtype=int)
    elements = [ele for n, ele in zip(n_atoms, element_strings) for _ in range(n)]
    return (n_atoms, types, elements)


def get_structure_list_element(element_strings: tuple):
    """Return structure list for elemental systems."""
    fcc = structure_fcc(element_strings, a=5.0)
    bcc = structure_bcc(element_strings, a=4.0)
    return [fcc, bcc]


def structure_fcc(element_strings: tuple, a: float = 5.0):
    """Return FCC structure."""
    axis = np.eye(3) * a
    positions = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]
    ).T
    n_atoms, types, elements = _get_structure_attrs(positions.shape[1], element_strings)

    fcc = PolymlpStructure(
        axis=axis,
        positions=positions,
        n_atoms=n_atoms,
        types=types,
        elements=elements,
    )
    return Prototype(fcc, "fcc", 52914, 4, (4, 4, 4))


def structure_bcc(element_strings: tuple, a: float = 4.0):
    """Return BCC structure."""
    axis = np.eye(3) * a
    positions = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]).T
    n_atoms, types, elements = _get_structure_attrs(positions.shape[1], element_strings)

    fcc = PolymlpStructure(
        axis=axis,
        positions=positions,
        n_atoms=n_atoms,
        types=types,
        elements=elements,
    )
    return Prototype(fcc, "bcc", 76156, 2, (4, 4, 4))
