"""Class for handling atomic displacements."""

import copy

import numpy as np

from pypolymlp.core.data_format import PolymlpStructure


def convert_disps_to_positions(disps, axis, positions):
    """Convert displacements into positions.

    Parameters
    ----------
    disps: Displacements, shape=(n_str, 3, n_atoms)
    axis: Axis of base structure.
    positions: Positions of base structure.
    """
    assert disps.shape[1] == 3
    axis_inv = np.linalg.inv(axis)
    positions_all = np.array([positions + (axis_inv @ d) for d in disps])
    return positions_all


def get_structures_from_multiple_positions(
    positions_all: np.ndarray,
    structure: PolymlpStructure,
) -> list[PolymlpStructure]:
    """Convert positions into structures.

    Parameters
    ----------
    positions_all: Positions, shape=(n_str, 3, n_atom).
    structure: Base structure.
    """
    structures = []
    for i, positions_iter in enumerate(positions_all):
        st = copy.deepcopy(structure)
        st.positions = positions_iter
        st.name = "disp-" + str(i + 1)
        structures.append(st)
    return structures


def get_structures_from_displacements(
    disps: np.ndarray,
    structure: PolymlpStructure,
) -> list[PolymlpStructure]:
    """Convert displacements into structures.

    Parameters
    ----------
    disps: Displacements, shape=(n_str, 3, n_atoms)
    structure: Base structure.

    Retrun
    ------
    List of structures with displacements.
    """
    positions_all = convert_disps_to_positions(
        disps,
        structure.axis,
        structure.positions,
    )
    return get_structures_from_multiple_positions(positions_all, structure)


def generate_random_const_displacements(
    structure: PolymlpStructure,
    n_samples: int = 100,
    displacements: float = 0.03,
    is_plusminus: bool = False,
) -> tuple[np.ndarray, list[PolymlpStructure]]:
    """Generate a set of structures including random displacements

    Parameters
    ----------
    structure: Base structure.

    Return
    ------
    disps: Displacements, shape=(n_str, 3, n_atoms)
    structures: Structures, shape=(n_str)
    """
    disps = []
    for i in range(n_samples):
        rand = np.random.randn(3, structure.positions.shape[1])
        rand = rand / np.linalg.norm(rand, axis=0)
        disps.append(rand * displacements)
        if is_plusminus:
            disps.append(-rand * displacements)
    disps = np.array(disps)
    structures = get_structures_from_displacements(disps, structure)
    return disps, structures


def convert_positions_to_disps(st_disps: PolymlpStructure, st_origin: PolymlpStructure):
    """Transform structure into displacements.

    Return
    ------
    disps: Displacements, shape=(3, n_atoms)
    """
    diff = st_disps.positions - st_origin.positions
    diff -= np.rint(diff)
    disps = st_origin.axis @ diff
    return disps
