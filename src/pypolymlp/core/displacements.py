#!/usr/bin/env python

import copy

import numpy as np

from pypolymlp.core.data_format import PolymlpDataDFT, PolymlpStructure
from pypolymlp.core.utils import permute_atoms


def set_dft_data(
    forces: np.ndarray,
    energies: np.ndarray,
    positions_all: np.ndarray,
    structure: PolymlpStructure,
    element_order: list[str] = None,
) -> PolymlpDataDFT:
    """Generate DFT dataset dataclass from a force-position dataset.

    Parameters
    ----------
    forces: (n_str, 3, n_atom)
    energies: (n_str)
    positions_all: (n_str, 3, n_atom)
    structure: Structure

    Return
    ------
    dft: DFT training or test dataset in PolymlpDataDFT format
    """
    assert forces.shape[1] == 3
    assert positions_all.shape[1] == 3
    assert forces.shape[2] == positions_all.shape[2]
    assert forces.shape[0] == energies.shape[0] == positions_all.shape[0]

    stresses = np.zeros(forces.shape[0] * 6)
    forces_update, structures = [], []
    for positions_iter, forces_iter in zip(positions_all, forces):
        st = copy.deepcopy(structure)
        st.positions = positions_iter
        if element_order is not None:
            st, forces_iter = permute_atoms(st, forces_iter, element_order)

        forces_update.extend(forces_iter.T.reshape(-1))
        structures.append(st)
    forces = np.array(forces_update)
    volumes = np.array([st.volume for st in structures])

    if element_order is not None:
        elements = element_order
    else:
        elements_rep = structure.elements
        elements = sorted(set(elements_rep), key=elements_rep.index)

    total_n_atoms = np.array([sum(st.n_atoms) for st in structures])
    files = ["disp-" + str(i + 1).zfill(5) for i, _ in enumerate(structures)]

    dft = PolymlpDataDFT(
        energies=energies,
        forces=forces,
        stresses=stresses,
        volumes=volumes,
        structures=structures,
        total_n_atoms=total_n_atoms,
        files=files,
        elements=elements,
        include_force=True,
        weight=1.0,
    )
    return dft


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
    for positions_iter in positions_all:
        st = copy.deepcopy(structure)
        st.positions = positions_iter
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
