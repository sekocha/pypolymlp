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
    stresses = np.zeros(forces.shape[0] * 6)
    forces, structures = [], []
    for positions_iter, forces_iter in zip(positions_all, forces):
        st = copy.deepcopy(structure)
        st.positions = positions_iter
        if element_order is not None:
            st, forces_iter = permute_atoms(st, forces_iter, element_order)

        forces.extend(forces_iter.T.reshape(-1))
        structures.append(st)
    forces = np.array(forces)
    volumes = np.array([st.volume for st in structures])

    if element_order is not None:
        elements = element_order
    else:
        elements_rep = structure.elements
        elements = sorted(set(elements_rep), key=elements_rep.index)

    total_n_atoms = np.array([sum(st.n_atoms) for st in structures])
    files = ["disp-" + str(i + 1).zfill(5) for i, _ in enumerate(structures)]

    dft = PolymlpDataDFT(
        energies,
        forces,
        stresses,
        volumes,
        structures,
        total_n_atoms,
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
    """
    axis_inv = np.linalg.inv(axis)
    positions_all = np.array([positions + (axis_inv @ d) for d in disps])
    return positions_all


def get_structures_from_multiple_positions(
    positions_all: np.ndarray,
    structure: PolymlpStructure,
) -> list[PolymlpStructure]:
    """positions_all: (n_str, 3, n_atom)"""
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
    """Generate a set of structures including random displacements"""
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
