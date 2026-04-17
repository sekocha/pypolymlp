"""Utility functions for symfc."""

from typing import Optional

import numpy as np
from symfc.basis_sets.basis_sets_O1 import FCBasisSetO1
from symfc.utils.utils import SymfcAtoms

from pypolymlp.core.data_format import PolymlpStructure


def structure_to_symfc_cell(structure: PolymlpStructure) -> SymfcAtoms:
    """Convert PolymlpStructure to SymfcAtoms."""
    symfc_cell = SymfcAtoms(
        numbers=structure.types,
        cell=structure.axis.T,
        scaled_positions=structure.positions.T,
    )
    return symfc_cell


def construct_basis_cartesian(cell: PolymlpStructure) -> np.ndarray:
    """Generate a basis set for atomic positions in Cartesian coordinates."""
    cell_symfc = structure_to_symfc_cell(cell)
    try:
        fc_basis = FCBasisSetO1(cell_symfc, use_mkl=False).run()
    except ValueError:
        return None
    return fc_basis.full_basis_set.toarray()


def construct_basis_fractional_coordinates(cell: PolymlpStructure) -> np.ndarray:
    """Generate a basis set for atomic positions in fractional coordinates."""
    basis_c = construct_basis_cartesian(cell)
    if basis_c is None or basis_c.size == 0:
        return None
    basis_f = _basis_cartesian_to_fractional_coordinates(basis_c, cell)
    return basis_f


def _basis_cartesian_to_fractional_coordinates(
    basis_c: np.ndarray,
    unitcell: PolymlpStructure,
) -> np.ndarray:
    """Convert basis set in Cartesian coord. to basis set in fractional coordinates."""
    n_basis = basis_c.shape[1]
    n_atom = len(unitcell.elements)
    unitcell.axis_inv = np.linalg.inv(unitcell.axis)

    basis_c = np.array([b.reshape((n_atom, 3)) for b in basis_c.T])
    basis_c = basis_c.transpose((2, 1, 0))
    basis_c = basis_c.reshape(3, -1)
    basis_f = unitcell.axis_inv @ basis_c
    basis_f = basis_f.reshape((3, n_atom, n_basis))
    basis_f = basis_f.transpose((1, 0, 2)).reshape(-1, n_basis)
    basis_f, _, _ = np.linalg.svd(basis_f, full_matrices=False)
    return basis_f


def compute_projector_cartesian(cell: PolymlpStructure) -> np.ndarray:
    """Compute projector for cartesian components."""
    three_n = len(cell.elements) * 3
    try:
        basis = construct_basis_cartesian(cell)
        if len(basis) == 0:
            return np.zeros((three_n, three_n))
        return basis @ basis.T
    except:
        return np.zeros((three_n, three_n))


def set_symfc_cutoffs(
    cutoff_fc2: Optional[float] = None,
    cutoff_fc3: Optional[float] = None,
    cutoff_fc4: Optional[float] = None,
):
    """Set cutoffs used in symfc."""
    cutoff = {2: cutoff_fc2, 3: cutoff_fc3, 4: cutoff_fc4}
    return cutoff
