"""Tests of phonopy utility functions."""

from pathlib import Path

import numpy as np

from pypolymlp.utils.phonopy_utils import (
    phonopy_cell_to_structure,
    phonopy_supercell,
    structure_to_phonopy_cell,
)

cwd = Path(__file__).parent


def test_cell(structure_rocksalt):
    """Test structure converter."""
    cell = structure_to_phonopy_cell(structure_rocksalt)
    st = phonopy_cell_to_structure(cell)
    np.testing.assert_allclose(st.axis, structure_rocksalt.axis)
    np.testing.assert_allclose(st.positions, structure_rocksalt.positions)
    assert st.elements == structure_rocksalt.elements
    assert st.types == structure_rocksalt.types
    assert st.n_atoms == structure_rocksalt.n_atoms


def test_supercell(structure_rocksalt):
    """Test supercell in phonopy."""
    supercell_matrix = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 2]])
    supercell = phonopy_supercell(
        structure_rocksalt,
        supercell_matrix=supercell_matrix,
        return_phonopy=False,
    )
    np.testing.assert_equal(supercell_matrix, supercell.supercell_matrix)

    np.testing.assert_allclose(
        supercell.axis, [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [4.0, 0.0, 8.0]], atol=1e-6
    )
    np.testing.assert_allclose(supercell.positions[:, -1], [0.5, 0.5, 0.5], atol=1e-6)
    assert supercell.positions.shape == (3, 16)
    np.testing.assert_equal(supercell.n_atoms, [8, 8])
    np.testing.assert_equal(supercell.types, np.repeat([0, 1], 8))
    np.testing.assert_equal(supercell.elements, np.repeat(["Mg", "O"], 8))


# TODO: get_nac_params
