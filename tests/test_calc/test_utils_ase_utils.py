"""Tests of utility functions for ASE calculations."""

from pathlib import Path

import numpy as np

from pypolymlp.calculator.utils.ase_utils import (
    ase_atoms_to_phonopy_atoms,
    ase_atoms_to_structure,
    phonopy_atoms_to_ase_atoms,
    structure_to_ase_atoms,
)

cwd = Path(__file__).parent


def test_structure_to_ase_atoms(structure_rocksalt):
    """Test structure_to_ase_atoms and ase_atoms_to_structure."""
    ase_atoms = structure_to_ase_atoms(structure_rocksalt)
    st = ase_atoms_to_structure(ase_atoms)
    np.testing.assert_allclose(structure_rocksalt.positions, st.positions)
    np.testing.assert_equal(st.elements, ["Mg"] * 4 + ["O"] * 4)


def test_phonopy_to_ase_atoms(structure_rocksalt):
    """Test phonopy_atoms_to_ase_atoms and ase_atoms_to_phonopy_atoms."""
    ase_atoms = structure_to_ase_atoms(structure_rocksalt)
    phonopy_atoms = ase_atoms_to_phonopy_atoms(ase_atoms)
    ase_atoms = phonopy_atoms_to_ase_atoms(phonopy_atoms)
    st = ase_atoms_to_structure(ase_atoms)
    np.testing.assert_allclose(structure_rocksalt.positions, st.positions)
    np.testing.assert_equal(st.elements, ["Mg"] * 4 + ["O"] * 4)
