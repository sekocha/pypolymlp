"""Tests of atomic energies."""

from pathlib import Path

import numpy as np

from pypolymlp.utils.atomic_energies.atomic_energies import (
    get_atomic_energies,
    get_atomic_energies_polymlp_in,
    get_elements,
)

cwd = Path(__file__).parent


def test_get_elements():
    """Test for get_elements."""
    get_elements("Al2O3") == ["Al", "O"]
    get_elements("MgO") == ["Mg", "O"]


def test_get_atomic_energies():
    """Test for get_atomic_energies."""
    energies, _ = get_atomic_energies(elements=["Ag", "Au"])
    np.testing.assert_allclose(energies, [-0.19820116, -0.18494148], atol=1e-6)

    energies, _ = get_atomic_energies(formula="Al2O3")
    np.testing.assert_allclose(energies, [-0.31455471, -1.85321219], atol=1e-6)

    energies, _ = get_atomic_energies(formula="Al2O3", functional="PBEsol")
    np.testing.assert_allclose(energies, [-0.28161711, -1.79668297], atol=1e-6)

    get_atomic_energies_polymlp_in(elements=["Ag", "Au"])
