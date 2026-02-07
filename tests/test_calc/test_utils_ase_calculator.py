"""Tests of ASE calculator using polymlp."""

import numpy as np
import pytest
from ase.build import bulk

from pypolymlp.calculator.utils.ase_calculator import PolymlpASECalculator

atoms_fcc = bulk("Al", "fcc", a=4.03)


def test_PolymlpASECalculator(unitcell_mlp_Al):
    """Test PolymlpASECalculator."""
    _, pot = unitcell_mlp_Al
    calc = PolymlpASECalculator(pot=pot)
    calc = PolymlpASECalculator(require_mlp=False)
    calc.set_calculator(pot=pot)

    calc.calculate(atoms_fcc)
    assert calc.results["energy"] == pytest.approx(-3.4276371758140347)
    assert np.sum(calc.results["forces"]) == pytest.approx(0.0)
    np.testing.assert_allclose(
        calc.results["stress"][:3], 0.05306918446864798, atol=1e-7
    )
    np.testing.assert_allclose(calc.results["stress"][3:], 0.0, atol=1e-7)
