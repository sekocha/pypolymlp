"""Tests of utility functions for MD."""

from pathlib import Path

import numpy as np
import pytest

from pypolymlp.calculator.md.md_utils import (
    calc_integral,
    calculate_fc2_free_energy,
    get_p_roots,
)

# from pypolymlp.calculator.md.ase_md import IntegratorASE
# from pypolymlp.calculator.utils.ase_calculator import PolymlpASECalculator
# from pypolymlp.calculator.utils.ase_calculator_ref import (
#     PolymlpFC2ASECalculator,
#     PolymlpGeneralRefASECalculator,
#     PolymlpRefASECalculator,
# )
# from pypolymlp.calculator.utils.ase_utils import structure_to_ase_atoms
# from pypolymlp.calculator.utils.fc_utils import load_fc2_hdf5

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/"


def test_get_p_roots():
    """Test get_p_roots."""
    x, w = get_p_roots(n=5, a=0, b=1)
    np.testing.assert_allclose(x, [0.04691008, 0.23076534, 0.5, 0.76923466, 0.95308992])
    np.testing.assert_allclose(
        w, [0.23692689, 0.47862867, 0.56888889, 0.47862867, 0.23692689]
    )


def test_calc_integral():
    """Test calc_integral using Gauss-Legendre quadrature."""
    x, w = get_p_roots(n=5, a=0.0, b=1.0)
    f = x**2
    val = calc_integral(w, f, a=0.0, b=1.0)
    assert val == pytest.approx(1 / 3)


def test_calculate_fc2_free_energy(unitcell_mlp_Al):
    """Test calculate_fc2_free_energy."""
    unitcell, pot = unitcell_mlp_Al
    fc2 = path_file + "others/fc2_Al_111.hdf5"
    free_energy = calculate_fc2_free_energy(unitcell, (1, 1, 1), fc2, temperature=700)
    assert free_energy == pytest.approx(-0.5991411348114344)


# TODO: Add tests for thermodynamic integration
#    atoms_fcc = structure_to_ase_atoms(unitcell)
#    calc = PolymlpASECalculator(pot=pot)
