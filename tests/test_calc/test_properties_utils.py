"""Tests of property utility functions."""

import numpy as np

from pypolymlp.calculator.utils.properties_utils import convert_stresses_in_gpa


def test_convert_stresses_in_gpa(unitcell_disp_pair_MgO):
    """Test convert_stresses_in_gpa."""
    unitcell, pot, prop = unitcell_disp_pair_MgO
    _, _, stresses = prop.eval(unitcell)

    stresses_true = [
        -0.17379186,
        -0.17521681,
        -0.16909401,
        0.00004465,
        0.00008926,
        0.00029751,
    ]

    stresses_gpa = convert_stresses_in_gpa(np.array([stresses]), [unitcell])[0]
    np.testing.assert_allclose(stresses_gpa, stresses_true, atol=1e-5)

    stresses2 = np.array([stresses, stresses])
    stresses_gpa = convert_stresses_in_gpa(stresses2, [unitcell, unitcell])
    np.testing.assert_allclose(stresses_gpa, [stresses_true, stresses_true], atol=1e-5)

    stresses_gpa = convert_stresses_in_gpa(stresses, unitcell)
    np.testing.assert_allclose(stresses_gpa, stresses_true, atol=1e-5)
