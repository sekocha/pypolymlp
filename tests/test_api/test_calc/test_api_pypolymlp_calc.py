"""Tests of polynomial MLP calculation using API"""

from pathlib import Path

import pytest

from pypolymlp.api.pypolymlp_calc import PypolymlpCalc

cwd = Path(__file__).parent
path_files = str(cwd) + "/../../files/"


def test_load_mlp():
    """Test for loading polymlp files."""
    # Parse polymlp.lammps.pair
    filename = cwd / "mlps/polymlp.lammps.pair"
    coeff_true = 9.352307613515078e00 / 2.067583465937491e-01

    mlp_calc = PypolymlpCalc(pot=filename)
    coeffs = mlp_calc._prop.prop._coeffs
    assert coeffs[0] == pytest.approx(coeff_true, rel=1e-8)

    # Parse polymlp.yaml.gtinv
    filename = cwd / "mlps/polymlp.yaml.gtinv"
    coeff_true = 5.794375827500248e01

    mlp_calc = PypolymlpCalc(pot=filename)
    coeffs = mlp_calc._prop.prop._coeffs
    assert coeffs[0] == pytest.approx(coeff_true, rel=1e-8)


# def test_load_mlps_hybrid():
#     """Test for loading hybrid polymlp files."""
#     pass
