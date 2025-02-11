"""Tests of polynomial MLP development using API"""

from pathlib import Path

import pytest

from pypolymlp.api.pypolymlp_calc import PypolymlpCalc
from pypolymlp.mlp_dev.pypolymlp import Pypolymlp

cwd = Path(__file__).parent


def test_load_mlp():
    """Test for loading polymlp files."""
    # Parse polymlp.lammps.pair
    filename = cwd / "polymlp.lammps.pair"
    coeff_true = 9.352307613515078e00 / 2.067583465937491e-01

    mlp = Pypolymlp()
    with open(filename, "rt") as fp:
        mlp.load_mlp(fp)
    assert mlp.coeffs[0] == pytest.approx(coeff_true, rel=1e-8)

    mlp = Pypolymlp()
    mlp.load_mlp(filename)
    assert mlp.coeffs[0] == pytest.approx(coeff_true, rel=1e-8)

    mlp_calc = PypolymlpCalc(pot=filename)
    coeffs = mlp_calc._prop.prop._coeffs
    assert coeffs[0] == pytest.approx(coeff_true, rel=1e-8)

    # Parse polymlp.yaml.gtinv
    filename = cwd / "polymlp.yaml.gtinv"
    coeff_true = 5.794375827500248e01

    mlp = Pypolymlp()
    with open(filename, "rt") as fp:
        mlp.load_mlp(fp)
    assert mlp.coeffs[0] == pytest.approx(coeff_true, rel=1e-8)

    mlp = Pypolymlp()
    mlp.load_mlp(filename)
    assert mlp.coeffs[0] == pytest.approx(coeff_true, rel=1e-8)

    mlp_calc = PypolymlpCalc(pot=filename)
    coeffs = mlp_calc._prop.prop._coeffs
    assert coeffs[0] == pytest.approx(coeff_true, rel=1e-8)


# def test_load_mlps_hybrid():
#     """Test for loading hybrid polymlp files."""
#     pass
